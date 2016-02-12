from philharmonic.scheduler.ischeduler import IScheduler
from philharmonic import Schedule, Migration, VMRequest
from philharmonic.logger import info, debug, error
from philharmonic import conf

import philharmonic as ph
import pandas as pd
import numpy as np

class BCDScheduler(IScheduler):
    """Best cost decreasing scheduler. It determines the cheapest location based on 
    current and future energy prices and decides whether a job should be migrated 
    based on remaining execution length while taking into account possible SLA penalties.  

    """
    def __init__(self, cloud=None, driver=None):
        IScheduler.__init__(self, cloud, driver)

    def _fits(self, vm, server):
        """Returns the utilisation of adding vm to server
        or -1 in case some resource's capacity is exceeded.

        """
        #TODO: this method should probably be a part of Cloud
        current = self.cloud.get_current()
        total_utilisation = 0.
        utilisations = {}

        for i in server.resource_types:
            used = 0.
            for existing_vm in current.alloc[server]:
                used += existing_vm.res[i]
            # add our own VM's resource demand
            used += vm.res[i]
            utilisations[i] = used/server.cap[i]
            if used > server.cap[i]: # capacity exceeded for this resource
                return -1

        # take custom weights from conf
        weights = conf.custom_weights
        # uniform_weight = 1./len(server.resource_types)  
        # weights = {res : uniform_weight for res in server.resource_types}
        for resource_type, utilisation in utilisations.iteritems():
            total_utilisation += weights[resource_type] * utilisation
        return total_utilisation


    # TODO: maybe split into multiple functions and make this one immutable
    def _remove_vms_from_underutilised_hosts(self):
        """mutable method that finds underutilised hosts, removes VMs from
        them in the current state, updates the _original_vm_hosts dictionary
        and returns all such VMs.

        """
        vms = []
        state = self.cloud.get_current()
        for s in self.cloud.servers:
            if state.underutilised(s):
                vms.extend(state.alloc[s])
                for vm in state.alloc[s]:
                    self._original_vm_hosts[vm] = s
                # remove the VMs from that host for now
                self.cloud.get_current().remove_all(s) # transition?
        return vms


    def get_cheapest_location(self, prices, t):
        locations = prices.axes[1]
        min_price = min([prices[loc][t] for loc in locations])
        location = [loc for loc in locations if prices[loc][t] <= min_price]
        return location[0]


    def _calculate_avg_price(self, curr_price, fc_prices, fc_start, fc_end, weighted=False):
        """ calculate average forecast price """
        forecasts = fc_prices[fc_start:fc_end]
        prices = [curr_price]
        prices.extend(forecasts)
        if weighted:
            # TODO Andreas: consider forecast errors in weights
            return np.average(prices, weights=range(len(prices),0,-1))  # optional weights , e.g. weights=range(10,0,-1)
        else:
            return np.average(prices)


    def get_cheapest_locations(self, t, forecast=False, ideal=False, weighted=False, horizon=8):
        """Get the cheapest locations at the given timestamp
        If forecast is True the average price over the forecast window
            is calculated and returned as list sorted by el price for each location
        Otherwise the prices are sorted by location and returned as sorted array

        Return list of tuples of (location, price) sorted by price (ascending)

        """
        def getPricesKey(item):
            """ Get second entry in item """
            return item[1]
        prices = self.environment.el_prices
        fc_prices = self.environment.forecast_el
        if ideal:
            fc_prices = prices
        period = self.environment.get_period()
        locations = prices.axes[1]
        if forecast:
            fc_list = []
            for loc in locations:
                avg = self._calculate_avg_price(prices[loc][t], fc_prices[loc], t+period, t+period*(horizon), weighted=weighted)
                fc_list.append((loc, avg))
            fc_list.sort(key=getPricesKey)
            return fc_list
        else:
            min_prices = sorted([(loc, prices[loc][t]) for loc in locations], key=getPricesKey)
            return min_prices


    def _find_random_host(self, vm, current_loc=None):
        """Assign the given vm to a server in a FCFS fashion
        If current_loc is given a server from this location is chosen

        """
        if current_loc is None:
            servers = self.cloud.servers
        else:
            servers = [server for s in self.cloud.servers if s.loc == current_loc]
        for server in servers:
            utilisation = self._fits(vm, server)
            #TODO: compare utilisations of different potential hosts
            if utilisation != -1:
                print "Server {} chosen at location {}".format(server, server.loc)
                return server

    def _find_cheapest_host(self, vm, current_loc=None, forecast=False, ideal=False, weighted=False):
        """Find cheapest host, 
        if forecast is True 
        find host at cheapest location based on average of forecasted values

        Otherwise find host at cheapest location based on current energy price
        If current_loc is given, skip host finding for this location

        """
        t = self.environment.get_time()
        cheapest_loc = self.get_cheapest_locations(t,forecast,ideal,weighted)
        # print "cheapest_locations: {}".format(cheapest_loc)

        # iterate over "cheapest locations"
        # if there is not enough space at the cheapest location
        # go to the second cheapest location at this point in time
        for loc_item in cheapest_loc: # loc_item consists of tuples of (location, price)            
            location = loc_item[0]
            # skip the location the vm is currently located
            if location == current_loc and current_loc is not None:
                continue
            # get all servers at that location
            servers = [ server for server in self.cloud.servers if server.loc == location ]
            for server in servers:
                utilisation = self._fits(vm, server)
                #TODO: compare utilisations of different potential hosts
                if utilisation != -1:
                    # print "Server {} chosen at location {}".format(server, server.loc)
                    return server
        return None


    def _get_migration_vms(self, t_next, forecast=False, ideal=False, weighted=False):
        """Get all vms that are currently located at more expensive
        locations with a remaining duration that exceeds the migration
        time. In addition, only migrate vms for which the following 
        equation holds:  mig_cost + price_remote < price_current 

        """
        prices = self.environment.el_prices
        cheapest_loc = self.get_cheapest_locations(t_next, forecast, ideal, weighted)
        current = self.cloud.get_current()
        vms = self.cloud.get_vms()
        if len(vms) == 0:
            return []
        # clear vms of all vms located at the currently cheapest location
        vms = vms.difference(current.unallocated_vms())
        if len(vms) == 0:
            return []        

        # vms_cleared = [vm for vm in vms if current.allocation(vm).loc != cheapest_loc[0][0]]

        # sort vms by duration, with longest ones at the beginning
        def keyDuration(item):
            return item[1]        
        # create sorted list of tupels of (vm, remaining_duration)
        sorted_vms = sorted([(vm, self.environment.get_remaining_duration(vm, t_next)) 
                                        for vm in vms ], key=keyDuration, reverse=True)
        max_duration = sorted_vms[0][1]
        # calculate periods based on max_duration
        fc_range_end = int(max_duration.total_seconds() / 3600) + 1
        # the forecasts for different horizons (job lengths) are precalculated such that 
        # they can be mapped to the current vm's duration
        fc_dict = { i: self.get_cheapest_locations(t_next, forecast, ideal, weighted, horizon=i)
                                        for i in range(1,fc_range_end+1)}
        vms_to_migrate = []
        for vm_item in sorted_vms:
            vm = vm_item[0]
            duration = vm_item[1]
            # Only migrate when duration exceeds migration time
            # break since vms are sorted by duration
            migration_time = ph.calculate_migration_time(vm, conf.fixed_bandwidth)
            if duration.total_seconds() < migration_time:
                break
            # Determine which calculated forecast value
            # is applied based on the current vm's duration
            idx = int(duration.total_seconds() / 3600) + 1
            cheapest_loc = fc_dict[idx]
            loc = current.allocation(vm).loc
            price_current = prices[loc][t_next]
            # get price from cheapest remote location and next timestamp
            price_remote = prices[cheapest_loc[0][0]][t_next]
            mig_cost = ph.calculate_migration_cost(vm, price_current, price_remote)
            # TODO Andreas: calculate migration penalty (cent per minute)
            # Formula to fulfill before migration
            # migrate = Migration Costs + Remote Costs < Current Costs
            if mig_cost + price_remote < price_current:
                vms_to_migrate.append(vm)
        return vms_to_migrate

    def get_migration_vms(self, t_next, scenario, weighted=False):
        if scenario == 1:
            return []
        elif scenario == 2: 
            return []
        elif scenario == 3: 
            return []
        elif scenario == 4: 
            return self._get_migration_vms(t_next)
        elif scenario == 5: 
            return self._get_migration_vms(t_next, forecast=True, weighted=weighted)
        elif scenario == 6: 
            return self._get_migration_vms(t_next, forecast=True, ideal=True, weighted=weighted)

    def find_host(self, vm, scenario, current_loc=None, weighted=False):
        if scenario == 1:
            return self._find_random_host(vm, current_loc)
        elif scenario == 2: 
            return self._find_cheapest_host(vm, current_loc)
        elif scenario == 3: 
            return self._find_cheapest_host(vm, current_loc, forecast=True, weighted=weighted)
        elif scenario == 4: 
            return self._find_cheapest_host(vm, current_loc)
        elif scenario == 5: 
            return self._find_cheapest_host(vm, current_loc, forecast=True, weighted=weighted)
        elif scenario == 6: 
            return self._find_cheapest_host(vm, current_loc, forecast=True, ideal=True, weighted=weighted)

    def reevaluate(self):
        self.schedule = Schedule()
        # get new requests
        requests = self.environment.get_requests()
        current = self.cloud.get_current()
        prices = self.environment.el_prices
        t_next = self.environment.get_time() + self.environment.get_period()
        weighted = False
        # params 
        # vms_to_exclude = []
        # t_mig = self.environment.get_time() + self.environment.get_period() - pd.offsets.Minute(5)
        for t_req, request in requests.iteritems():
            # for each boot request:
            # find the best server
            #  - find server that can host this VM
            #  - make sure the server's resources are now reserved
            # add new migration to the schedule
            if request.what == 'boot':
                server = self.find_host(request.vm, self.scenario, weighted=weighted)
                if server is None:
                    error('not enough free resources for VM {}'.format(request.vm))
                    # self.cloud.get_current().vms.remove(request.vm)
                    # action = VMRequest(request.vm, 'boot')
                    # vms_to_exclude.append(request.vm)
                    # self.cloud.apply(action)
                    # # delete at the end of this simulation timeframe
                    # self.schedule.add(action, t_next)
                else:
                    # "Migrate" the vm from none to a new server (=boot)
                    action = Migration(request.vm, server)
                    self.cloud.apply(action)
                    # important! take time of request (t_req) 
                    # instead of t to add to actions (not rounded to hours)
                    self.schedule.add(action, t_req)


        #############################################
        # Another possibility: 
        # just take current time stamp to place requests
        # Taken from bfd scheduler
        #############################################
    
        # VMs = []
        # # get VMs that need to be placed
        # #  - VMs from boot requests
        # requests = self.environment.get_requests()
        # for request in requests:
        #     #import ipdb; ipdb.set_trace()
        #     if request.what == 'boot':
        #         VMs.append(request.vm)
        # #  - select VMs on underutilised PMs
        # VMs.extend(self._remove_vms_from_underutilised_hosts())

        # VMs = sort_vms_decreasing(VMs)


        
        if t_next < self.environment.end:

            # already chosen vms that should be migrated
            vms_to_migrate = self.get_migration_vms(t_next, self.scenario, weighted=weighted)
            # vms_to_migrate = list(set(vms_to_migrate) - set(vms_to_exclude))
            for vm in vms_to_migrate:
                current_server = self.cloud.get_current().allocation(vm)
                current_loc = current_server.loc
                server = self.find_host(vm, self.scenario, current_loc=current_loc, weighted=weighted)
                if server is None:
                    error('No space to migrate VM {}'.format(vm))
                else:
                    # Actually migrate the vm to a server at a cheaper location
                    action = Migration(vm, server)
                    self.cloud.apply(action)
                    # migrate at the end of this simulation timeframe
                    self.schedule.add(action, t_next)
        self.cloud.reset_to_real()
        return self.schedule

    def finalize(self):
        pass
