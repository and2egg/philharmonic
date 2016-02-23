from philharmonic.scheduler.ischeduler import IScheduler
from philharmonic import Schedule, Migration, VMRequest
from philharmonic.logger import info, debug, error
from philharmonic import conf

import philharmonic as ph
import pandas as pd
import numpy as np
import math

class BCUScheduler(IScheduler):
    """Best cost decreasing scheduler. It determines the cheapest location based on 
    current and future energy prices and decides whether a job should be migrated 
    based on remaining execution length while taking into account possible SLA penalties.  

    """


    # map number of simulation time stamps for which each vm is blocked for migration
    blocked_vms = {}
    # store forecast horizons for maximum cost benefit for each vm and location
    blocked_vms_horizons = {}
    # map of price differences for current time stamp
    price_difference_map = {}



    def __init__(self, cloud=None, driver=None):
        IScheduler.__init__(self, cloud, driver)


    def sort_vms_decreasing(self, VMs):
        return sorted(VMs, key=lambda x : (x.res['#CPUs'], x.res['RAM']),
                      reverse=True)

    def sort_pms_increasing(self, PMs, state):
        return sorted(PMs,
                  key=lambda x : (state.free_cap[x]['#CPUs'],
                                  state.free_cap[x]['RAM']))

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
        if conf.custom_weights is not None:
            weights = conf.custom_weights
        else:
            uniform_weight = 1./len(server.resource_types)  
            weights = {res : uniform_weight for res in server.resource_types}

        for resource_type, utilisation in utilisations.iteritems():
            total_utilisation += weights[resource_type] * utilisation
        return total_utilisation


    def _init_blocked_vms(self):
        # add all vms to blocked_vms array the first time
        vms_all = self.environment.get_vms_from_requests()
        locations = self.environment.locations
        for vm in vms_all: 
            self.blocked_vms[vm] = 0
            self.blocked_vms_horizons[vm] = {}
            for loc in locations:
                self.blocked_vms_horizons[vm][loc] = 0


    def block_vm_for_migrations(self, vm, loc):
        """put a number to blocked_vms that states how many iterations
        this vm should not be migrated"""
        self.blocked_vms[vm] = self.blocked_vms_horizons[vm][loc]


    def add_downtime(self, vm, loc):
        """Add predicted downtime for migration of vm to location loc."""
        d_pred = ph.calculate_predicted_downtime(vm, vm.server.loc, loc, conf.bandwidth_map)
        vm.downtime = vm.downtime + d_pred


    def _get_list_difference(self, l1, l2):
        """get the difference from two lists
        e.g. items that are in l1 but not in l2"""
        s = set(l2)
        return [x for x in l1 if x not in s]


    def _place_vm(self, vm, host, t):
        """Place vm on host. """
        action = Migration(vm, host)
        self.cloud.apply(action)
        self.schedule.add(action, t)


    def assign_hosts_by_size(self, vms, loc, t):
        current = self.cloud.get_current()
        servers = self.environment.servers_per_loc[loc]
        hosts = filter(lambda s : not current.server_free(s), servers)
        hosts = self.sort_pms_increasing(hosts, current)
        inactive_hosts = filter(lambda s : current.server_free(s), servers)
        inactive_hosts = self.sort_pms_increasing(inactive_hosts, current)

        assigned_vms = []
        for vm in vms:
            mapped = False
            while not mapped:
                for host in hosts:
                    if self._fits(vm, host) != -1:
                        self._place_vm(vm, host, t)
                        assigned_vms.append(vm)
                        mapped = True
                        break
                if not mapped:
                    if len(inactive_hosts) > 0:
                        host = inactive_hosts.pop(0)
                        hosts.append(host)
                        hosts = self.sort_pms_increasing(hosts, current)
                    else:
                        break
        return assigned_vms


    def assign_host_by_size(self, vm, loc, t):
        current = self.cloud.get_current()
        servers = self.environment.servers_per_loc[loc]
        hosts = filter(lambda s : not current.server_free(s), servers)
        hosts = self.sort_pms_increasing(hosts, current)
        inactive_hosts = filter(lambda s : current.server_free(s), servers)
        inactive_hosts = self.sort_pms_increasing(inactive_hosts, current)

        mapped = False
        while not mapped:
            for host in hosts:
                if self._fits(vm, host) != -1:
                    self._place_vm(vm, host, t)
                    mapped = True
                    break
            if not mapped:
                if len(inactive_hosts) > 0:
                    host = inactive_hosts.pop(0)
                    hosts.append(host)
                    hosts = self.sort_pms_increasing(hosts, current)
                else:
                    break
        return mapped



    def evaluate_BFD(self, t, requests):
        """evalute requests in a best fit decreasing fashion.
        sort pms by size increasing, vms by size decreasing
        and find host starting from least utilised data center
        """
        VMs = []
        # get VMs that need to be placed
        #  - VMs from boot requests
        for request in requests:
            if request.what == 'boot':
                VMs.append(request.vm)
        #  - select VMs on underutilised PMs
        # VMs.extend(self._remove_vms_from_underutilised_hosts())

        current = self.cloud.get_current()

        VMs = self.sort_vms_decreasing(VMs)

        if len(VMs) == 0:
            return []

        util = current.calculate_utilisations_per_location()
        sorted_util = sorted(util.items(), key=lambda x: x[1])

        assigned_vms = []

        for util_item in sorted_util:
            loc = util_item[0]
            VMs = self._get_list_difference(VMs, assigned_vms)
            assigned_vms = self.assign_hosts_by_size(VMs, loc, t)
            

    def evaluate_BCF(self, t, requests, forecast=False, ideal=False, optimized=False):
        """evalute requests in a best cost fit fashion.
        Get location based on current or future energy prices,
        sort pms by size increasing, vms by size decreasing and 
        find host within the assigned location
        """
        VMs = []
        # get VMs that need to be placed
        #  - VMs from boot requests
        for request in requests:
            if request.what == 'boot':
                VMs.append(request.vm)
        #  - select VMs on underutilised PMs
        # VMs.extend(self._remove_vms_from_underutilised_hosts())

        VMs = self.sort_vms_decreasing(VMs)

        if len(VMs) == 0:
            return []

        self.assign_to_cheapest_hosts(t, VMs, forecast, ideal, optimized)


    def assign_to_cheapest_hosts(self, t, vms, forecast=False, ideal=False, weighted=False, optimized=False):
        """Find cheapest host
        if forecast is True assign to hosts at cheapest location based on average of forecasted values
        Otherwise assign to hosts at cheapest location based on current energy price

        """
        prices = self.environment.el_prices
        fc_prices = self.environment.forecast_el
        if ideal or forecast == False:
            fc_prices = prices

        max_h = 1
        if forecast:
            max_h = conf.max_fc_horizon

        if optimized:
            total_remaining_dur = self.environment.end - t_next
            total_remaining_dur = total_remaining_dur.total_seconds() / 3600
            max_h = int(total_remaining_dur)

        for vm in vms:
            vm_remaining = self.environment.get_remaining_duration(vm, t)
            vm_remaining = int(vm_remaining.total_seconds() / 3600)

            if vm_remaining <= 1:
                cheapest_loc = self.get_cheapest_locations(t, False, False, weighted, 1)
            else:
                max_fc = min(vm_remaining-1, max_h)
                cheapest_loc = self.get_cheapest_locations(t, forecast, ideal, weighted, max_fc)

            mapped = False
            # iterate over "cheapest locations"
            # if there is not enough space at the cheapest location
            # go to the second cheapest location at this point in time
            for loc_item in cheapest_loc:
                # loc_item consists of tuples of (fc horizon, location, avg price)
                h = loc_item[0] # optimal fc horizon with minimum avg cost
                location = loc_item[1]
                mapped = self.assign_host_by_size(vm, location, t)
                if mapped:
                    self.blocked_vms_horizons[vm][location] = h
                    self.block_vm_for_migrations(vm, location)
                    break


    def get_cheapest_locations(self, t, forecast=False, ideal=False, weighted=False, max_fc=2):
        """Get the cheapest locations at the given timestamp
        If forecast is True the average price over the forecast window
            is calculated and returned as list sorted by el price for each location
        Otherwise the prices are sorted by location and returned as sorted array

        Return list of tuples of (location, price) sorted by price (ascending)

        """
        prices = self.environment.el_prices
        fc_prices = self.environment.forecast_el
        if ideal or forecast == False:
            fc_prices = prices
        period = self.environment.get_period()
        locations = prices.axes[1]
        if forecast:
            cheapest_loc = []
            for loc in locations:
                fc_list = []
                for h in range(1, max_fc+1):
                    avg = self._calculate_avg_price(prices[loc][t], fc_prices[loc], t+period, t+period*(h), weighted=weighted)
                    fc_list.append((h, loc, avg))
                min_price = min(fc_list, key=lambda x: x[2])
                cheapest_loc.append(min_price)
            return sorted(cheapest_loc, key=lambda x: x[2])
        else:
            # return list of tuples of fc horizon(=0), location and energy price, sorted by energy prices
            return sorted([(0, loc, prices[loc][t]) for loc in locations], key=lambda x: x[2])


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


    def find_host_for_vm(self, vm, loc):
        """find a host for the given location"""
        current = self.cloud.get_current()
        servers = self.environment.servers_per_loc[loc]
        hosts = filter(lambda s : not current.server_free(s), servers)
        hosts = self.sort_pms_increasing(hosts, current)
        inactive_hosts = filter(lambda s : current.server_free(s), servers)
        inactive_hosts = self.sort_pms_increasing(inactive_hosts, current)
        while True:
            for host in hosts:
                if self._fits(vm, host) != -1:
                    return host
            if len(inactive_hosts) > 0:
                host = inactive_hosts.pop(0)
                hosts.append(host)
                hosts = self.sort_pms_increasing(hosts, current)
            else:
                break
        return None


    def find_host_for_vm_no_sorting(self, vm, loc):
        servers = self.environment.servers_per_loc[loc]
        for server in servers:
            utilisation = self._fits(vm, server)
            if utilisation != -1:
                return server




    #############################################
    ######  Utility preparation functions  ######
    #############################################

    def _get_probability_of_sla_penalty(self, vm, loc):
        """utility criteria
        retrieves the probability of an sla penalty
        given this vm's current memory, dirty page rate
        and the bandwidth connection from the location
        it should be migrated to
        """
        # get accumulated downtime of vm
        down_acc = vm.downtime
        # get predicted downtime when migrated to location loc with dpr and bandwidth values
        down_pred = ph.calculate_predicted_downtime(vm, vm.server.loc, loc, conf.bandwidth_map)
        
        if vm.penalties < 3:
            sla_th = self.environment.vm_sla_ths[vm][vm.penalties]
            if (down_acc + down_pred) < sla_th:
                prob_pen = (down_acc + down_pred) / float(sla_th)
            else:
                prob_pen = 1
        else:
            prob_pen = 1
        return prob_pen

    def _calculate_migration_energy(self, vm, loc):
        return ph.calculate_migration_energy(vm, loc, conf.bandwidth_map)

    def _get_dc_load(self):
        """utility criteria
        retrieves the current loads (utilisation) in all locations
        and returns a dict of loads and the location with currently
        maximum utilisation as tuple, (loc, util)
        """
        state = self.cloud.get_current()
        util = state.calculate_utilisations_per_location()
        max_util = max(util.items(), key=lambda x: x[1])
        return [util, max_util]

    def _get_relative_dc_load(self, dc_loads, loc):
        """get the relative load (utilisation) for location loc
        to the maximum utilisation of all locations
        """
        return dc_loads[0][loc] / float(dc_loads[1][1])

    def _get_max_benefit(self, fc_dict, max_fc, loc1, loc2):
        """get maximum cost benefit given a dict with current price differences, 
        a maximum forecast horizon up to which mean values should be compared, 
        and two locations loc1 and loc2 where migration is planned from 
        location loc1 to location loc2
        @return a tuple containing the forecast horizon and the maximum mean distance
                to location 2
        """
        start_value = fc_dict[0][loc1][loc2]
        if start_value <= 0:
            return (0, start_value)

        mean_values = []
        for h in range(max_fc+1):
            mean_dist = fc_dict[h][loc1][loc2]
            mean_values.append((h, mean_dist))
        max_value = max(mean_values, key=lambda x: x[1])

        # check if the calculated horizon is below the maximum forecast threshold
        if max_value[0] < conf.max_fc_horizon:
            return max_value
        else:
            return (0, -1) # do not migrate at all


    def _setup_price_map(self, t_next, forecast=False, ideal=False):
        if len(self.price_difference_map) == 0:
            prices = self.environment.el_prices
            fc_prices = self.environment.forecast_el
            if ideal or forecast == False:
                fc_prices = prices
            self.price_difference_map = self._calculate_price_differences(t_next, fc_prices, 0, conf.max_fc_horizon-1)


    def _calculate_price_differences(self, t_next, fc_prices, min_h=0, max_h=0, normalise=True):
        """calculate the price differences between each two different
        locations given a minimum and maximum forecast horizon.
        the maximum cost benefit for a vm is defined as migrating to 
        a location compared to which the (positive) price differences are highest

        """
        def _create_dict(locations, min_h, max_h):
            d = {}
            for h in range(min_h,max_h+1): # range is exclusive last value, therefore +1
                d[h] = {}
                for l in locations: 
                    d[h][l] = {}
            return d

        def _normaliseMeanError(me, min_me, max_me):
            """normalise the mean error to the range [-1,1]"""
            # min and max will be always exactly the same
            # -> if one location is max, the other location is min
            return me / float(max_me)

        period = self.environment.get_period()
        locations = fc_prices.axes[1]
        
        min_ME = None
        max_ME = None
        d = _create_dict(locations, min_h, max_h)
        eval_start = t_next # next time stamp

        # calculate mean errors for each pair of locations and for each forecast horizon
        for h in range(min_h,max_h+1): # range is exclusive last value, therefore +1
            curr_h = eval_start+period*h
            visited = {}
            for l in locations:
                visited[l] = False

            for l in d[h]:
                for other in d[h]:
                    if other != l and not visited[other]:
                        vec1 = fc_prices[l][eval_start : curr_h]
                        vec2 = fc_prices[other][eval_start : curr_h]

                        mean_error = sum([v[0] - v[1] for v in zip(vec1,vec2)]) # mean error

                        d[h][l][other] = mean_error         # set mean error for this combination of fc horizon, 
                        d[h][other][l] = -mean_error        # location l (from) and location other (to) migration action
                                                            # ... the (-) means negation, not setting the number as negative
                visited[l] = True

        # for each fc horizon h and location l save a tuple of (location, maximum mean error)
        for h in range(min_h,max_h+1): # range is exclusive last value, therefore +1
            for l in locations: 
                min_T = min(d[h][l].items(), key=lambda x: x[1])  # tuple of location and minimum ME value
                max_T = max(d[h][l].items(), key=lambda x: x[1])  # tuple of location and maximum ME value
                if min_ME is None or min_T[1] < min_ME:
                    min_ME = min_T[1]
                if max_ME is None or max_T[1] > max_ME:
                    max_ME = max_T[1]

        # if a minimum price difference is not reached, do not migrate
        if max_ME < conf.min_price_threshold:
            return {}

        if normalise:
            # normalise values to min and max mean errors
            for h in range(min_h,max_h+1): # range is exclusive last value, therefore +1
                for l in locations: 
                    mean_errors = d[h][l].items()
                    for me in mean_errors:
                        loc = me[0]
                        err = me[1]
                        d[h][l][loc] = _normaliseMeanError(err, min_ME, max_ME)

        return d


    def _prepare_utility_function(self, t_next, forecast=False, ideal=False, optimized=False):
        """prepare all criterias to be evaluated in a 
        utility function. Do this in a common method
        to save computation time (iterate over the set
        of vms just once)
        """
        prices = self.environment.el_prices
        current = self.cloud.get_current()
        locations = prices.axes[1]

        vms = self.cloud.get_vms()
        if len(vms) == 0:
            return {}
        # vms = vms.difference(current.unallocated_vms())
        # if len(vms) == 0:
        #     return {}

        prices = self.environment.el_prices
        fc_prices = self.environment.forecast_el
        if ideal or forecast == False:
            fc_prices = prices

        min_h = 0
        max_h = 0 # inclusive
        if forecast:
            max_h = conf.max_fc_horizon-1

        if optimized:
            total_remaining_dur = self.environment.end - t_next
            total_remaining_dur = total_remaining_dur.total_seconds() / 3600
            max_h = int(total_remaining_dur)

        fc_dict = self._calculate_price_differences(t_next, fc_prices, min_h, max_h)

        if len(fc_dict) == 0:
            return []
        
        sla_penalty = {}
        mig_energy = {}
        remaining_dur = {}
        cloud_util = self._get_dc_load()
        cost_benefit = {}

        max_mig_energy = 0

        migration_vms = []
        not_migrated_vms = []


        for vm in vms:

            # check if current vm is currently blocked for migration. if yes, do not migrate!
            if self.blocked_vms[vm] > 0:
                self.blocked_vms[vm] -= 1
                not_migrated_vms.append(vm)
                continue

            vm_remaining = self.environment.get_remaining_duration(vm, t_next)
            vm_remaining = int(vm_remaining.total_seconds() / 3600)
            if vm_remaining < conf.min_vm_remaining:
                not_migrated_vms.append(vm)
                continue

            # preparing remaining duration criteria
            remaining_dur[vm] = vm_remaining
            max_fc = min(vm_remaining-1, max_h)
            current_loc = current.allocation(vm).loc

            sla_penalty[vm]     = {}
            mig_energy[vm]      = {}
            cost_benefit[vm]    = {}

            for loc in locations:
                if loc != current_loc: 
                   # preparing sla penalty criteria
                    sla_penalty[vm][loc] = self._get_probability_of_sla_penalty(vm, loc)

                    # preparing migration energy criteria
                    mig_energy[vm][loc] = self._calculate_migration_energy(vm, loc) # in Joules

                    # preparing cost benefit criteria
                    [h, max_diff] = self._get_max_benefit(fc_dict, max_fc, current_loc, loc) # get maximum cost benefit for vm when migrating to another location
                    self.blocked_vms_horizons[vm][loc] = h
                    cost_benefit[vm][loc] = max_diff

                    if mig_energy[vm][loc] > max_mig_energy:
                        max_mig_energy = mig_energy[vm][loc]

                else:
                    sla_penalty[vm][loc] = None
                    mig_energy[vm][loc] = None
                    cost_benefit[vm][loc] = None

        migration_vms = vms.difference(not_migrated_vms)

        return [ migration_vms, sla_penalty, mig_energy, max_mig_energy, remaining_dur, cloud_util, cost_benefit ]


    def calculate_utility_function(self, t_next, forecast=False, ideal=False, optimized=False):

        current = self.cloud.get_current()
        result = self._prepare_utility_function(t_next, forecast, ideal, optimized)

        if  len(result) == 0:
            return []

        for i in range(len(result)):
            if type(result[i]) is list or type(result[i]) is set or type(result[i]) is dict:
                if len(result[i]) == 0:
                    return []

        [ migration_vms, sla_pen,mig_energy,max_mig_energy,remaining_dur,cloud_util,estimated_savings ] = result

        fc_prices = self.environment.forecast_el
        locations = fc_prices.axes[1]

        max_rem = max(remaining_dur.items(), key=lambda x:x[1])[1]

        max_estimated_savings = {}
        vms = migration_vms
        u_result = []

        prices = self.environment.el_prices
        period = self.environment.get_period()

        for vm in vms:
            current_loc = vm.server.loc
            u_value = {}

            # if t_next == pd.Timestamp('2013-07-03 06:00') and vm.id == 343:
            #     import ipdb;ipdb.set_trace()
            
            for loc in locations:
                if loc != current_loc:

                    sla_penalty = 1 - sla_pen[vm][loc]
                    migration_energy = 1 - (mig_energy[vm][loc] / max_mig_energy)
                    remaining_vm = remaining_dur[vm] / float(max_rem)
                    dcload = self._get_relative_dc_load(cloud_util, current_loc)
                    savings = estimated_savings[vm][loc]

                    result   =  conf.w_sla       * sla_penalty       +  \
                                conf.w_energy    * migration_energy  +  \
                                conf.w_vm_rem    * remaining_vm      +  \
                                conf.w_dcload    * dcload            +  \
                                conf.w_cost      * savings

                    # make sure to only migrate when there are at least some energy cost savings expected
                    if savings <= 0:
                        result = 0

                    u_value[loc] = result

            # taking only maximum utility value calculated over all fc horizons for all locations applicable to the vm
            max_u_value = max(u_value.items(), key=lambda x: x[1])
            u_result.append((vm, max_u_value[0], max_u_value[1])) # add result tuple of (vm, location, max utility value)

        u_result = sorted(u_result, key=lambda x: x[2], reverse=True)

        return u_result


    def evaluate_utility_function(self, t_next, forecast=False, ideal=False, optimized=False):
        """Evaluate the utility function results for each vm 
        and return vms with a utility value higher than a 
        specified threshold

        """

        u_result = self.calculate_utility_function(t_next, forecast, ideal, optimized)
        if len(u_result) == 0:
            return []

        # print ', '.join(map(str, u_result))

        result = [ u_item for u_item in u_result if u_item[2] > conf.utility_threshold ]

        if len(result) > 0:
            print ', '.join(map(str, result))

        return result


    def assign_request(self, t, requests, scenario):
        """Assign requests to servers depending on the selected scenario 
        (see scheduler conf for scenario description)"""
        if scenario == 1:
            self.evaluate_BFD(t, requests)
        elif scenario == 2: 
            self.evaluate_BCF(t, requests)
        elif scenario == 3: 
            self.evaluate_BCF(t, requests, forecast=True)
        elif scenario == 4: 
            self.evaluate_BCF(t, requests, forecast=True, ideal=True)
        elif scenario == 5: 
            self.evaluate_BCF(t, requests)
        elif scenario == 6: 
            self.evaluate_BCF(t, requests, forecast=True)
        elif scenario == 7: 
            self.evaluate_BCF(t, requests, forecast=True, ideal=True)
        elif scenario == 8: 
            self.evaluate_BCF(t, requests, forecast=True, ideal=True, optimized=True)



    def vms_to_migrate(self, t_next, scenario):
        """call method depending on selected scenario to get the vms 
        that should be migrated (see scheduler conf for scenario description)"""
        if scenario == 1:
            return []
        elif scenario == 2: 
            return []
        elif scenario == 3: 
            return []
        elif scenario == 4: 
            return []
        elif scenario == 5: 
            return self.evaluate_utility_function(t_next)
        elif scenario == 6: 
            return self.evaluate_utility_function(t_next, forecast=True)
        elif scenario == 7: 
            return self.evaluate_utility_function(t_next, forecast=True, ideal=True)
        elif scenario == 8:
            return self.evaluate_utility_function(t_next, forecast=True, ideal=True, optimized=True)


    def reevaluate(self):
        self.schedule = Schedule()
        # get new requests
        requests = self.environment.get_requests()
        t = self.environment.get_time()
        t_next = self.environment.get_time() + self.environment.get_period()
        self.price_difference_map = {}
        if len(self.blocked_vms) == 0:
            self._init_blocked_vms()

        print "Timestamp "+str(t_next)

        self.assign_request(t, requests, self.scenario)
        
        if t_next < self.environment.end:

            # vm items with maximum utility values that should be migrated
            migration_vm_items = self.vms_to_migrate(t_next, self.scenario)
            for vm_item in migration_vm_items:
                vm = vm_item[0]
                loc = vm_item[1]
                server = self.find_host_for_vm(vm, loc)
                if server is not None:
                    # Actually migrate the vm to a server at a cheaper location
                    self.add_downtime(vm, server.loc)
                    self.block_vm_for_migrations(vm, loc)
                    action = Migration(vm, server)
                    self.cloud.apply(action)
                    # migrate at the end of this simulation timeframe
                    self.schedule.add(action, t_next)
        self.cloud.reset_to_real()


        return self.schedule

    def finalize(self):
        pass

