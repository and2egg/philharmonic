from philharmonic.scheduler.ischeduler import IScheduler
from philharmonic import Schedule, Migration, VMRequest
from philharmonic.logger import info, debug, error
import philharmonic as ph
import pandas as pd

class SimpleScheduler(IScheduler):
    """Simple scheduler. Should find host based on capacities
        and energy prices applying a greedy algorithm. """

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
        # take weight from class instead of the standard weight
        weights = server._weights
        # uniform_weight = 1./len(server.resource_types)  
        # weights = {res : uniform_weight for res in server.resource_types}
        for resource_type, utilisation in utilisations.iteritems():
            total_utilisation += weights[resource_type] * utilisation
        return total_utilisation


    def get_cheapest_location(self, prices, t):
        locations = prices.axes[1]
        min_price = min([prices[loc][t] for loc in locations])
        location = [loc for loc in locations if prices[loc][t] <= min_price]
        return location[0]


    def get_cheapest_locations(self, prices, t):
        def getKeyLocation(item):
            return item[1]
        locations = prices.axes[1]
        min_prices = sorted([(loc, prices[loc][t]) for loc in locations], key=getKeyLocation)
        return min_prices


    def _find_random_host(self, vm):
        for server in self.cloud.servers:
            utilisation = self._fits(vm, server)
            #TODO: compare utilisations of different potential hosts
            if utilisation != -1:
                print "Server {} chosen at location {}".format(server, server.loc)
                return server

    def _find_cheapest_host(self, vm, current_loc=None):
        prices = self.environment.el_prices
        t = self.environment.get_time()
        cheapest_loc = self.get_cheapest_locations(prices,t)
        # print "cheapest_locations: {}".format(cheapest_loc)

        for loc_item in cheapest_loc:
            location = loc_item[0]
            if location == current_loc and current_loc is not None:
                continue
            servers = [ server for server in self.cloud.servers if server.loc == location ]
            for server in servers:
                utilisation = self._fits(vm, server)
                #TODO: compare utilisations of different potential hosts
                if utilisation != -1:
                    # print "Server {} chosen at location {}".format(server, server.loc)
                    return server
        return None


    def find_host(self, vm, cost_aware=True):
        if cost_aware:
            return self._find_cheapest_host(vm)
        else:
            return self._find_random_host(vm)


    def get_migration_vms(self, t, forecast=False):
        prices = self.environment.el_prices
        cheapest_loc = self.get_cheapest_locations(prices,t)
        current = self.cloud.get_current()
        # iter_loc = [loc for loc in cheapest_loc if loc[0] != cheapest_loc[0][0]]  # exclude item from zero list element

        vms = self.cloud.get_vms()
        if len(vms) == 0:
            return []

        vms = vms.difference(current.unallocated_vms())
        vms_cleared = [vm for vm in vms if current.allocation(vm).loc != cheapest_loc[0][0]]

        # sort vms by duration, reversed
        def getKeyDuration(item):
            return item[1]        
        sorted_vms = sorted([(vm, self.environment.get_remaining_duration(vm)) 
                                        for vm in vms_cleared ], key=getKeyDuration, reverse=True)

        vms_to_migrate = []

        price_remote = cheapest_loc[0][1]
        
        for vm_item in sorted_vms:
            vm = vm_item[0]
            loc = current.allocation(vm).loc
            price_current = prices[loc][t]
            mig_cost = ph.calculate_migration_cost(vm, price_current, price_remote)
        
            # Formula to fulfill before migration
            # migrate = Migration Costs + (Expected) Remote Costs < (Expected) Current Costs
            if mig_cost + price_remote < price_current:
                vms_to_migrate.append(vm)

        return vms_to_migrate     



    def reevaluate(self):
        self.schedule = Schedule()
        # get new requests
        requests = self.environment.get_requests()
        current = self.cloud.get_current()
        prices = self.environment.el_prices
        t_next = self.environment.get_time() + self.environment.get_period()
        t_mig = self.environment.get_time() + self.environment.get_period() - pd.offsets.Minute(5)



        for t_req, request in requests.iteritems():

            # for each boot request:
            # find the best server
            #  - find server that can host this VM
            #  - make sure the server's resources are now reserved
            # add new migration to the schedule
            if request.what == 'boot':
                server = self.find_host(request.vm)
                if server is None:
                    error('not enough free resources for VM {}'.format(request.vm))
                    action = VMRequest(request.vm, 'delete')
                    self.cloud.apply(action)
                    # delete at the end of this simulation timeframe
                    self.schedule.add(action, t_next)
                else:
                    # "Migrate" the vm from none to a new server (=boot)
                    action = Migration(request.vm, server)
                    self.cloud.apply(action)
                    
                    # important! take time of request (t_req) 
                    # instead of t to add to actions (not rounded to hours)
                    self.schedule.add(action, t_req)
        
        if t_next < self.environment.end:
            # import ipdb;ipdb.set_trace()
            # already chosen vms that should be migrated
            vms_to_migrate = self.get_migration_vms(t_next)
            
            for vm in vms_to_migrate:
                current_loc = self.cloud.get_current().allocation(vm).loc
                server = self._find_cheapest_host(vm, current_loc)
                if server is None:
                    error('No space to migrate VM {}'.format(vm))
                else:
                    # Actually migrate the vm to a server at a cheaper location
                    action = Migration(vm, server)
                    self.cloud.apply(action)
                    # migrate at the end of this simulation timeframe
                    self.schedule.add(action, t_mig)

        self.cloud.reset_to_real()
        return self.schedule

    def finalize(self):
        pass
