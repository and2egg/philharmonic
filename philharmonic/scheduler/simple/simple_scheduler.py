from philharmonic.scheduler.ischeduler import IScheduler
from philharmonic import Schedule, Migration
from philharmonic.logger import info, debug, error

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
        uniform_weight = 1./len(server.resource_types)  
        # TODO_Andreas: take weight from class instead of the standard weight
        weights = server._weights
        # weights = {res : uniform_weight for res in server.resource_types}
        for resource_type, utilisation in utilisations.iteritems():
            total_utilisation += weights[resource_type] * utilisation
        return total_utilisation

    def find_host(self, vm):
        for server in self.cloud.servers:
            utilisation = self._fits(vm, server)
            #TODO: compare utilisations of different potential hosts
            if utilisation != -1:
                return server
        return None

    def reevaluate(self):
        self.schedule = Schedule()
        t = self.environment.get_time()
        # get new requests
        requests = self.environment.get_requests()
        
        for t_req, request in requests.iteritems():
        # for request in requests:
            import ipdb; ipdb.set_trace()
            if request.what == 'boot':

                server = self.find_host(request.vm)
                if server is None:
                    error('not enough free resources for VM {}'.format(request.vm))
                else:
                    # import ipdb; ipdb.set_trace()
                    action = Migration(request.vm, server)
                    self.cloud.apply(action)
                    # important! take time of request (t_req) 
                    # instead of t to add to actions (not rounded to hours)
                    self.schedule.add(action, t_req)
        # for each boot request:
        # find the best server
        #  - find server that can host this VM
        #  - make sure the server's resources are now reserved
        # add new migration to the schedule
        self.cloud.reset_to_real()
        return self.schedule

    def finalize(self):
        pass
