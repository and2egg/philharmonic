'''
Created on Oct 9, 2012

@author: kermit
'''

from philharmonic import conf
from energy_predictor import EnergyPredictor
import philharmonic.openstack.console_api as openstack
from philharmonic.scheduler.ischeduler import IScheduler
from philharmonic.logger import log



class PeakPauser(IScheduler):

    def __init__(self, cloud=None):
        IScheduler.__init__(self, cloud)
        self.paused=False
        openstack.dummy = conf.dummy
        openstack.authenticate()


    def parse_prices(self, location, percentage_to_pause):
        self.energy_price = EnergyPredictor(location, percentage_to_pause)

    def price_is_expensive(self):
        return self.energy_price.is_expensive()

    def pause(self, vm):
        if not self.paused:
            vm.pause()
            if not conf.dummy: # TODO: this should go inside vm.pause()
                openstack.pause(conf.instance)
            self.paused = True
            log("paused")

    def unpause(self, vm):
        if self.paused:
            vm.unpause()
            if not conf.dummy: # TODO: this should go inside vm.unpause()
                openstack.unpause(conf.instance)
            log("unpaused")
            self.paused = False

    def initialize(self):
        #TODO: act on a set of VMs, not only one
        self.unpause()  # in case the VM was paused before we started
        self.parse_prices(conf.historical_en_prices, conf.percentage_to_pause)

    def finalize(self):
        self.unpause()  # don't leave a VM hanging after the experiment's done

    def reevaluate(self):
        if self.price_is_expensive():
            for vm in self.cloud.vms: #TODO: green instances only
                self.pause(vm)
        else:
            for vm in self.cloud.vms: #TODO: green instances only
                self.unpause(vm)


class NoScheduler(IScheduler):

    def __init__(self):
        # call IScheduler's constructor
        #super(PeakPauser, self).__init__()
        IScheduler.__init__(self)

    def initialize(self):
        pass

    def finalize(self):
        pass

    def reevaluate(self):
        pass
