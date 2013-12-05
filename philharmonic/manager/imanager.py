from philharmonic import conf

class IManager(object):
    """abstract cloud manager. Asks the scheduler what to do, given the current
    state of the environment and arbitrates the actions to the cloud.

    """

    def __init__(self, scheduler):
        """Create manager's assets."""
        self.scheduler = scheduler

    def run(self):
        raise NotImplemented


class ManagerFactory():
    """Easier manager creation"""

    @staticmethod
    def create_from_conf(conf):
        """pass a conf module to read paramenters from"""

        # schedulers to choose from
        from philharmonic.scheduler.peak_pauser import PeakPauser, NoScheduler
        # managers to choose from
        from philharmonic.manager.manager import Manager
        from philharmonic.simulator.simulator import Simulator

        # create the scheduler
        ChosenScheduler = globals()[conf.scheduler]
        scheduler = ChosenScheduler()
        if not scheduler:
            scheduler = NoScheduler
        # connect everything in a manager
        ChosenManager = globals()[conf.manager]
        manager = ChosenManager(scheduler)
        return manager
