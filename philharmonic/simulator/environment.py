class Environment(object):
    """provides data about all the data centers
    - e.g. the temperature and prices at different location

    """
    def __init__(self):
        pass

    def current_data(self):
        """return all the current data for all the locations"""
        raise NotImplemented

class SimulatedEnvironment(Environment):
    """stores and provides simulated data about the environment
    - e.g. the temperature and prices at different location

    """
    def __init__(self):
        super(SimulatedEnvironment, self).__init__()
        self._t = None

    def set_time(self, t):
        self._t = t

    def get_time(self, t):
        return t

    t = property(get_time, set_time, doc="current time")