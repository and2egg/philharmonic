import pandas as pd
import numpy as np

import inputgen

def cleaned_requests(requests):
    """return requests with simultaneous boot & delete actions removed"""
    times = []
    values = []
    del_vms = set([req.vm for req in requests.values if req.what == 'delete'])
    boot_vms = set([req.vm for req in requests.values if req.what == 'boot'])
    useless_vms = boot_vms.intersection(del_vms)
    for t, req in requests.iteritems():
        if req.vm not in useless_vms:
            values.append(req)
            times.append(t)
    return pd.TimeSeries(values, times)
    return requests

class Environment(object):
    """provides data about all the data centers
    - e.g. the temperature and prices at different location

    """
    def __init__(self):
        pass

    def __repr__(self):
        return repr({'start': self.start, 'end': self.end,
                     'period': self.period,
                     '_forecast_periods': self._forecast_periods})

    def current_data(self):
        """return all the current data for all the locations"""
        raise NotImplemented

class SimulatedEnvironment(Environment):
    """stores and provides simulated data about the environment
    - e.g. the temperature and prices at different location

    """
    def __init__(self, *args):
        super(SimulatedEnvironment, self).__init__()
        self._t = None

    def set_time(self, t):
        self._t = t

    def get_time(self):
        return self._t

    t = property(get_time, set_time, doc="current time")

    def set_period(self, period):
        self._period = period

    def get_period(self):
        return self._period

    period = property(get_period, set_period, doc="period between time steps")

    def set_forecast_periods(self, num_periods):
        self._forecast_periods = num_periods

    def get_forecast_periods(self, num_periods):
        return self._forecast_periods

    forecast_periods = property(get_forecast_periods, set_forecast_periods,
                                doc="number of periods we get forecasts for")

    def get_forecast_end(self): # TODO: parametrize
        return self._t + self._forecast_periods * self._period

    forecast_end = property(get_forecast_end, doc="time by which we forecast")

    def current_data(self, forecast=True):
        """Return el. prices and temperatures from now to forecast_end with
        optional forecasting error (for forecast=True).

        """
        if forecast:
            if hasattr(self, 'forecast_el'):
                el_prices = self.forecast_el[self.t:self.forecast_end]
            elif hasattr(self, 'real_forecasts'):
                el_prices = self.real_forecasts[self.t:self.forecast_end]
            elif hasattr(self, 'forecast_data_map'):
                el_prices = self.forecast_data_map[self.t]['values']
            else:
                el_prices = self.el_prices[self.t:self.forecast_end]
        else: 
            el_prices = self.el_prices[self.t:self.forecast_end]

        if self.temperature is not None:
            if forecast and hasattr(self, 'forecast_temp'):
                temperature = self.forecast_temp[self.t:self.forecast_end]
            else:
                temperature = self.temperature[self.t:self.forecast_end]
        return el_prices, temperature

    def _generate_forecast(self, data, SD):
        return data + SD * np.random.randn(*data.shape) # data.shape returns the dimensions of the data matrix
                                                        # to be able to add to the existing matrix

    def _get_forecast_values(self, price, horizon): 
        return self._get_forecast_dummy_values(price, 0.1, horizon)

    def _get_forecast_dummy_values(self, price, SD, horizon): 
        dummy_vals = range(int(price+1), int(price + horizon+1)) # dummy values for forecasts
        # dummy_vals = []
        # for i in range(1,horizon):
        #     val = price + SD * np.random.random()
        #     dummy_vals.append(val)
        return dummy_vals

    def model_forecast_errors(self, SD_el, SD_temp):
        self.forecast_el = self._generate_forecast(self.el_prices, SD_el)
        if not self.temperature is None:
            self.forecast_temp = self._generate_forecast(self.temperature, SD_temp)



class PPSimulatedEnvironment(SimulatedEnvironment):
    """Peak pauser simulation scenario with one location, el price"""
    pass

# TODO: merge these two with SimulatedEnvironment
class FBFSimpleSimulatedEnvironment(SimulatedEnvironment):
    """Couple of requests in a day."""
    def __init__(self, times=None, requests=None, forecast_periods=24):
        """@param times: list of time ticks"""
        super(SimulatedEnvironment, self).__init__()
        if not times is None:
            self._times = times
            self._period = times[1] - times[0]
            self._t = self._times[0]
            self.start = self._times[0]
            self.end = self._times[-1]
            self.locations = None # set in simulator
            if requests is not None:
                self._requests = requests
            else:
                self._requests = inputgen.normal_vmreqs(self.start, self.end)

        else:
            self._t = 0
            self._period = 1
            self.el_prices = []
            self.temperature = []
        self._forecast_periods = forecast_periods

    # TODO: better to make the environment immutable
    def itertimes(self):
        """Generator that iterates over times. To be called by the simulator."""
        for t in self._times:
            self._t = t
            yield t

    def itertimes_immutable(self):
        """Like itertimes, but doesn't alter state. Goes from start to end."""
        t = self.start
        while t <= self.end:
            yield t
            t += self._period

    def times_index(self):
        """Return a pandas Index based on the set times"""
        idx = pd.date_range(start=self.start, end=self.end, freq=self.period)
        return idx

    def forecast_window_index(self):
        """Return a pandas Index from t to forecast_end"""
        idx = pd.date_range(start=self.t, end=self.forecast_end,
                            freq=self.period)
        return idx

    def get_requests(self, clean=True):
        start = self.get_time()
        justabit = pd.offsets.Micro(1)
        end = start + self._period - justabit
        if clean:
            #TODO: if same vm booted & deleted at once, skip it
            return cleaned_requests(self._requests[start:end])
        else:
            return self._requests[start:end]

    def get_request_type(self, what, clean=True):
        """Get requests for type 'what'
        which is either 'boot' or 'delete' """
        start = self.get_time()
        justabit = pd.offsets.Micro(1)
        end = start + self._period - justabit
        requ_vms = set([req.vm for req in self._requests.values if req.what == what])
        if clean:
            return cleaned_requests(requ_vms[start:end])
        else:
            return requ_vms[start:end]


class GASimpleSimulatedEnvironment(FBFSimpleSimulatedEnvironment):
    pass

class SimpleSimulatedEnvironment(FBFSimpleSimulatedEnvironment):
    def get_end_request_for(self, vm):
        start = self.start
        end = self.end
        requests = self._requests
        request = set([req.vm for req in requests.values if req.what == 'delete' and req.vm == vm])
        return request

    
class BCUSimulatedEnvironment(FBFSimpleSimulatedEnvironment):
    """Create environment suitable for the best cost utility scheduler"""
    def __init__(self, times=None, requests=None, forecast_periods=24):
        """@param times: list of time ticks"""
        super(FBFSimpleSimulatedEnvironment, self).__init__()
        if not times is None:
            self._times = times
            self._period = times[1] - times[0]
            self._t = self._times[0]
            self.start = self._times[0]
            self.end = self._times[-1]
            self.locations = None # set in simulator
            if requests is not None:
                self._requests = requests
            else:
                self._requests = inputgen.normal_vmreqs(self.start, self.end)
            [vm_start, vm_end, vm_duration] = self._get_vm_durations()
            self.vm_start = vm_start
            self.vm_end = vm_end
            self.vm_duration = vm_duration
            self.vm_sla_ths = self._get_vm_sla_ths()
            self.servers_per_loc = {}

        else:
            self._t = 0
            self._period = 1
            self.el_prices = []
            self.temperature = []
        self._forecast_periods = forecast_periods


    def get_vms_from_requests(self):
        """get all vms in stored requests """
        requests = self._requests
        vms = [req.vm for req in requests.values]
        return vms

    def _get_vm_durations(self):
        """get the start, end and duration of each vm 
        and store them in separate arrays 

        """
        requests = self._requests
        vms = [req.vm for req in requests.values]

        vm_start = {}
        vm_end = {}
        vm_duration = {}

        for vm in vms:
            [start, end, duration] = self._get_vm_duration(vm)
            vm_start[vm] = start
            vm_end[vm] = end
            vm_duration[vm] = duration

        return [vm_start, vm_end, vm_duration]

    def _get_vm_duration(self, vm):
        """get the start, end and duration for a single vm"""
        start = self.start
        end = self.end
        requests = self._requests

        # iterate through all requests to find 
        # the start and end times of this vm
        for t, req in requests.iteritems():
            if req.what == 'boot' and req.vm == vm:
                start = t
            if req.what == 'delete' and req.vm == vm:
                end = t

        # e.g. duration of 3 hours:
        # datetime.timedelta(0, 10800)
        duration = end - start
        return [start, end, duration]

    def get_remaining_duration(self, vm, t=None):
        """get the remaining duration for a given vm"""
        if t is None:
            start = self.get_time()
        else:
            start = t

        end = self.vm_end[vm]
        # # e.g. duration of 3 hours:
        # # datetime.timedelta(0, 10800)
        return end - start

    def _get_vm_sla_ths(self):
        """Save a list of threshold values (in seconds) 
        for penalties applicable for each vm
        """
        requests = self._requests
        vms = [req.vm for req in requests.values]

        vm_sla_ths = {}
        for vm in vms:
            vm_sla_ths[vm] = self._get_sla_th(vm)

        return vm_sla_ths

    def _get_sla_th(self, vm):
        """get list of sla penalty thresholds (in seconds)
        applicable to the given vm based on its expected 
        total duration
        """
        duration = self.vm_duration[vm].total_seconds()
        penalty_mappings = self._get_sla_penalty_mapping(vm.sla)
        sla_ths = []
        for pen_level in penalty_mappings:
            sla_ths.append(duration*(1 - pen_level / 100.0))
        return sla_ths

    def get_sla_penalty(self, vm):
        """get penalty applicable for this vm 
        based on its accumulated downtime
        """
        if vm.downtime > self.vm_sla_ths[vm][2]:
            return 1.5
        if vm.downtime > self.vm_sla_ths[vm][1]:
            return 1.25
        if vm.downtime > self.vm_sla_ths[vm][0]:
            return 1.1

    def _get_sla_penalty_mapping(self, sla):
        """get sla penalty mapping for each defined sla
        to decide when a new penalty schema should be applied
        """
        if sla == 99.95:
            return [99.95, 99, 95]
        if sla == 99.9:
            return [99.9, 99, 95]
        if sla == 99:
            return [99, 98, 95]

    def update_sla(self, vm):
        """update the penalty status of a vm"""
        if vm.penalties < 3 and \
            vm.downtime > self.vm_sla_ths[vm][vm.penalties]:
                vm.penalties += 1



class ForecastSimulatedEnvironment(FBFSimpleSimulatedEnvironment):

    def _retrieve_forecasts(self):
        # retrieve forecasts from web service
        import urllib2
        url = 'http://localhost:8081/em-app/rest/r/forecastAll/1,3,4/14/2014-07-07/2014-07-10'
        response = urllib2.urlopen(url).read()
        with open("csvData.csv", "w") as csv_file:
            csv_file.write(response)
        df = pd.read_csv('csvData.csv', engine='python', parse_dates=[0], escapechar='\\', index_col=0)
        return df

    def get_real_forecasts(self):
        print("get real forecasts for simulation period (REST interface)")
        self.real_forecasts = self._retrieve_forecasts()

    def _generate_forecast_map(self, el_prices, forecast_periods=5, forecast_freq='H'):
        """ Generate a map to assign forecast values to each timestamp
            1. create mapping -> timestamp to DataFrame
            2. iterate through each timestamp
            3. At each timestamp add forecasts for each location to DataFrame
                  for given forecast_periods (fc horizon)
            4. Return mapping over simlation period

        """
        locations = el_prices.axes[1]
        price_df_list = []
        
        #iterate through all simulation times
        for index, prices in el_prices.iterrows():
            # start with the next hour
            ind = index + pd.DateOffset(hours=1)
            ind = pd.date_range(start=ind, periods=forecast_periods, freq=forecast_freq)
            price_df = pd.DataFrame(index=ind, columns=locations)
            for loc in locations:
                price = prices[loc]
                values = self._get_forecast_values(price, forecast_periods)
                price_df[loc] = values
            
            # add to df list
            price_df_list.append(price_df)

        data_map = pd.DataFrame(index=el_prices.index,
                                data={'values': price_df_list})
        # Sample df entry retrieval
        # idx = pd.Timestamp('2014-07-07 00:00:00')  # or just '2014-07-07 00:00:00'
        # print "map at index {}: {}".format(idx, data_map.loc[idx]['values'])
        return data_map

    def get_real_forecast_map(self, forecast_periods=5, forecast_freq='H'):
        print("get real forecasts for {} periods".format(forecast_periods))
        self.forecast_data_map = self._generate_forecast_map(self.el_prices, forecast_periods, forecast_freq)