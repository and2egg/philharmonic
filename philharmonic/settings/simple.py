from .base import *
from philharmonic.logger import *
from philharmonic import conf

output_folder = os.path.join(base_output_folder, "simple/")



# scheduler configuration

# possible scenarios: 
# 1) not cost aware (random assignments, migrations based on load?)
# 2) cost aware request scheduling (take cheapest dc?)
# 3) cost aware request scheduling + forecast (take cheapest dc?)
# 4) cost aware requests and migrations (take cheapest dc?)
# 5) cost aware requests and migrations + forecast (take cheapest dc?)
# 6) cost aware requests and migrations + ideal forecast (take cheapest dc?)

simpleconf = {
	'scenario': 6
}


# Input gen settings

inputgen_settings['resource_distribution'] = 'normal'

inputgen_settings['server_num'] = 3
inputgen_settings['min_server_cpu'] = 8 # 16,
inputgen_settings['max_server_cpu'] = 8 # 16,
inputgen_settings['min_server_ram'] = 16 # 32,
inputgen_settings['max_server_ram'] = 16 # 32,

inputgen_settings['VM_num'] = 20
inputgen_settings['min_cpu'] = 4 # 2,
inputgen_settings['max_cpu'] = 4 # 4,
inputgen_settings['min_ram'] = 2 # 4,
inputgen_settings['max_ram'] = 2 # 16,

# inputgen_settings['min_duration'] = 60 * 5 # 5 minute
# inputgen_settings['max_duration'] = 3600 * 2 # 2 hours

# inputgen_settings['min_duration'] = 60 * 5 # 5 minute
# inputgen_settings['min_duration'] = 60 * 5 # 5 minute

# inputgen_settings['min_duration'] = 60 * 60 # 1 hour
# inputgen_settings['max_duration'] = 60 * 60 * 3 # 3 hours

inputgen_settings['min_duration'] = 60 * 60 * 5 # 5 hours
inputgen_settings['max_duration'] = 60 * 60 * 5 # 5 hours

# inputgen_settings['min_duration'] = 60, # 1 minute
# inputgen_settings['max_duration'] = 60 * 60 * 10, # ten hours

inputgen_settings['cloud_infrastructure'] = 'uniform_infrastructure'
inputgen_settings['VM_request_generation_method'] = 'normal_vmreqs_interval' # ? TODO check appropriate request gen method
inputgen_settings['round_to_hour'] = False
inputgen_settings['show_beta_value'] = False

# General Settings

MIXED = True

ignore_ram = True

add_date_to_folders = True

dynamic_locations = True

prompt_configuration = True

show_pm_frequencies = False

power_freq_model = False

power_randomize = False

save_power = True

save_util = True

transform_to_jouls = False

prices_in_mwh = True

alternate_cost_model = True

# show_cloud_interval = pd.offsets.Hour(12) # interval at which simulation output should be done

show_cloud_interval = None

# possible values: el_prices_from_conf, mixed_2_loc
factory['el_prices'] = 'mixed_2_loc'
# possible values: forecast_el_from_conf, mixed_2_loc_fc
factory['forecast_el'] = 'mixed_2_loc_fc'
factory['temperature'] = None

factory['scheduler'] = 'SimpleScheduler'
factory['scheduler_conf'] = simpleconf
factory['environment'] = 'SimpleSimulatedEnvironment'

factory['forecast_periods'] = 5
factory['SD_el'] = 0.2
factory['real_forecasts'] = True
factory['local_forecasts'] = True
factory['clean_requests'] = False


# plotting on server, no X server session
plotserver = False

# show plots on the fly instead of saving to a file (pdf)
plot_live = True

if plot_live:
    liveplot = True
    fileplot = False
else:
    liveplot = False
    fileplot = True


date_parser = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')

# the time period of the simulation
start = pd.Timestamp('2014-07-07 00:00')

# - two weeks
times = pd.date_range(start, periods=24 * 1, freq='H')
end = times[-1]
