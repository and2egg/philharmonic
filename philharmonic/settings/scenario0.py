from .base import *
from philharmonic.logger import *
from philharmonic import conf



# I/O
#======

save_power = True
save_util = True
#----------------------

# stop and check settings with user (only initially)
prompt_configuration = True
# interval at which to print cloud usage: pd.offsets.* or None
# show_cloud_interval = pd.offsets.Hour(12)
show_cloud_interval = None
# stop the simulation for inspection?
prompt_show_cloud = False
prompt_ipdb = True

# control whether the input and/or output folders should be time-stamped
add_date_to_folders = True


# the datasets used in the simulation
USA = False # USA or world-wide
# MIXED = True
FIXED_EL_PRICES = False # fixed el. prices world-wide

if USA:
    DATA_LOC = DATA_LOC_USA
else:
    DATA_LOC = DATA_LOC_WORLD
    if FIXED_EL_PRICES:
        DATA_LOC = DATA_LOC_WORLD_FIXED_EL


temperature_dataset = os.path.join(DATA_LOC, 'temperatures.csv')
el_price_dataset = os.path.join(DATA_LOC, 'prices.csv')
# is read when forecast_type = local_forecasts
el_price_forecast = os.path.join(DATA_LOC, 'prices.csv')

dynamic_locations = False

date_parser = None

# the time period of the simulation
start = pd.Timestamp('2010-06-03 00:00')
# - one week
times = pd.date_range(start, periods=24 * 7, freq='H')
end = times[-1]

custom_weights = None
# custom_weights = {'RAM': 0, '#CPUs': 1}


# plotting results
plotserver = True
#plotserver = False
if plotserver: # plotting in GUI-less environment
    liveplot = True
# else: # GUI present (desktop OS)
#     liveplot = False
#     #liveplot = True
#     fileplot = True


plot_live = plotserver

if plot_live:
    liveplot = True
    fileplot = False
else:
    liveplot = False
    fileplot = True



# Manager factory
#=================

factory['cloud'] = 'servers_from_pickle'

factory['forecast_periods'] = 12 # 5
factory['SD_el'] = 0 # no error
factory['SD_temp'] = 0


# generate_forecasts: forecasts will be generated based on a given standard deviation
# local_forecasts: forecasts will be read from file, specified under property el_price_forecast
# real_forecasts: "real" forecasts will be retrieved from server (web service)
# real_forecast_map: "real" forecasts will be retrieved from server for each hour separately
factory['forecast_type'] = 'local_forecasts'

# Timestamps of the simulation. Can be:
#  times_from_conf (take times from conf.times, recommended),
#  two_days, world_three_months,
#  usa_two_hours, usa_two_days, usa_three_months
#  world_two_hours, world_two_days, world_three_months
#  dynamic_usa_times, usa_whole_period
factory['times'] = 'times_from_conf'

# VM requests. Can be:
#  requests_from_pickle (recommended), simple_vmreqs, medium_vmreqs
factory['requests'] = 'requests_from_pickle'
# offset by which to shift requests (None for no shifting)
# - mostly just for use by the explorer
factory['requests_offset'] = None
# state whether requests that last shorter than one period
# should be deleted
factory['clean_requests'] = True # False
# Geotemporal inputs. *_from_conf recommended
# (they read CSV files located at conf.*_dataset)
# Can also be:
#  simple_el, medium_el, usa_el, world_el, dynamic_usa_el
#  simple_temperature, medium_temperature, usa_temperature,
#  world_temperature, dynamic_usa_temp, mixed_2_loc
factory['el_prices'] = 'el_prices_from_conf'
# will be read when forecast_type == 'local_forecasts'
# possible values: forecast_el_from_conf, mixed_2_loc_fc
factory['forecast_el'] = 'forecast_el_from_conf'
# reads temperature data as specified in conf
factory['temperature'] = 'temperature_from_conf'


# Driver that takes the manager's actions and controls the cloud:
#  nodriver (no actions)
#  simdriver (only logs actions)
#  (real OpenStack driver not implemented yet)
factory['driver'] = 'nodriver'




# Various scheduling settings
#============================
# Percentage of utilisation under which a PM is considered underutilised
underutilised_threshold = 0.0


# inputgen settings
#==================

# general settings
    #  - the statistical distribution to draw resources and duration from
    #    ( uniform or normal)
inputgen_settings['resource_distribution'] = 'uniform' # 'normal'

# path to electricity price file
inputgen_settings['location_dataset'] = el_price_dataset  # temperature_dataset
# cloud's servers
inputgen_settings['server_num'] = 10
inputgen_settings['min_server_cpu'] = 8 # 16,
inputgen_settings['max_server_cpu'] = 8 # 16,
inputgen_settings['min_server_ram'] = 16 # 32,
inputgen_settings['max_server_ram'] = 16 # 32,

# 1 to generate beta, 2 to read them directly from file and
# 3 for all beta equal to 1
inputgen_settings['beta_option'] = 3
inputgen_settings['fixed_beta_value'] = 1.
# indicates whether the beta value should be displayed on output
inputgen_settings['show_beta_value'] = True # False
inputgen_settings['max_cloud_usage'] = 0.8
# method of generating servers for cloud infrastructure: 
# normal_infrastructure, uniform_infrastructure
inputgen_settings['cloud_infrastructure'] = 'normal_infrastructure' # uniform_infrastructure

# VM requests
# method of generating requests: normal_vmreqs, auto_vmreqs, normal_vmreqs_interval
inputgen_settings['VM_request_generation_method'] = 'auto_vmreqs' # 'normal_vmreqs_interval' # ? TODO check appropriate request gen method
inputgen_settings['round_to_hour'] = True # False
# only applicable when VM_request_generation_method is 'normal_vmreqs_interval' 
inputgen_settings['vm_req_interval'] = '5min'

# number of VMs to generate in requests
# only important with normal_vmreqs, not auto_vmreqs (?)
inputgen_settings['VM_num'] = 20
inputgen_settings['min_cpu'] = 4 # 2,
inputgen_settings['max_cpu'] = 4 # 4,
inputgen_settings['min_ram'] = 2 # 4,
inputgen_settings['max_ram'] = 2 # 16,
# e.g. seconds
inputgen_settings['min_duration'] = 60 * 60 * 5 # 5 hours
inputgen_settings['max_duration'] = 60 * 60 * 5 # 5 hours



# Simulation details
#===================

# the frequency at which to generate the power signals
power_freq = '5min'
# various power values of the servers in Watt hours
P_peak = 200
P_idle = 100
# the standard deviation of the power signal
# P_std = 5
P_std = 0
P_base = 150
P_dif = 15

power_randomize = True
# power_randomize = False

# VM cost components for ElasticHosts
C_base = 0.027028 #0.0520278  # in future use C_base = 0.027028
C_dif_cpu = 0.018
C_dif_ram = 0.025
# CPU frequency parameters
f_max = 2600 # the maximum CPU frequency in MHz
power_freq_model = True # consider CPU frequency in the power model
# power_freq_model = False

# VM cost components for ElasticHosts
#C_base = 0.004529 # $/hour #C_base=0.012487, C_dif_ram=0 if we don't vary memory
#C_dif_cpu = 0.001653 # $/hour
#C_dif_ram=0.007958 # $/hour
#rel_ram_size=2 #at least 2: min ram size charged

show_pm_frequencies = True
# show_pm_frequencies = False

# TODO: apply to model
freq_scale_max = 1.0
freq_scale_digits = 5
freq_scale_min = round(1800./2600, 1) # = 0.7
freq_scale_delta = round((freq_scale_max - freq_scale_min) / 4,
                         freq_scale_digits) # = 0.075
f_min = f_max * freq_scale_min
f_base = 1000

# freq_scales = np.round(frange(freq_scale_max, freq_scale_min,
#                                  delta=-freq_scale_delta),
#                        freq_scale_digits)

# pricing
# the frequency at which to generate the VM price is calculated
pricing_freq = '1h'

# specifies the network bandwidth between each two data centers (in MBit/s)
fixed_bandwidth = 100

# checks whether a price transformation from kWh (MWh) to jouls will be done
transform_to_jouls = True
# transform_to_jouls = False

prices_in_mwh = False
# prices_in_mwh = True

alternate_cost_model = False
# alternate_cost_model = True

location_based = False
# location_based = True