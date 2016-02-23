from .base import *
from philharmonic.logger import *
from philharmonic import conf

output_folder = os.path.join(base_output_folder, "bcu/")



# scheduler configuration

# scenarios: 
# 1) Best fit decreasing Scheduler: non cost aware request scheduling (assignments based on load)
# 2) Best cost fit Scheduler: cost aware request scheduling (assign to currently cheapest DC)
# 3) Best cost fit Scheduler: cost aware request scheduling + forecast (assign to cheapest DC based on forecasts and job length)
# 4) Best cost fit Scheduler: cost aware request scheduling + ideal forecast (assign to cheapest DC based on ideal forecasts and job length)
# 5) Best cost utility scheduler: No Forecast, migrations. Evaluate utility function for migrations without forecasts
# 6) Best cost utility scheduler: Forecast, migrations. Evaluate utility function for migrations including forecasts
# 7) Best cost utility scheduler: Ideal Forecast, migrations. Evaluate utility function for migrations including ideal forecasts




# scenarios: 
# 1) Balanced fit Scheduler: No forecast, no migrations. not cost aware (assignments based on load, no migrations)
# 2) Best cost fit Scheduler: No forecast, no migrations. cost aware request scheduling (assign to currently cheapest DC)
# 3) Best cost utility scheduler: No forecast, no migrations. Evaluate utility function for the current time stamp
# 4) Best cost utility scheduler: Forecast, no migrations. Evaluate utility function including forecasts
# 5) Best cost utility scheduler: Ideal Forecast, no migrations. Evaluate utility function including ideal forecasts
# 6) Best cost utility scheduler: No Forecast, migrations. Evaluate utility function for assignments and migrations without forecasts
# 7) Best cost utility scheduler: Forecast, migrations. Evaluate utility function for assignments and migrations including forecasts
# 8) Best cost utility scheduler: Ideal Forecast, migrations. Evaluate utility function for assignments and migrations including ideal forecasts



# possible scenarios: 
# 1) not cost aware (random assignments, migrations based on load?)
# 2) cost aware request scheduling (assign to currently cheapest DC)
# 3) cost aware request scheduling + forecast (assign to cheapest DC based on forecasts and job length)
# 4) cost aware request scheduling + ideal forecast (assign to cheapest DC based on ideal forecasts and job length)
# 5) cost aware requests and migrations (assign to currently cheapest DC)
# 6) cost aware requests and migrations + forecast (assign to cheapest DC based on forecasts and job length)
# 7) cost aware requests and migrations + ideal forecast (assign to cheapest DC based on ideal forecasts and job length)

bcuconf = {
	'scenario': 6
}

factory['scheduler'] = 'BCUScheduler'
factory['scheduler_conf'] = bcuconf
factory['environment'] = 'BCUSimulatedEnvironment'


# Input gen settings

inputgen_settings['resource_distribution'] = 'uniform' # uniform or normal

inputgen_settings['server_num'] = 10
inputgen_settings['min_server_cpu'] = 4 # 16,
inputgen_settings['max_server_cpu'] = 8 # 16,
inputgen_settings['min_server_ram'] = 8 # 32,
inputgen_settings['max_server_ram'] = 16 # 32,

inputgen_settings['VM_num'] = 20
inputgen_settings['min_cpu'] = 1 # 2,
inputgen_settings['max_cpu'] = 4 # 4,
inputgen_settings['min_ram'] = 1 # 4,
inputgen_settings['max_ram'] = 4 # 16,

# inputgen_settings['min_duration'] = 60 * 5 # 5 minute
# inputgen_settings['max_duration'] = 3600 * 2 # 2 hours

# inputgen_settings['min_duration'] = 60 * 5 # 5 minute
# inputgen_settings['min_duration'] = 60 * 5 # 5 minute

# inputgen_settings['min_duration'] = 60 * 60 # 1 hour
# inputgen_settings['max_duration'] = 60 * 60 * 3 # 3 hours

# only take into account when fixed_duration is False
inputgen_settings['min_duration'] = 60 * 60 * 1 
inputgen_settings['max_duration'] = 60 * 60 * 5 # 5 hours

# inputgen_settings['min_duration'] = 60, # 1 minute
# inputgen_settings['max_duration'] = 60 * 60 * 10, # ten hours

inputgen_settings['cloud_infrastructure'] = 'uniform_infrastructure' # uniform_infrastructure or normal_infrastructure
inputgen_settings['VM_request_generation_method'] = 'uniform_vmreqs' # uniformly generate vm requests
inputgen_settings['round_to_hour'] = True
inputgen_settings['show_beta_value'] = False




#################################
#####	Utility function 	#####
#################################

#### Defining CONSTANTS ####

max_fc_horizon = 12

# weights

# w_sla = 0.8
# w_energy = 0.1
# w_vm_rem = 0.4
# w_dcload = 0.2
# w_cost = 0.9
w_sla = 0.9
w_energy = 1.0
w_vm_rem = 0.1
w_dcload = 0.1
w_cost = 0.5

# threshold for deciding on vm migration
# the sum of all utility values except costs (w_cost) should be smaller
# than this utility threshold, to make it impossible to reach this 
# threshold without a positive value of the cost utility

utility_threshold = 2.0


# to evaluate: 
# simulation, inputgen 2016-02-18 200211
# 
# 2014-07-11 20:00 - normal u value (1.5)
# 2014-07-17 10:00 - high u value (2.7)
# 2014-07-18 21:00 - ultra high u value (4.4)



### simulation parameters ###

# dirty page rates applied to vms
dpr_values = [20,40,70,90]
# possible slas to apply to vms
sla_values = [99.95,99.9,99]
# list of distinct vm duration values (in hours)
# if None they will be assigned by a resource distribution
duration_values = [1,2,5,8,12,24,48]
# indicates whether different dirty page rates should be assigned to vms
generate_dpr = True
# indicates whether different sla values should be assigned to vms
generate_sla = False
# indicates whether fixed duration values (see above) should be used
fixed_duration = True
# indicates whether it is possible that vms run for the whole duration of the simulation
total_duration = True
# indicates whether the sla status of vms should be continuously updated in the simulator
update_vm_sla = True
# sets a threshold for price differences, only if the maximum absolute price differences 
# are above the threshold will vms be chosen for migration for the current iteration
min_price_threshold = 0.001
# minimum remaining duration of vm in simulation timesteps to be considered for migration
min_vm_remaining = 1



bandwidth_da = {
	'Hamina': {
		'Potsdam': 1000, 'St.Ghislain': 800, 'Portland': 400, 'Boston': 400
	},
	'Potsdam': {
		'Hamina': 1000, 'St.Ghislain': 800, 'Portland': 400, 'Boston': 400
	},
	'St.Ghislain': {
		'Hamina': 1000, 'Potsdam': 800, 'Portland': 400, 'Boston': 400
	},
	'Portland': {
		'Hamina': 400, 'St.Ghislain': 400, 'Potsdam': 400, 'Boston': 800
	},
	'Boston': {
		'Hamina': 400, 'St.Ghislain': 400, 'Potsdam': 400, 'Portland': 800
	}
}

bandwidth_rt = {
	'Portland': {
		'Boston': 800, 'Richmond': 400, 'Brighton': 400, 'Hatfield': 400, 'Madison': 400, 'Georgetown': 400
	},
	'Boston': {
		'Portland': 800, 'Richmond': 400, 'Brighton': 400, 'Hatfield': 400, 'Madison': 400, 'Georgetown': 400
	},
	'Richmond': {
		'Brighton': 1000, 'Hatfield': 1000, 'Madison': 1000, 'Georgetown': 1000, 'Portland': 400, 'Boston': 400
	},
	'Brighton': {
		'Richmond': 1000, 'Hatfield': 1000, 'Madison': 1000, 'Georgetown': 1000, 'Portland': 400, 'Boston': 400
	},
	'Hatfield': {
		'Richmond': 1000, 'Brighton': 1000, 'Madison': 1000, 'Georgetown': 1000, 'Portland': 400, 'Boston': 400
	},
	'Madison': {
		'Richmond': 1000, 'Brighton': 1000, 'Hatfield': 1000, 'Georgetown': 1000, 'Portland': 400, 'Boston': 400
	},
	'Georgetown': {
		'Richmond': 1000, 'Brighton': 1000, 'Hatfield': 1000, 'Madison': 1000, 'Portland': 400, 'Boston': 400
	}
}



#### General Settings ####

# DATA_LOC_SIM_DA: DA input from USA and Europe (ISO-NE, PJM, NordPoolSpot)
# DATA_LOC_SIM_RT: RT input from USA (ISO-NE, PJM)

sim_type = "DA" # DA or RT

if sim_type == "DA":
	DATA_LOC = DATA_LOC_SIM_DA

	el_price_dataset = os.path.join(DATA_LOC, 'prices_da_Jun_July_2013.csv')
	el_price_forecast = os.path.join(DATA_LOC, 'prices_da_fc_Jun_July_2013.csv')

elif sim_type == "RT":
	DATA_LOC = DATA_LOC_SIM_RT

	el_price_dataset = os.path.join(DATA_LOC, 'prices_rt_Jun_July_2013.csv')
	el_price_forecast = os.path.join(DATA_LOC, 'prices_rt_fc_Jun_July_2013.csv')


# if empty, bandwidth will be assigned a fixed value
bandwidth_map = {}
if sim_type == "DA":
	bandwidth_map = bandwidth_da

elif sim_type == "RT":
	bandwidth_map = bandwidth_rt

# bandwidth_map = {}

bandwidth_costs = 0.001 # $ / GB

vm_price = 0.04 # $ / h


add_date_to_folders = True

dynamic_locations = True

prompt_configuration = True


power_freq_model = False
power_randomize = False
show_pm_frequencies = False
save_power = True
save_util = True
transform_to_jouls = False
prices_in_mwh = True
alternate_cost_model = True
location_based = True

# the frequency at which to generate the power signals
power_freq = 'H' # '5min'
# various power values of the servers in Watt hours
P_peak = 200
P_idle = 100

# show_cloud_interval = pd.offsets.Hour(12) # interval at which simulation output should be done

show_cloud_interval = None

# generate_forecasts: forecasts will be generated based on a given standard deviation
# local_forecasts: forecasts will be read from file, specified under property el_price_forecast
# real_forecasts: "real" forecasts will be retrieved from server (web service)
# real_forecast_map: "real" forecasts will be retrieved from server for each hour separately
factory['forecast_type'] = 'local_forecasts'
# possible values: el_prices_from_conf, mixed_2_loc
factory['el_prices'] = 'el_prices_from_conf'
# possible values: forecast_el_from_conf, mixed_2_loc_fc
factory['forecast_el'] = 'forecast_el_from_conf'
factory['temperature'] = None
factory['forecast_periods'] = 5
factory['SD_el'] = 0
factory['clean_requests'] = False


# plotting on server, no X server session
plotserver = False

# show plots on the fly instead of saving to a file (pdf)
plot_live = False

if plot_live:
    liveplot = True
    fileplot = False
else:
    liveplot = False
    fileplot = True


# date_parser = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')


# TODO Andreas: Make this setting dynamic!

# probable possibility: 
# start = prices.index[0]
# end= prices.index[-1]

# the time period of the simulation
start = pd.Timestamp('2013-06-20 00:00')

# TODO Andreas: Make this setting dynamic, or define as property in each settings file
times = pd.date_range(start, periods=24 * 14, freq='H') # * 38
end = times[-1]

custom_weights = {'RAM': 0.3, '#CPUs': 0.7}
# custom_weights = None
