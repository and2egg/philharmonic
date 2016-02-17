from .base import *
from philharmonic.logger import *
from philharmonic import conf

output_folder = os.path.join(base_output_folder, "bcd/")



# scheduler configuration

# possible scenarios: 
# 1) not cost aware (random assignments, migrations based on load?)
# 2) cost aware request scheduling (assign to currently cheapest DC)
# 3) cost aware request scheduling + forecast (assign to cheapest DC based on forecasts and job length)
# 4) cost aware request scheduling + ideal forecast (assign to cheapest DC based on ideal forecasts and job length)
# 5) cost aware requests and migrations (assign to currently cheapest DC)
# 6) cost aware requests and migrations + forecast (assign to cheapest DC based on forecasts and job length)
# 7) cost aware requests and migrations + ideal forecast (assign to cheapest DC based on ideal forecasts and job length)

bcdconf = {
	'scenario': 6
}

factory['scheduler'] = 'BCDScheduler'
factory['scheduler_conf'] = bcdconf
factory['environment'] = 'SimpleSimulatedEnvironment'


# Input gen settings

inputgen_settings['resource_distribution'] = 'uniform' # uniform or normal

inputgen_settings['server_num'] = 20
inputgen_settings['min_server_cpu'] = 4 # 16,
inputgen_settings['max_server_cpu'] = 8 # 16,
inputgen_settings['min_server_ram'] = 8 # 32,
inputgen_settings['max_server_ram'] = 16 # 32,

inputgen_settings['VM_num'] = 100
inputgen_settings['min_cpu'] = 1 # 2,
inputgen_settings['max_cpu'] = 2 # 4,
inputgen_settings['min_ram'] = 0.5 # 4,
inputgen_settings['max_ram'] = 2 # 16,

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



#### General Settings ####

# DATA_LOC_SIM_DA: DA input from USA and Europe (ISO-NE, PJM, NordPoolSpot)
# DATA_LOC_SIM_RT: RT input from USA (ISO-NE, PJM)

sim_type = "DA" # DA or RT

if sim_type == "DA":
	DATA_LOC = DATA_LOC_SIM_DA

	el_price_dataset = os.path.join(DATA_LOC, 'prices_da_all.csv')
	el_price_forecast = os.path.join(DATA_LOC, 'prices_da_fc_all.csv')

elif sim_type == "RT":
	DATA_LOC = DATA_LOC_SIM_RT

	el_price_dataset = os.path.join(DATA_LOC, 'prices_rt_all.csv')
	el_price_forecast = os.path.join(DATA_LOC, 'prices_rt_fc_all.csv')

add_date_to_folders = True

dynamic_locations = True

prompt_configuration = True





#################################
#####	Utility function 	#####
#################################

#### Defining CONSTANTS ####

max_fc_horizon = 12

# weights

w_sla = 0.8
w_energy = 0.1
w_vm_rem = 0.4
w_dcload = 0.2
w_cost = 0.9

# threshold for deciding on vm migration

utility_threshold = 0.8



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
generate_sla = True
# indicates whether fixed duration values (see above) should be used
fixed_duration = True
# indicates whether it is possible that vms run for the whole duration of the simulation
total_duration = True
# indicates whether the sla status of vms should be continuously updated in the simulator
update_vm_sla = True



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

# if empty, bandwidth will be assigned a fixed value
bandwidth_map = {}
if sim_type == "DA":
	bandwidth_map = bandwidth_da

elif sim_type == "RT":
	bandwidth_map = bandwidth_rt


power_freq_model = False
power_randomize = False
show_pm_frequencies = False
save_power = True
save_util = True
transform_to_jouls = False
prices_in_mwh = True
alternate_cost_model = True
location_based = True

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
# the time period of the simulation
start = pd.Timestamp('2014-07-07 00:00')

# TODO Andreas: Make this setting dynamic, or define as property in each settings file
times = pd.date_range(start, periods=24 * 28, freq='H')
end = times[-1]

custom_weights = {'RAM': 0.5, '#CPUs': 0.5}
# custom_weights = None
