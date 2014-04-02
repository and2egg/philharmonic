'''
Created on Oct 9, 2012

@author: kermit
'''

# I/O
#======

historical_en_prices = "./io/energy_price/train/3month.csv"
#historical_en_prices = "./io/energy_price_data-quick_test.csv"
#historical_en_prices = "./io/energy_price_data-single_day.csv"

results = "./io/results.pickle"

# Manager
#=========
# Manager - actually sleeps and wakes up the scheduler
# Simulator - just runs through the simulation
manager = "Simulator"

# Manager factory
#=================

# won't have to be function once I kick out conf from PeakPauser
def get_factory_fbf():
    # these schedulers are available:
    from philharmonic.scheduler import PeakPauser, NoScheduler, FBFScheduler

    # environments
    from philharmonic.simulator.environment import FBFSimpleSimulatedEnvironment

    from philharmonic.simulator import inputgen
    from philharmonic.cloud.driver import simdriver

    factory = {
        "scheduler": FBFScheduler,
        "environment": FBFSimpleSimulatedEnvironment,
        "cloud": inputgen.small_infrastructure,
        "driver": simdriver,

        "times": inputgen.two_days,
        "requests": inputgen.normal_vmreqs,
        "servers": inputgen.small_infrastructure,
    }


    return factory

def get_factory_ga():
    # these schedulers are available:
    from philharmonic.scheduler import FBFScheduler, GAScheduler

    # environments
    from philharmonic.simulator.environment import GASimpleSimulatedEnvironment

    from philharmonic.simulator import inputgen
    from philharmonic.cloud.driver import simdriver

    gaconf = {
        "population_size": 100,
        "recombination_rate": 0.15,
        "mutation_rate": 0.05,
        "max_generations": 100,
        "random_recreate_ratio": 0.3,
        "no_temperature": False,
        "no_el_price": False,
    }

    factory = {
        "scheduler": GAScheduler,
        #"scheduler": FBFScheduler,
        "scheduler_conf": gaconf,
        "environment": GASimpleSimulatedEnvironment,
        #"cloud": inputgen.small_infrastructure,
        #"cloud": inputgen.usa_small_infrastructure,
        #"cloud": inputgen.servers_from_pickle,
        #"cloud": inputgen.servers_from_pickle,
        "cloud": inputgen.dynamic_infrastructure,

        "forecast_periods": 12,
        ### no error
        "SD_el": 0,
        "SD_temp": 0,

        ### small error
        #"SD_el": 0.01,
        #"SD_temp": 1.41,

        ### medium error
        #"SD_el": 0.03,
        #"SD_temp": 3,

        ### large error
        #"SD_el": 0.05,
        #"SD_temp": 5,

        #"times": inputgen.two_days,
        #"times": inputgen.usa_two_days,
        #"times": inputgen.usa_two_hours,
        #"times": inputgen.usa_three_months,
        "times": inputgen.dynamic_usa_times,
        #"times": inputgen.usa_whole_period,
        #"requests": inputgen.simple_vmreqs,
        "requests": inputgen.medium_vmreqs,
        #"requests": inputgen.requests_from_pickle,

        #"el_prices": inputgen.simple_el,
        #"el_prices": inputgen.medium_el,
        #"el_prices": inputgen.usa_el,
        "el_prices": inputgen.dynamic_usa_el,
        #"temperature": inputgen.simple_temperature,
        #"temperature": inputgen.medium_temperature,
        #"temperature": inputgen.usa_temperature,
        "temperature": inputgen.dynamic_usa_temp,

        "driver": simdriver,
    }

    return factory

def get_factory():
    return get_factory_ga()

# Simulator settings
#===========================

#plotserver = True
plotserver = False
liveplot = True
#liveplot = False

inputgen_settings = {
    # cloud's servers
    'server_num': 20,
    'min_server_cpu': 4,
    'max_server_cpu': 8,

    # VM requests
    'VM_num': 80,
    #'VM_num': 2000,
    # e.g. CPUs
    'min_cpu': 1,
    'max_cpu': 2,
    'min_ram': 1,
    'max_ram': 2,
    # e.g. seconds
    'min_duration': 60 * 60, # 1 hour
    #'max_duration': 60 * 60 * 3, # 3 hours
    #'max_duration': 60 * 60 * 24 * 10, # 10 days
    'max_duration': 60 * 60 * 24 * 90, # 90 days
}

# Benchmark
#===========

# if dummy == True, will do just a local dummy benchmark,
# faking all the OpenStack commands
dummy = True
#dummy = False
# for False set all these other settings...

# host on which the benchmark VM is deployed (for energy measurements)
host = "snowwhite"

# VM (instance) which executes the benchmark
instance = "kermit-test"

# The command to execute as a benchmark (use ssh to execute something in a VM).
# If command=="noscript" then just some local execution will be done
#command="noscript"
#command = "/usr/bin/ssh 192.168.100.4 ls"
#command = "./io/benchmark.sh"
command = "./io/benchmark-local.sh"


# how many % of hours in a day should the VM be paused
#percentage_to_pause = 0.04 # *100%
percentage_to_pause = 0.15 # *100%

# time to sleep between checking if the benchmark finished or needs to be paused
sleep_interval = 1 # seconds
