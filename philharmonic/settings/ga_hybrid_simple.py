from .ga_hybrid import *

output_folder = os.path.join(base_output_folder, "ga_hybrid_simple/")

factory['cloud'] = "small_infrastructure"
factory['requests'] = "simple_vmreqs"
    #  simple_el, medium_el, usa_el, world_el, dynamic_usa_el
    #  simple_temperature, medium_temperature, usa_temperature,
    #  world_temperature, dynamic_usa_temp
factory['el_prices'] = "simple_el"
factory['temperature'] = "simple_temperature"


inputgen_settings['server_num'] = 10
inputgen_settings['VM_num'] = 20
