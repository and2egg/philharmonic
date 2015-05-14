from .base import *
from philharmonic.logger import *
from philharmonic import conf

output_folder = os.path.join(base_output_folder, "simple/")

inputgen_settings['server_num'] = 10
inputgen_settings['VM_num'] = 20

add_date_to_folders = True

prompt_configuration = True

power_freq_model = False