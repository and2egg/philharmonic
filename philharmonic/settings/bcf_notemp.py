from .baseprod import *

output_folder = os.path.join(base_output_folder, "bcf_notemp/")

factory['scheduler'] = "BCFNoTempScheduler"
