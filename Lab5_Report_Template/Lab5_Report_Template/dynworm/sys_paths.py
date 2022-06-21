
# coding: utf-8

############################################################
#PATHS FOR WINDOWS AND UNIX BASED OPERATING SYSTEMS#########
############################################################

import os
import platform

platform = platform.system()
default_dir = os.getcwd()

if platform == 'Windows':

    main_dir = default_dir + '\dynworm'
    connectome_data_dir = main_dir + '\connectome_data'

else:

    main_dir = default_dir + '/dynworm'
    connectome_data_dir = main_dir + '/connectome_data'

