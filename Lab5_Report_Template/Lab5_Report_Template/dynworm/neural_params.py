# coding: utf-8

###########################################
#NEURAL PARAMETERS FOR NETWORK SIMULATIONS#
###########################################

"""
N: Number of Neurons
Gc: Cell membrane conductance (pS)
C: Cell Membrane Capacitance
ggap: Gap Junctions scaler (Electrical, 279*279)
gsyn: Synaptic connections scaler (Chemical, 279*279)
Ec: Leakage potential (mV) 
ar: Synaptic activity's rise time
ad: Synaptic activity's decay time
B: Width of the sigmoid (mv^-1)
rate: Rate for continuous stimuli transition
offset: Offset for continuous stimuli transition
init_fdb: Timepoint in seconds in which feedback initiates
t_delay: Time delay in seconds for the feedback 
"""

import os
import numpy as np

from dynworm import sys_paths as paths

##################################
### PARAMETERS / CONFIGURATION ###
##################################

os.chdir(paths.connectome_data_dir)

""" Gap junctions (Chemical, 279*279) """
Gg_Static = np.load('Gg_v3.npy') 

""" Synaptic connections (Chemical, 279*279) """
Gs_Static = np.load('Gs_v3.npy') 

""" Directionality (279*1) """
EMat_mask = np.load('emask_mat_v1.npy')

os.chdir(paths.default_dir)

default = {

    "N" : 279, 
    "Gc" : 0.1,
    "C" : 0.015,
    "ggap" : 1.0,
    "gsyn" : 1.0,
    "Ec" : -35.0,
    "E_rev": -48.0, 
    "ar" : 1.0/1.5,
    "ad" : 5.0/1.5,
    "B" : 0.125,
    "rate" : 0.025,
    "offset" : 0.15,
    "iext" : 100000.,
    "init_key_counts" : 13

    }

pA_unit = {

    "N" : 279, 
    "Gc" : 0.01,
    "C" : 0.0015,
    "ggap" : 0.1,
    "gsyn" : 0.1,
    "Ec" : -35.0,
    "E_rev": -48.0, 
    "ar" : 1.0/1.5,
    "ad" : 5.0/1.5,
    "B" : 0.125,
    "rate" : 0.025,
    "offset" : 0.15,
    "iext" : 100000.,
    "init_key_counts" : 13

    }