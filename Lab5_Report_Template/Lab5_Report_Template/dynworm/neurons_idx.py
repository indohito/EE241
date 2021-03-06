# coding: utf-8

#################################################
#PRE-SELECTED NEURON INDICES FOR QUALITY OF LIFE#
#################################################

import numpy as np

from dynworm import neural_params as n_params

####################################
### B-group, A-group and D-group ###
####################################

VB_ind = np.asarray([150, 138, 170, 179, 186, 193, 202, 212, 217, 229, 234])
DB_ind = np.asarray([164, 152, 172, 188, 203, 218, 235])
VA_ind = np.asarray([160, 169, 177, 185, 191, 201, 211, 216, 228, 233, 239, 244])
DA_ind = np.asarray([167, 173, 184, 194, 207, 224, 237, 247, 248])
VD_ind = np.asarray([166, 168, 174, 183, 190, 200, 205, 215, 221, 232, 238, 241, 250])
DD_ind = np.asarray([163, 181, 195, 214, 231, 245])
AS_ind = np.asarray([165, 171, 182, 189, 199, 204, 213, 219, 230, 236, 240])

########################
### Hub Interneurons ###
########################

AVB_ind = np.asarray([96, 105])
PVC_ind = np.asarray([261, 267])
AVA_ind = np.asarray([47, 55])
AVD_ind = np.asarray([116, 118])
AVE_ind = np.asarray([58, 66])

##########################################
### Combinations of motor/interneurons ###
##########################################

B_grp = np.union1d(DB_ind, VB_ind)
D_grp = np.union1d(VD_ind, DD_ind)
A_grp = np.union1d(VA_ind, DA_ind)
AVB_B_grp = np.union1d(B_grp, AVB_ind)
AVB_PVC_B_grp = np.union1d(AVB_B_grp, PVC_ind)
FWD_group = np.union1d(AVB_PVC_B_grp, D_grp)

AVA_AVD_grp = np.union1d(AVA_ind, AVD_ind)
AVA_AVD_AVE_grp = np.union1d(AVA_AVD_grp, AVE_ind)
AD_grp = np.union1d(A_grp, D_grp)
BWD_group = np.union1d(AVA_AVD_AVE_grp, AD_grp)

BD_grp = np.union1d(B_grp, D_grp)
BDA_grp = np.union1d(BD_grp, A_grp)

#######################################
### Boolean masks for motor neurons ###
#######################################

B_grp_bool = np.zeros(n_params.default['N'], dtype = 'bool')
B_grp_bool[B_grp] = True

AVB_B_grp_bool = np.zeros(n_params.default['N'], dtype = 'bool')
AVB_B_grp_bool[AVB_B_grp] = True

BD_grp_bool = np.zeros(n_params.default['N'], dtype = 'bool')
BD_grp_bool[BD_grp] = True

BDA_grp_bool = np.zeros(n_params.default['N'], dtype = 'bool')
BDA_grp_bool[BDA_grp] = True