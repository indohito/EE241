
# coding: utf-8

###########################
#NETWORK SIMULATION MODULE#
###########################

import time
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import integrate, sparse, linalg, interpolate
from IPython.display import Video

from dynworm import sys_paths as paths
from dynworm import neural_params as n_params
from dynworm import neurons_idx as n_idx

np.random.seed(10)

###########################################################
### MASTER EXECUTION FUNCTIONS (BASIC) ####################
###########################################################

def run_network_constinput(t_duration, input_vec, ablation_mask, \
    t_delta = 0.01, custom_initcond = False, ablation_type = "all"):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    tf = t_duration
    dt = t_delta # recommend 0.01

    params_obj_neural['simulation_type'] = 'constant_input'

    nsteps = int(np.floor(tf/dt) + 1)
    params_obj_neural['inmask'] = input_vec
    progress_milestones = np.linspace(0, nsteps, 10).astype('int')

    """ define the connectivity """

    modify_Connectome(ablation_mask, ablation_type)

    """ Calculate V_threshold """

    params_obj_neural['vth'] = EffVth_rhs(params_obj_neural['inmask'])

    """ Set initial condition """

    if custom_initcond == False:

        initcond = 10**(-4)*np.random.normal(0, 0.94, 2*params_obj_neural['N'])

    else:

        initcond = custom_initcond
        print("using the custom initial condition")

    """ Configuring the ODE Solver """
    r = integrate.ode(membrane_voltageRHS_constinput, compute_jacobian_constinput).set_integrator('vode', atol = 1e-3, min_step = dt*1e-6, method = 'bdf')
    r.set_initial_value(initcond, 0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))

    t[0] = 0
    traj[0, :] = initcond[:params_obj_neural['N']]
    vthmat = np.tile(params_obj_neural['vth'], (nsteps, 1))

    print("Network integration prep completed...")

    """ Integrate the ODE(s) across each delta_t timestep """
    print("Computing network dynamics...")
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:params_obj_neural['N']]

        k += 1

        if k in progress_milestones:

            print(str(np.round((float(k) / nsteps) * 100, 1)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "dt": dt,
            "steps": nsteps,
            "raw_v_solution": traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

#######################################################################################################
### Vth COMPUTATION, JACOBIAN, ABLATION, VOLTAGE FITERING, INTERPOLATION FUNCTIONS ####################
#######################################################################################################

def initialize_params_neural(custom_params = False):

    global params_obj_neural

    if custom_params == False:

        params_obj_neural = n_params.default
        #print('Using the default neural parameters')

    else:

        assert type(custom_params) == dict, "Custom neural parameters should be of dictionary format"

        if validate_custom_neural_params(custom_params) == True:

            params_obj_neural = custom_params
            #print('Accepted the custom neural parameters')

def validate_custom_neural_params(custom_params):

    key_checker = []

    for key in n_params.default.keys():
        
        key_checker.append(key in custom_params)

    all_keys_present = np.sum(key_checker) == n_params.default['init_key_counts']
    
    assert np.sum(key_checker) == n_params.default['init_key_counts'], "Provided dictionary is incomplete"

    return all_keys_present

def initialize_connectivity(custom_connectivity_dict = False):

    assert 'params_obj_neural' in globals(), "Neural parameters must be initialized before initializing the connectivity"

    if custom_connectivity_dict == False:

        params_obj_neural['Gg_Static'] = n_params.Gg_Static
        params_obj_neural['Gs_Static'] = n_params.Gs_Static
        EMat_mask = n_params.EMat_mask
        #print('Using the default connectivity')

    else:

        assert type(custom_connectivity_dict) == dict, "Custom connectivity should be of dictionary format"

        params_obj_neural['Gg_Static'] = custom_connectivity_dict['gap']
        params_obj_neural['Gs_Static'] = custom_connectivity_dict['syn']
        EMat_mask = custom_connectivity_dict['directionality']

    params_obj_neural['EMat'] = params_obj_neural['E_rev'] * EMat_mask
    params_obj_neural['mask_Healthy'] = np.ones(params_obj_neural['N'], dtype = 'bool')

def EffVth(Gg, Gs):

    Gcmat = np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    EcVec = np.multiply(params_obj_neural['Ec'], np.ones((params_obj_neural['N'], 1)))

    M1 = -Gcmat
    b1 = np.multiply(params_obj_neural['Gc'], EcVec)

    Ggap = np.multiply(params_obj_neural['ggap'], Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(params_obj_neural['gsyn'], Gs)
    s_eq = round((params_obj_neural['ar']/(params_obj_neural['ar'] + 2 * params_obj_neural['ad'])), 4)
    sjmat = np.multiply(s_eq, np.ones((params_obj_neural['N'], params_obj_neural['N'])))
    S_eq = np.multiply(s_eq, np.ones((params_obj_neural['N'], 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()

    #b3 = np.dot(Gs_ij, np.multiply(s_eq, params_obj_neural['E']))
    b3 = np.dot(np.multiply(Gs_ij, params_obj_neural['EMat']), s_eq * np.ones((params_obj_neural['N'], 1)))

    M = M1 + M2 + M3

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, params_obj_neural['N'])

    params_obj_neural['LL'] = LL
    params_obj_neural['UU'] = UU
    params_obj_neural['bb'] = bb

def EffVth_rhs(inmask):

    InputMask = np.multiply(params_obj_neural['iext'], inmask)
    b = np.subtract(params_obj_neural['bb'], InputMask)

    vth = linalg.solve_triangular(params_obj_neural['UU'], linalg.solve_triangular(params_obj_neural['LL'], b, lower = True, check_finite=False), check_finite=False)

    return vth

def modify_Connectome(ablation_mask, ablation_type):

    # ablation_type can be 'all': ablate both synaptic and gap junctions, 'syn': Synaptic only and 'gap': Gap junctions only

    if np.sum(ablation_mask) == params_obj_neural['N']:

        apply_Mat = np.ones((params_obj_neural['N'], params_obj_neural['N']))

        params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
        params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

        print("All neurons are healthy")

        EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])

    else:

        apply_Col = np.tile(ablation_mask, (params_obj_neural['N'], 1))
        apply_Row = np.transpose(apply_Col)

        apply_Mat = np.multiply(apply_Col, apply_Row)

        if ablation_type == "all":

            params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
            params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

            print("Ablating both Gap and Syn")

        elif ablation_type == "syn":

            params_obj_neural['Gg_Dynamic'] = params_obj_neural['Gg_Static'].copy()
            params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

            print("Ablating only Syn")

        elif ablation_type == "gap":

            params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
            params_obj_neural['Gs_Dynamic'] = params_obj_neural['Gs_Static'].copy()

            print("Ablating only Gap")

        EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])


def voltage_filter(v_vec, vmax, scaler):
    
    filtered = vmax * np.tanh(scaler * np.divide(v_vec, vmax))
    
    return filtered

########################################################################################################
### RIGHT HAND SIDE FUNCTIONS + JACOBIAN ###############################################################
########################################################################################################

def membrane_voltageRHS_constinput(t, y):

    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth'])))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def compute_jacobian_constinput(t, y):

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth']))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

def test_brain_repair(repaired_connectome):

    equivalence_check = repaired_connectome.T == n_params.Gs_Static

    if equivalence_check.sum() == 279*279:

        print('Repair operation successful! - Simulating the nervous system and body for gentle tale touch')

        modified_connectomes = {

        "gap": n_params.Gg_Static,
        "syn": repaired_connectome.T,
        "directionality": n_params.EMat_mask

        }

        initialize_params_neural()
        initialize_connectivity(modified_connectomes)

        input_vec = np.zeros(params_obj_neural['N'])
        ablation_mask = np.ones(params_obj_neural['N'], dtype = 'bool')

        input_vec[276] = 0.35
        input_vec[278] = 0.35

        result_dict = run_network_constinput(t_duration=15, input_vec = input_vec, ablation_mask = ablation_mask)

        v_sol = result_dict['v_solution'].T

        fig = plt.figure(figsize=(10, 6))
        plt.subplot(2,1,1)
        plt.pcolor(v_sol[n_idx.VD_ind, 100:600], cmap='bwr')
        plt.xlabel('Time (unit = 10ms)', fontsize = 15)
        plt.ylabel('Neurons', fontsize = 15)
        plt.ylim(len(n_idx.VD_ind), 0)
        plt.title('Dorsal Motorneurons Voltage Activity (mV)', fontsize = 20)
        plt.colorbar()

        fig = plt.figure(figsize=(10, 6))
        plt.subplot(2,1,2)
        plt.pcolor(v_sol[n_idx.AS_ind, 100:600], cmap='bwr')
        plt.xlabel('Time (unit = 10ms)', fontsize = 15)
        plt.ylabel('Neurons', fontsize = 15)
        plt.ylim(len(n_idx.AS_ind), 0)
        plt.title('Ventral Motorneurons Voltage Activity (mV)', fontsize = 20)
        plt.colorbar()
        return Video("escaped_response_fixed.mp4", embed = True, height = 500, width = 500)

    else:

        print('Repair operation unsuccessful! - Please check your rewire_neurons() function')





