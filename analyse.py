import numpy as np
from matplotlib import pyplot as plt
from julia import Pkg
Pkg.activate("C:\\Users\\fr293\\code\\brightfield_processing\\rheos_env")
from julia import RHEOS as rh


def read_data(filepath):
    data = np.genfromtxt(filepath, dtype=float, delimiter=',')
    exp_time, position_x, position_y, position_z, displacement, alignment_ratio, x_mean_force, y_mean_force, \
        z_mean_force, eigenforce, force_on = data.T

    return exp_time, position_x, position_y, position_z, displacement, alignment_ratio, x_mean_force, y_mean_force, \
        z_mean_force, eigenforce, force_on


# this could be improved by directly interpreting the force on and off values in the experimental file
# find the nearest time value before the force on time, and the nearest time value after the time off (on + duration)
# value
def force_switch_indices(force_on_vector):
    indices = np.argwhere(force_on_vector)
    start_index = indices[0] - 1
    end_index = indices[-1] + 1
    return start_index, end_index


# extract the peak displacement value and peak displacement time from a run, with robustness to noise and drift
def peak_displacement(start_index, end_index, displacement):
    index_range = np.round(0.2 * (end_index - start_index))
    y_data = displacement[int(end_index - index_range):int(end_index + index_range)]
    peak_val = y_data.max()
    return peak_val


def force_magnitude(x_mean_force, y_mean_force):
    plane_force = [x_mean_force, y_mean_force]
    peak_force_val = np.linalg.norm(plane_force, 2)
    return peak_force_val


def terminal_displacement(displacement):
    index_range = int(round(0.1 * displacement.size))
    y_data = displacement[-index_range:-1]
    mean = np.mean(y_data)
    return mean


def weight_function(data_smooth, time_period):
    # takes the smoothed data array and returns a weighting array of equal length
    tail_proportion = 0.1
    weight_slope = 1

    measured_time = rh.gettime(data_smooth)
    time_size = measured_time.size
    # calculate the base proportion excluding the initial 10 second experiment startup period
    startup_length = 10.0 / time_period
    base_proportion = np.int(np.ceil(((time_size - startup_length)*(1-tail_proportion)) + startup_length))
    addendum_proportion = time_size - base_proportion
    # julia uses 1 indexing
    weights_base = np.arange(base_proportion) + 1
    addendum_multiplier = np.round((weight_slope * np.arange(addendum_proportion)) + 1.0)
    addendum_multiplier = addendum_multiplier.astype(int)
    index = base_proportion + 1
    for i in addendum_multiplier:
        weights_addendum = np.ones(i) * index
        weights_base = np.concatenate((weights_base, weights_addendum))
        index = index + 1
    weights_int = weights_base.astype(int)
    weights_list = weights_int.tolist()

    return weights_list


def rheos_fract_maxwell(filename, filepath):

    bead_radius = 20E-6
    time_step = 1
    strain_resolution = 1E-7/bead_radius

    file_location = filepath + filename + '_rheos.csv'

    data = rh.importcsv(file_location, rh.Tweezers(bead_radius), t_col=1, d_col=2, f_col=3, header=True)
    data_res = rh.resample(data, dt=time_step)
    data_smooth = rh.smooth(data_res, 2.5*time_step)
    time_weights = weight_function(data_smooth, time_step)
    # extract and sum over stress data to get maximum detectable viscosity
    stress_history = np.sum(rh.getstress(data_smooth))*time_step
    visco_ceiling = stress_history/strain_resolution

    primitive_maxwell = rh.modelfit(data_smooth, rh.Maxwell, rh.stress_imposed, p0={'eta': 10.0, 'k': 1.0},
                                    lo={'k': 0.0, 'eta': 0.0}, hi={'k': 10, 'eta': visco_ceiling},
                                    optmethod='LN_COBYLA', opttimeout=30)
    primitive_springpot = rh.modelfit(data_smooth, rh.Springpot, rh.stress_imposed, p0={'beta': 0.05, 'c_beta': 0.05},
                                      lo={'beta': 0.001, 'c_beta': 0.0}, hi={'beta': 0.999, 'c_beta': 1000},
                                      optmethod='LN_COBYLA', opttimeout=30)
    dashpot_start = rh.dict(rh.getparams(primitive_maxwell, unicode=False))
    springpot_start = rh.dict(rh.getparams(primitive_springpot, unicode=False))
    if dashpot_start['eta'] < 0.99*visco_ceiling:
        p0_eta = dashpot_start['eta']
    else:
        p0_eta = 0.99*visco_ceiling

    model = rh.modelfit(data_smooth, rh.FractD_Maxwell, rh.stress_imposed,
                        p0={'beta': springpot_start['beta'], 'c_beta': springpot_start['c_beta'],
                            'eta': p0_eta}, lo={'beta': 0.001, 'c_beta': 0.0, 'eta': 0.0},
                        hi={'beta': 0.999, 'c_beta': 1000, 'eta': visco_ceiling}, weights=time_weights,
                        optmethod='LN_COBYLA', opttimeout=30)
    parameters = rh.dict(rh.getparams(model, unicode=False))

    if parameters['eta'] < (0.99*visco_ceiling):
        eta = parameters['eta']
        c_beta = parameters['c_beta']
        beta = parameters['beta']
        model_plasticity = (stress_history/eta)*bead_radius

    else:
        model = rh.modelfit(data_smooth, rh.Springpot, rh.stress_imposed, p0={'beta': springpot_start['beta'],
                                                                              'c_beta': springpot_start['c_beta']},
                            lo={'beta': 0.001, 'c_beta': 0.0}, hi={'beta': 0.999, 'c_beta': np.inf})
        parameters = rh.dict(rh.getparams(model, unicode=False))
        eta = 0
        c_beta = parameters['c_beta']
        beta = parameters['beta']
        model_plasticity = 0

    data_stress = rh.extract(data_smooth, rh.stress_only)
    data_fit = rh.modelpredict(data_stress, model)

    measured_time = rh.gettime(data_smooth)
    measured_strain = rh.getstrain(data_smooth)
    fit_strain = rh.getstrain(data_fit)

    strain_residual = fit_strain-measured_strain
    strain_residual = strain_residual / np.abs(strain_residual).max()

    fig, ax = plt.subplots()
    ax.plot(measured_time, measured_strain, linewidth=2, label='measured strain')
    ax.plot(measured_time, fit_strain, linewidth=2, label='fit strain')
    ax.set_xlabel('Time/s')
    ax.set_ylabel('Characteristic Strain')
    ax.legend(loc='upper right', fontsize='medium', framealpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(measured_time, strain_residual, color='black', ls=':')
    ax2.set_ylim([-5, 5])
    ax2.yaxis.set_ticks([])
    ax2.set_ylabel('Normalised Model Error/AU')
    fig.suptitle(filename)
    fig.savefig(filepath + filename + '_rheos.jpeg')
    plt.close(fig)

    return eta, c_beta, beta, model_plasticity


def full_analysis(filename, filepath):
    exp_time, position_x, position_y, position_z, displacement, alignment_ratio, x_mean_force, y_mean_force, \
            z_mean_force, eigenforce, force_on = read_data(filepath + filename + '_full.csv')

    start_index, end_index = force_switch_indices(force_on)

    peak_deformation = peak_displacement(start_index, end_index, displacement)

    residual_deformation = terminal_displacement(displacement)

    peak_force = force_magnitude(x_mean_force[end_index], y_mean_force[end_index])

    eta, c_beta, beta, model_plasticity = rheos_fract_maxwell(filename, filepath)

    return peak_deformation, peak_force, residual_deformation, model_plasticity, eta, c_beta, beta
