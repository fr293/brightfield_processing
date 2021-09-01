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


def rheos_fract_maxwell(filename, filepath):

    file_location = filepath + filename + '_rheos.csv'

    data = rh.importcsv(file_location, rh.Tweezers(20E-6), t_col=1, d_col=2, f_col=3, header=True)
    data_res = rh.resample(data, dt=1)
    data_smooth = rh.smooth(data_res, 2.5)
    primitive_dashpot = rh.modelfit(data_smooth, rh.Dashpot, rh.stress_imposed, p0={'eta': 10.0}, lo={'eta': 0.0})
    primitive_springpot = rh.modelfit(data_smooth, rh.Springpot, rh.stress_imposed, p0={'beta': 0.05, 'c_beta': 0.05},
                                      lo={'beta': 0.001, 'c_beta': 0.0}, hi={'beta': 0.999, 'c_beta': np.inf})
    dashpot_start = rh.dict(rh.getparams(primitive_dashpot, unicode=False))
    springpot_start = rh.dict(rh.getparams(primitive_springpot, unicode=False))
    model = rh.modelfit(data_smooth, rh.FractD_Maxwell, rh.stress_imposed,
                        p0={'beta': springpot_start['beta'], 'c_beta': springpot_start['c_beta'],
                            'eta': dashpot_start['eta']}, lo={'beta': 0.001, 'c_beta': 0.0, 'eta': 0.0},
                        hi={'beta': 0.999, 'c_beta': np.inf, 'eta': np.inf})

    parameters = rh.dict(rh.getparams(model, unicode=False))
    eta = parameters['eta']
    c_beta = parameters['c_beta']
    beta = parameters['beta']

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
    fig.savefig(filepath + filename + '_rheos.svg')
    fig.savefig(filepath + filename + '_rheos.jpeg')

    return eta, c_beta, beta


def full_analysis(filename, filepath):
    exp_time, position_x, position_y, position_z, displacement, alignment_ratio, x_mean_force, y_mean_force, \
            z_mean_force, eigenforce, force_on = read_data(filepath + filename + '_full.csv')

    start_index, end_index = force_switch_indices(force_on)

    peak_deformation = peak_displacement(start_index, end_index, displacement)

    residual_deformation = terminal_displacement(displacement)

    peak_force = force_magnitude(x_mean_force[end_index], y_mean_force[end_index])

    eta, c_beta, beta = rheos_fract_maxwell(filename, filepath)

    return peak_deformation, peak_force, residual_deformation, eta, c_beta, beta
