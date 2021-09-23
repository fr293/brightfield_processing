from julia import RHEOS as rh
import numpy as np
from sklearn.covariance import empirical_covariance
from matplotlib import pyplot as plt
from julia import Pkg

Pkg.activate("C:\\Users\\fr293\\code\\brightfield_processing\\rheos_env")

bead_radius = 20E-6
time_step = 1
strain_resolution = 1E-7 / bead_radius


def read_data(filepath):
    data = np.genfromtxt(filepath, dtype=float, delimiter=',', skip_header=1)
    exp_time, position_x, position_y, position_z, displacement, alignment_ratio, x_mean_force, y_mean_force, z_mean_force, eigenforce, force_on = data.T

    return exp_time, position_x, position_y, position_z, displacement, alignment_ratio, x_mean_force, y_mean_force, z_mean_force, eigenforce, force_on


def read_analysis(filepath):
    analysis = np.genfromtxt(filepath, usecols=(0, 5, 6, 7), delimiter=',', skip_header=1, dtype=None, encoding=None)
    analysis_list = analysis.tolist()
    return analysis_list


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


def rheos_import(filepath, filename):
    file_location = filepath + filename + '_rheos.csv'
    data = rh.importcsv(file_location, rh.Tweezers(bead_radius), t_col=1, d_col=2, f_col=3, header=True)
    data_res = rh.resample(data, dt=time_step)
    data_smooth = rh.smooth(data_res, 2.5 * time_step)
    return data_smooth


def weight_function(data_smooth, time_period):
    # takes the smoothed data array and returns a weighting array of equal length
    tail_proportion = 0.1
    weight_slope = 1

    measured_time = rh.gettime(data_smooth)
    time_size = measured_time.size
    # calculate the base proportion excluding the initial 10 second experiment startup period
    startup_length = 10.0 / time_period
    base_proportion = np.int(
        np.ceil(((time_size - startup_length) * (1 - tail_proportion)) + startup_length))
    addendum_proportion = time_size - base_proportion
    # julia uses 1 indexing
    weights_base = np.arange(base_proportion) + 1
    addendum_multiplier = np.round(
        (weight_slope * np.arange(addendum_proportion)) + 1.0)
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
    # extract and sum over stress data to get maximum detectable viscosity
    data_smooth = rheos_import(filename, filepath)
    time_weights = weight_function(data_smooth, time_step)
    stress_history = np.sum(rh.getstress(data_smooth)) * time_step
    visco_ceiling = stress_history / strain_resolution

    primitive_maxwell = rh.modelfit(data_smooth, rh.Maxwell, rh.stress_imposed, p0={'eta': 10.0, 'k': 1.0},
                                    lo={'k': 0.0, 'eta': 0.0}, hi={'k': 10, 'eta': visco_ceiling},
                                    optmethod='LN_SBPLX', opttimeout=30)
    primitive_springpot = rh.modelfit(data_smooth, rh.Springpot, rh.stress_imposed, p0={'beta': 0.05, 'c_beta': 0.05},
                                      lo={'beta': 0.001, 'c_beta': 0.0}, hi={'beta': 0.999, 'c_beta': 1000},
                                      optmethod='LN_SBPLX', opttimeout=30)
    dashpot_start = rh.dict(rh.getparams(primitive_maxwell, unicode=False))
    springpot_start = rh.dict(rh.getparams(primitive_springpot, unicode=False))
    if dashpot_start['eta'] < 0.99 * visco_ceiling:
        p0_eta = dashpot_start['eta']
    else:
        p0_eta = 0.99 * visco_ceiling

    if not 0.01 < springpot_start['beta'] < 0.99:
        p0_beta = 0.05
    else:
        p0_beta = springpot_start['beta']

    model = rh.modelfit(data_smooth, rh.FractD_Maxwell, rh.stress_imposed,
                        p0={'beta': p0_beta,
                            'c_beta': springpot_start['c_beta'], 'eta': p0_eta},
                        lo={'beta': 0.001, 'c_beta': 0.0, 'eta': 0.0},
                        hi={'beta': 0.999, 'c_beta': 1000, 'eta': visco_ceiling},
                        weights=time_weights, optmethod='LN_SBPLX', opttimeout=30)
    parameters = rh.dict(rh.getparams(model, unicode=False))

    if parameters['eta'] < (0.99 * visco_ceiling):
        eta = parameters['eta']
        c_beta = parameters['c_beta']
        beta = parameters['beta']
        model_plasticity = (stress_history / eta) * bead_radius

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

    strain_residual = fit_strain - measured_strain
    strain_residual = strain_residual / np.abs(strain_residual).max()

    fig, ax = plt.subplots()
    ax.plot(measured_time, measured_strain,
            linewidth=2, label='measured strain')
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


def prediction_learn(filepath, training_force=None, training_duration=None):
    # set default values
    if training_force is None:
        training_force = ['0A5']
    if training_duration is None:
        training_duration = ['30', '90']
    # read in analysis output from CSV, transfer only name, [parameters]
    analysis_list = read_analysis(filepath + 'experiment_analysis.csv')
    # search for correct training force, duration
    parameters = np.empty((0, 3))
    for entry in analysis_list:
        label, eta, cbeta, beta = entry
        if any(force in label for force in training_force) and any(duration in label for duration in training_duration):
            # include only entries with non-zero viscosity
            if eta:
                parameters = np.vstack([parameters, [eta, cbeta, beta]])

    # check final number of entries: if <2, raise an error and exit
    dims = parameters.shape
    length = dims[0]
    if length < 2:
        print('error: insufficient training data for statistical analysis (n<2)')
        raise UserWarning

    # estimate mean
    meanval = np.mean(parameters, axis=0)
    # estimate variance
    cov_val = empirical_covariance(parameters)
    # Returns: 3D Gaussian mean and covariance parameters
    return meanval, cov_val


def prediction_run(mean, cov, stress_history, n=1000):
    # generate n parameter samples from the distribution
    params = np.random.multivariate_normal(mean, cov, n)
    paramslist = params.tolist()
    # for each sample, generate strain data with RHEOS
    time_array = rh.gettime(stress_history)
    exp_length = time_array.size
    strain_array = np.empty((0, exp_length))
    for sample in paramslist:
        paramdict = {'beta': sample[2], 'c_beta': sample[1], 'eta': sample[0]}
        model = rh.RheoModel(rh.FractD_Maxwell, rh.symbol_to_unicode(paramdict))
        data_predict = rh.modelpredict(stress_history, model)
        predict_strain = rh.getstrain(data_predict)
        strain_array = np.vstack([strain_array, predict_strain])

    # calculate mean of generated strain data
    mean_strain = np.mean(strain_array, axis=0)
    # calculate upper bound
    upper_strain = np.percentile(strain_array, 90, axis=0)
    # calculate lower bound
    lower_strain = np.percentile(strain_array, 10, axis=0)
    # return mean, upper, lower
    return mean_strain, upper_strain, lower_strain


def full_analysis(filename, filepath):
    exp_time, position_x, position_y, position_z, displacement, alignment_ratio, x_mean_force, y_mean_force, \
    z_mean_force, eigenforce, force_on = read_data(filepath + filename + '_full.csv')

    start_index, end_index = force_switch_indices(force_on)

    peak_deformation = peak_displacement(start_index, end_index, displacement)

    residual_deformation = terminal_displacement(displacement)

    peak_force = force_magnitude(
        x_mean_force[end_index], y_mean_force[end_index])

    eta, c_beta, beta, model_plasticity = rheos_fract_maxwell(filename, filepath)

    return peak_deformation, peak_force, residual_deformation, model_plasticity, eta, c_beta, beta
