import numpy as np
import time


def read_data(filepath):
    data = np.genfromtxt(filepath, dtype=float, delimiter=',')
    exp_time, position_x, position_y, position_z, displacement, x_mean_force, y_mean_force, z_mean_force, \
        x_std_dev_force, y_std_dev_force, z_std_dev_force, force_on = data.T

    return exp_time, position_x, position_y, position_z, displacement, x_mean_force, y_mean_force, z_mean_force, \
        x_std_dev_force, y_std_dev_force, z_std_dev_force, force_on


def force_switch_indices(force_on_vector):
    indices = np.argwhere(force_on_vector)
    start_index = indices[0] - 1
    end_index = indices[-1] + 1
    return start_index, end_index


def terminal_viscosity(start_index, end_index, exp_time, displacement):
    x_data = exp_time[int(start_index):int(end_index)]
    y_data = displacement[int(start_index):int(end_index)]
    index_range = np.round(0.33 * (end_index - start_index))
    x_data_fluid = exp_time[int(end_index-index_range):int(end_index)]
    y_data_fluid = displacement[int(end_index-index_range):int(end_index)]
    p = np.polyfit(x_data_fluid, y_data_fluid, 1)
    lin_flow = np.polyval(p, x_data_fluid)
    error = np.absolute(y_data_fluid - lin_flow)
    peak = peak_displacement(start_index, end_index, displacement)
    scaled_error = (error/peak)*100
    p_error = np.polyfit(range(scaled_error.size), scaled_error, 1)
    error_index = (10-p_error[1])/p_error[0]
    p_time = np.polyfit(range(x_data_fluid.size), x_data_fluid, 1)
    transition_time = np.polyval(p_time, error_index)
    t_creep = transition_time - x_data[0]
    d_fluid = p[0] * (x_data[-1] - x_data[0])
    return t_creep, p[0], peak, d_fluid


def peak_displacement(start_index, end_index, displacement):
    index_range = np.round(0.17 * (end_index - start_index))
    y_data = displacement[int(end_index - index_range):int(end_index + index_range)]
    peak_val = y_data.max()
    return peak_val


def terminal_displacement(displacement):
    index_range = int(round(0.1*displacement.size))
    y_data = displacement[-index_range:-1]
    mean = np.mean(y_data)
    return mean


def full_analysis(filepath):
    exp_time, position_x, position_y, position_z, displacement, x_mean_force, y_mean_force, z_mean_force, \
        x_std_dev_force, y_std_dev_force, z_std_dev_force, force_on = read_data(filepath)

    start_index, end_index = force_switch_indices(force_on)

    creep_time, creep_viscosity, peak_deformation, viscous_displacement = \
        terminal_viscosity(start_index, end_index, exp_time, displacement)

    residual_deformation = terminal_displacement(displacement)

    return viscous_displacement, creep_viscosity, creep_time, residual_deformation, peak_deformation