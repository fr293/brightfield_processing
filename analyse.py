import numpy as np
import time


def read_data(filepath):
    data = np.genfromtxt(filepath, dtype=float, delimiter=',')
    exp_time, position_x, position_y, position_z, displacement, x_mean_force, y_mean_force, z_mean_force, \
        x_std_dev_force, y_std_dev_force, z_std_dev_force, force_on = data.T

    return exp_time, position_x, position_y, position_z, displacement, x_mean_force, y_mean_force, z_mean_force, \
        x_std_dev_force, y_std_dev_force, z_std_dev_force, force_on


# this could be improved by directly interpreting the force on and off values in the experimental file
# find the nearest time value before the force on time, and the nearest time value after the time off (on + duration)
# value
def force_switch_indices(force_on_vector):
    indices = np.argwhere(force_on_vector)
    start_index = indices[0] - 1
    end_index = indices[-1] + 1
    return start_index, end_index

# this function finds the last value that's smaller than the nominal on or off time
def index_finder(time_vector, critical_time):
    difference = time_vector - critical_time
    difference[difference > 0] = np.nan
    index = np.nanargmin(np.abs(difference))
    return index

def force_switch_indices_semantic(filename, time_vector):
    # all filenames end with a two character duration value
    # extract this value and use it to interpret the indices
    # hand-tuned start and end index values are listed in comments for each condition
    duration_value = filename[-2:]

    if duration_value == '10':
        on_time = 10
        off_time = 20
        # start_index = 19
        # end_index = 35

    elif duration_value == '30':
        on_time = 10
        off_time = 40
        # start_index = 19
        # end_index = 71

    elif duration_value == '90':
        on_time = 10
        off_time = 100
        # start_index = 19
        # end_index = 188

    else:
        print('error: filename does not end with duration code')
        raise UserWarning

    start_index = index_finder(time_vector, on_time)
    end_index = index_finder(time_vector, off_time)

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


# extract the peak displacement value and peak displacement time from a run, with robustness to noise and drift
def peak_displacement(start_index, end_index, displacement):
    index_range = np.round(0.17 * (end_index - start_index))
    y_data = displacement[int(end_index - index_range):int(end_index + index_range)]
    peak_val = y_data.max()
    return peak_val


def force_magnitude(x_mean_force, y_mean_force):
    plane_force = [x_mean_force, y_mean_force]
    peak_force_val = np.linalg.norm(plane_force, 2)
    return peak_force_val


def terminal_displacement(displacement):
    index_range = int(round(0.1*displacement.size))
    y_data = displacement[-index_range:-1]
    mean = np.mean(y_data)
    return mean


def full_analysis(filepath, filename):
    exp_time, position_x, position_y, position_z, displacement, x_mean_force, y_mean_force, z_mean_force, \
        x_std_dev_force, y_std_dev_force, z_std_dev_force, force_on = read_data(filepath)

    start_index, end_index = force_switch_indices_semantic(filename, exp_time)

    creep_time, creep_viscosity, peak_deformation, viscous_displacement = \
        terminal_viscosity(start_index, end_index, exp_time, displacement)

    residual_deformation = terminal_displacement(displacement)

    peak_force = force_magnitude(x_mean_force[end_index], y_mean_force[end_index])

    return viscous_displacement, creep_viscosity, creep_time, residual_deformation, peak_deformation, peak_force
