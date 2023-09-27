# from defines import *
import time

import numpy as np
import tkinter as tk
import copy
from predefined_tests import *
from output import plot_average_speed_of_ions, plot_potential_profile, plot_multiple_ions_hist
from scipy import interpolate
import multiprocessing as mp
import csv
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from time import sleep
import random
from random import random
from mpl_toolkits.mplot3d import Axes3D
class Rims:

    # region setup (init)
    def __init__(self, ion_dict, potential_profile,l_runnig_time, manual_ss_criteria=0 ):
        """
        Instance holding attributes of the whole simulation
        :param ion_dict: ion dictionary to be simulated, key is name and value is diffusion parameter D
        :param potential_profile: list holding potential profile shape [L,a1,a2]
        :param manual_ss_criteria: a typical velocity of the simulation, to use as reference velocity to check for ss
        """

        '''ratchet attributes'''
        self.ion_dict = ion_dict.copy()
        self.start_time = datetime.now()
        self.end_time = datetime.now()  # will update at the end of the run
        self.L = potential_profile[0]  # length of the ratchet, in mm
        self.x_space_vec = potential_profile[1]  # preparation to set uneven step in the profile, currently only plot x
        self.time_vec = potential_profile[2]  # every cell is the time of the corresponding profile to start
        #  len(time_vec) = number of different profiles (example simple saw = 2)
        self.potential_profile_mat = potential_profile[3]  # the potential_profile, every line is one prof,
        # length of line can determine resolution (must be uniform)
        self.flash_period = self.time_vec[-1]  # full cycle of the ratchet - T
        self.flash_frequency = 1 / self.flash_period  # frequency cycle of the ratchet - f
        #self.interval = dt
        self.interval = self.get_intervals()  # interval of one step - delta t
        self.intervals_in_cycle = int(self.flash_period / self.interval)

        self.max_cycles = get_setting('MAX_CYCLES')
        '''simulation attributes'''
        self.end_run = False  # the stop condition of the ratchet, update only in check for SS func
        self.hit_css = False
        self.steady_state_code = 0  # 0 - None, 1 - SS by 3% standard, 2 - SS by 3% STD standard,
        # 3 - ended by 100 runs, 4 - SS by 3% manual standard, update only in check for SS func
        self.css = -1  # the # of cycles the ratchet get
        self.resolution = get_setting('RESOLUTION')  # the min resolution of ions location and profile differentiation
        if self.potential_profile_mat.shape[1] >= self.resolution:
            self.resolution = self.potential_profile_mat.shape[1]

        # all the array here are vector representation of ions, every column is the same ion in every vector
        #### i can print those for finding what i need
        self.ion_d = np.array(list(ion_dict.values()) * get_setting("PARTICLES_SIMULATED"))  # ions d vector
        self.ion_count = len(self.ion_d)
        # self.ion_x = np.random.random(self.ion_count) * self.L  # start the run with random loc for every ion
        self.ion_x = self.get_starting_pos()
        self.ion_x_global = np.zeros([self.intervals_in_cycle, self.ion_count])
        # the global loc of ion in cycle, updated every step, reset every cycle
        self.ion_x_short_list = self.ion_x.copy()  # the loc of ions, updated every cycle
        self.ion_charge = np.ones(self.ion_count)  # for now once only, setup for more complex logic
        self.ion_area_count = np.zeros(self.ion_count)  # count the times that ion cross the ratchet length
        self.velocity = np.zeros(self.ion_count)
        self.step_time = self.interval / self.intervals_in_cycle
        self.gamma = self.get_gamma()
        self.velocity_means = np.array([])
        self.manual_ss_criteria = manual_ss_criteria
        self.random_movement_std = self.simulate_only_random_movement()
        #self.first_run = True

    def get_gamma(self):
        """
        get gama for each ion by his d
        """
        gamma = BOLTZMANN_CONSTANT * TEMPERATURE / self.ion_d
        return gamma

    def get_starting_pos(self):
        """
        get avg loc of the ion approximation distribution in the ratchet
        """
        # the avg of profile
        u_tot = np.zeros(len(self.potential_profile_mat[0]))
        for profile in range(len(self.potential_profile_mat)):
            if profile >= 1:
                u_tot += self.potential_profile_mat[profile] * \
                         (self.time_vec[profile] - self.time_vec[profile - 1]) / self.time_vec[-1]
            else:
                u_tot += self.potential_profile_mat[profile] * self.time_vec[profile] / self.time_vec[-1]
        c_t = np.exp(u_tot / (BOLTZMANN_CONSTANT * TEMPERATURE))
        norm = np.sum(c_t)
        distribution = np.round(self.ion_count * c_t / norm, 0).astype(int)
        while np.sum(distribution) != self.ion_count:
            rand = np.random.randint(0, len(distribution) - 1)
            if distribution[rand] >= 0:
                distribution[rand] += np.sign(self.ion_count - np.sum(distribution))
        pos = np.array([])
        block_size = self.L / len(distribution)
        for cell in range(len(distribution)):
            pos = np.append(pos, distribution[cell] * [block_size * cell])
        if len(pos) < self.ion_count:
            raise ValueError
        if len(pos) > self.ion_count:
            pos = pos[:self.ion_count]
        return pos

    def get_intervals(self, dt = 0):
        """
        Calculates the length of a time interval delta_t
        """
        '''Check for manual dt overwrite in settings'''
        if (dt != 0 ):
            return dt *pow (10, -6)
        else :
            if get_setting('OVERWRITE_DELTA_T'):
                dt = get_setting('DELTA_T') * pow(10, -6)

                print("this is maybe delta t_1: " + str(dt))

                return dt

        # '''Special dt for electrons'''
        # if self.ion == "Electrons in Silicon":
        #     return self.flash_period / INTERVALS_FLASH_RATIO_ELECTRONS

        '''keeps delta t smaller than tau (diffusion time for L)'''
        critical_t = ((1 / INTERVALS_FLASH_RATIO) * self.L) ** 2 / (2 * np.min(list(self.ion_dict.values())))
        '''Further diminishes delta t if its bigger that T/INTERVALS_FLASH_RATIO'''
        while critical_t * 10 > self.flash_period / INTERVALS_FLASH_RATIO:  # dor fix
            critical_t /= INTERVALS_FLASH_RATIO
        critical_t = (self.flash_period / INTERVALS_FLASH_RATIO) / 100
        print("this is maybe delta t: " + str(critical_t))
        return critical_t

    def simulate_only_random_movement(self):
        std_vec = []
        for n, d in self.ion_dict.items():
            x_diff = np.random.normal(0, 1, [self.intervals_in_cycle, 10000]) * np.sqrt(2 * d * self.interval)
            b = np.average(x_diff[1:, :], axis=0) / self.interval
            std_vec.append(np.std(b))
        return np.array(std_vec)

    # endregion

    # region run
    def run_rims(self, new_runnig_time):
        """
        the main func for simulation
        return the steady state code:
        1 - SS by 3% standard
        2 - SS by 3% STD standard
        3 - ended by selecterd number of runs
        4 - SS by 3% manual standard
        5 - SS by std of last 5 runs < std of random movement only * 1.05
        """
        headline_panel()
        cycle = 0
        while not self.end_run:
            cycle += 1
            for interval in range(self.intervals_in_cycle):
                profile_num = np.argmax(self.time_vec - interval * self.interval > 0)
                profile = self.potential_profile_mat[profile_num]
                electronic_field = get_electric_field_vec(profile, self.L, self.resolution)
                electric_movement = get_electric_velocity(self.ion_x, self.gamma, electronic_field,
                                                          self.L) * self.interval
                ksai = np.random.normal(0, 1, self.ion_count)
                noise_movement = ksai * np.sqrt(2 * self.ion_d * self.interval)
                ion_new_loc = self.ion_x + electric_movement + noise_movement
                ion_new_loc = self.calc_pos_arena(ion_new_loc, interval)  # update ion_x_global and ion_area_count
                self.ion_x = ion_new_loc
            self.update_short_list()  # update output data: ion_x_short_list, velocity
            self.check_for_steady_state(cycle, new_runnig_time)
            self.ion_x_global = np.zeros([self.intervals_in_cycle, self.ion_count])
        self.end_time = datetime.now()
        return self.steady_state_code

    def check_for_steady_state(self, run_count, new_runnig_time):
        """
        Checks whether the system has reached ss
        set the ss code:
        1 - SS by 3% standard
        2 - SS by 3% STD standard
        3 - ended by selecterd number of runs
        4 - SS by 3% manual standard
        5 - SS by std of last 5 runs < std of random movement only * 1.05
        dor - tested for one ion type only after BIG change, probably do some trouble
        """
        mean_velo_in_ss = []
        # Allow for ss only after MIN_MEASUREMENTS_FOR_SS cycles
        if run_count < MIN_MEASUREMENTS_FOR_SS - 1:
            return
        # calc avg velocity for every ion type separately, data for CYCLES_MEASUREMENTS_FOR_SS cycles
        avg_velocity_mat = self.get_average_by_ion_type(self.velocity[-CYCLES_MEASUREMENTS_FOR_SS:, :])
        mean_velocity = np.average(avg_velocity_mat, axis=0)
        mean_velo_in_ss.append(mean_velocity)
        std_velocity = self.get_std_by_ion_type(self.velocity[-CYCLES_MEASUREMENTS_FOR_SS:, :])
        # mean velocity of all the cycles CYCLES_MEASUREMENTS_FOR_SS
        self.velocity_means = np.append(self.velocity_means, mean_velocity)
        if np.abs(np.mean(np.abs(avg_velocity_mat[-1] - avg_velocity_mat[:-1])) / mean_velocity) < 0.03:
            # if the mean deviation from the mean_velocity is 3% of the mean_velocity
            self.steady_state_code = 1
        if np.abs(np.std(avg_velocity_mat) / mean_velocity) < 0.03:
            # if the std of last avg_velocity_mat is 3% of the mean_velocity
            self.steady_state_code = 2
        if self.manual_ss_criteria != 0:
            # if manual_ss_criteria is defined and not 0
            if np.abs(np.mean(np.abs(avg_velocity_mat[-1] - avg_velocity_mat[:-1])) / self.manual_ss_criteria) < 0.03:
                # if the mean deviation from the mean_velocity is 3% of the manual_ss_criteria
                self.steady_state_code = 4
        if np.all(std_velocity < self.random_movement_std * 1.02):
            # SS by std of last 5 runs < std of random movement only * 1.05
            self.steady_state_code = 5

        #t1 = 1*10**(-6) / (self.ion_dict["10^-5"]) #Velocity due to Electric field- v=E*mu


        if (run_count*self.flash_period) > 0.01:
            # if the simulation run 100 cycles
            # dor - 100 is total arbitrary number
            self.steady_state_code = 3

        if (self.steady_state_code != 0) & (self.hit_css == False):
            '''Saves the cycle when ss was first reached'''
            self.css = run_count
            self.hit_css = True
        if (self.css == run_count - 20) & (self.hit_css == True):
            self.end_run = True
        self.first_run = False
        return

    def get_average_by_ion_type(self, mat):
        """
        help func, calc avg of given mat for every ion type separately
        """
        avg_mat = []
        for ion in self.ion_dict.values():
            avg_mat.append(np.average(mat[:, self.ion_d == ion], axis=1))
        return np.array(avg_mat).T

    def get_std_by_ion_type(self, mat):
        """
        help func, calc avg of given mat for every ion type separately
        """
        avg_mat = []
        for ion in self.ion_dict.values():
            avg_mat.append(np.std(mat[:, self.ion_d == ion], axis=1))
        return np.array(avg_mat).T

    def calc_pos_arena(self, loc_array, interval):
        """
        if ion is out of bound returning him to the ratchet ("circular ratchet", "packman")
        update the ion_x_global and ion_area_count
        """
        self.ion_x_global[interval] = loc_array + self.L * self.ion_area_count
        mask_up = loc_array > self.L
        mask_down = loc_array < 0
        while np.any(mask_up + mask_down):
            loc_array = loc_array - mask_up * self.L
            self.ion_area_count += mask_up
            loc_array = loc_array + mask_down * self.L
            self.ion_area_count -= mask_down
            mask_up = loc_array > self.L
            mask_down = loc_array < 0
        return loc_array.copy()

    def update_short_list(self):
        """
        update the output data - avg loc of ion in each cycle and the avg velocity of ion at that cycle
        ion_x_short_list
        velocity
        """
        last_run_mat = self.ion_x_global
        new_pos = np.average(self.ion_x_global, axis=0)
        self.ion_x_short_list = np.vstack([self.ion_x_short_list, new_pos])
        self.velocity = np.vstack([self.velocity,
                                   np.average(last_run_mat[1:, :] - last_run_mat[:-1, :], axis=0)
                                   / self.interval])

    # endregion

    # region plot and save
    def get_x_to_plot(self):
        plot_dict = {}
        for key, val in self.ion_dict.items():
            if len(self.ion_x_short_list.shape) == 2:
                plot_dict[(key, val)] = self.ion_x_short_list[-1, self.ion_d == val]
            else:
                plot_dict[(key, val)] = self.ion_x_short_list[self.ion_d == val]
        return plot_dict

    def get_vec_to_plot(self):
        return self.get_average_by_ion_type(self.velocity), self.ion_dict

    def get_vec_and_err_to_plot(self):
        return self.get_average_by_ion_type(self.velocity), self.ion_dict, \
               self.get_std_by_ion_type(self.velocity) / np.sqrt(get_setting("PARTICLES_SIMULATED"))

    def save_data(self, sub_file):
        """
        save all simulation data at the output dir at given sub_file
        saves:
        * velocity csv
        * location csv
        * avg velocity over cycles graph
        * full simulation config txt
        * TBA - start loc graph
        * TBA - end loc graph
        """
        sub_dir = os.path.join(get_save_dir(), sub_file)
        os.makedirs(sub_dir)
        pd_vec = pd.DataFrame(self.velocity)
        pd_vec.to_csv(os.path.join(sub_dir, "vec.csv"))
        pd_loc = pd.DataFrame(self.ion_x_short_list)
        pd_loc.to_csv(os.path.join(sub_dir, "loc.csv"))
        '''histogram'''
        pd_loc_transpose = pd_loc.T
        for i in range(pd_loc_transpose.shape[1]):
            pd_loc_transpose[i].plot(kind='hist', density=False, bins=1000)
        path_new = os.path.join(get_save_dir(), "histogram")
        plt.savefig(path_new)
        v_plot_list, ions = self.get_vec_to_plot()
        # unique_id = create_unique_id()
        # plt.figure(unique_id)
        plt.figure()
        x_axis = np.arange(v_plot_list.shape[0])
        avg_window = 0
        for col in range(v_plot_list.shape[1]):
            temp = v_plot_list[:, col].copy()
            for cell in range(len(temp)):
                temp[cell] = np.mean(v_plot_list[max(0, cell - avg_window):cell + 1, col])
            plt.plot(x_axis, temp)
        plt.xlabel(r"Ratchet Cycle")
        plt.ylabel(r"Particle Velocity [cm/sec]")
        plt.suptitle("RIMS: Average speed of ions over ratchet cycles", fontsize=14, fontweight='bold')
        plt.legend = ions.keys()
        plt.grid()
        plt.savefig(os.path.join(sub_dir, "fig"))
        plt.close()
        run_data = ""
        run_data += f"ion_dict: {str(self.ion_dict)}\n"
        run_data += f"run_time: {str(self.end_time - self.start_time)}\n"
        run_data += f"L: {str(self.L)}\n"
        run_data += f"flash_period: {str(self.flash_period)}\n"
        run_data += f"flash_frequency: {str(self.flash_frequency)}\n"
        run_data += f"interval: {str(self.interval)}\n"
        run_data += f"intervals_in_period: {str(self.intervals_in_cycle)}\n"
        run_data += f"max_cycles: {str(self.max_cycles)}\n"
        run_data += f"steady_state_code: {str(self.steady_state_code)}\n"
        run_data += f"resolution: {str(self.resolution)}\n"
        run_data += f"ion_count: {str(self.ion_count)}\n"
        run_data += f"css: {str(self.css)}\n"
        run_data += f"manual_ss_criteria: {str(self.manual_ss_criteria)}\n"
        run_data += f"potential_profile_mat:\n {str(self.potential_profile_mat)}\n"
        run_data += f"time_vec:\n {str(self.time_vec)}\n"
        run_data += f"random_movement_std:\n {str(self.random_movement_std)}\n"
        text_file = open(os.path.join(sub_dir, f"{sub_file}.txt"), "w")
        text_file.write(run_data)
        return sub_dir

    def get_class_str_parameters(self):
        """
        help function for save and costume saves
        """
        return [
            [
                f"ion_dict", f"run_time", f"L", f"flash_period", f"flash_frequency", f"interval",
                f"intervals_in_period", f"max_cycles", f"steady_state_code", f"resolution", f"ion_count",
                f"css", f"manual_ss_criteria"
            ],
            [
                str(self.ion_dict), str(self.end_time - self.start_time), str(self.L),
                str(self.flash_period), str(self.flash_frequency), str(self.interval),
                str(self.intervals_in_cycle), str(self.max_cycles), str(self.steady_state_code),
                str(self.resolution), str(self.ion_count), str(self.css), str(self.manual_ss_criteria)
            ]
        ]
    # endregion


def make_spline_func(rachet_resolution, amplitude, alfa):
    micron = 1
    pot1_mat = [rachet_resolution]
    pot2_mat = [rachet_resolution]
    for x1 in np.arange(1 / rachet_resolution, 1, 1 / rachet_resolution):


        A1 = np.array([[x1 ** 2, x1, 1], [2 * x1, 1, 0], [0, 0, 1]])
        B1 = np.array([amplitude, 0, 0])
        solve1 = np.linalg.solve(A1, B1)

        A2 = np.array([[x1 ** 2, x1, 1], [2 * x1, 1, 0], [1 ** 2, 1, 1]])
        B2 = np.array([amplitude, 0, 0])
        solve2 = np.linalg.solve(A2, B2)

        plot_sol_1 = []
        plot_sol_1.append(0)
        for x_func in np.arange(micron / rachet_resolution, micron, micron / rachet_resolution):
            if x_func < x1:
                plot_sol_1.append(solve1[0] * ((x_func) ** 2) + solve1[1] * x_func + solve1[2])
            else:
                plot_sol_1.append(solve2[0] * ((x_func) ** 2) + solve2[1] * x_func + solve2[2])

        tck = interpolate.splrep(np.arange(0, micron, micron / rachet_resolution), plot_sol_1)
        x1_index = np.arange(1 / rachet_resolution, 1, 1 / rachet_resolution)  # is the spatial peak options
        y1_index = interpolate.splev(x1_index, tck, der=0)  # the potential at each point
        pot1_mat.append(y1_index)

    for x2 in np.arange(1 / rachet_resolution, 1, 1 / rachet_resolution):

        A3 = np.array([[x2 ** 2, x2, 1], [2 * x2, 1, 0], [0, 0, 1]])
        B3 = np.array([amplitude * alfa, 0, 0])
        solve3 = np.linalg.solve(A3, B3)

        A4 = np.array([[x2 ** 2, x2, 1], [2 * x2, 1, 0], [micron ** 2, micron, 1]])
        B4 = np.array([amplitude * alfa, 0, 0])
        solve4 = np.linalg.solve(A4, B4)
        plot_sol_2 = []
        plot_sol_2.append(0)
        for x_func in np.arange(micron / rachet_resolution, micron, micron / rachet_resolution):
            if x_func < x2:
                plot_sol_2.append(solve3[0] * ((x_func) ** 2) + solve3[1] * x_func + solve3[2])
            else:
                plot_sol_2.append(solve4[0] * ((x_func) ** 2) + solve4[1] * x_func + solve4[2])

        tck = interpolate.splrep(np.arange(0, micron, micron / rachet_resolution), plot_sol_2)
        x2_index = np.arange(1 / rachet_resolution, 1, 1 / rachet_resolution)
        y2_index = interpolate.splev(x2_index, tck, der=0)
        pot2_mat.append(y2_index)
    #print(x1_index)
    #print(pot1_mat)
    #print(x2_index)
    #print(pot2_mat)
    return (x1_index, pot1_mat[1:], x2_index, pot2_mat[1:])  # pot1/2 have all the potential profiles durring the run


def normal_run():
    """
    Extraction of data from interface
    dor - alot change from last run, this is just for get the idia
    """
    ion_selection_dict = ion_selection_panel()  # dor - how the ion dict see
    '''Running simulation'''

    # ion_selection_dict, potential_profile = fast_start([1, 2, 3], 1)
    potential_profile = extract_data_from_interface()  # dor - same here

    ion_selection_dict, potential_profile = set_prams_dc_sawtooth(diffusion=1.2 * 10 ** -5, length=1, xc=0.7,
                                                                  dc=0.5, v_max=1, alpha=-1)
    plot_potential_profile(potential_profile[3], potential_profile[1])

    rims = Rims(ion_selection_dict, potential_profile)
    plot_multiple_ions_hist(rims.get_x_to_plot(), rims.resolution)
    # t = timeit(rims.run_rims,number=2)
    # print(t)
    print("SS?: " + str(rims.run_rims()))
    plot_multiple_ions_hist(rims.get_x_to_plot(), rims.resolution)
    vec, ions = rims.get_vec_to_plot()
    plot_average_speed_of_ions(vec, ions)


# this function was built for the parallel calculations
def costume_run_parallel(l_potential_matrix, l_time_vector, l_location_vector, l_length, l_resolution, l_diffusion_dict,
                         l_folder_to_save, first_peak_amplitude, alfa_ratio):
    l_running_time = 0
    x1 = l_location_vector[l_potential_matrix[0].index(max(l_potential_matrix[0]))]
    if (x1 == 0):
        x1 = l_location_vector[l_potential_matrix[0].index(min(l_potential_matrix[0]))]
    x2 = l_location_vector[l_potential_matrix[1].index(max(l_potential_matrix[1]))]
    if (x2 == 0):
        x2 = l_location_vector[l_potential_matrix[1].index(min(l_potential_matrix[1]))]
    l_running_time = running_time_and_delta_t(x1, x2, alfa_ratio, first_peak_amplitude, 2e-5)# need to change the last argument to parametric
    print("l_running_time:" + str(l_running_time))


    x = custom_run(l_potential_matrix, l_time_vector, l_location_vector, l_length, l_resolution, l_diffusion_dict,
                   l_folder_to_save, l_running_time )
    row_for_data_base = []
    special_case = 0  # indicator for letting us know if we insert manually mean velocity 0
    # extarcting data for data base
    i = 0

    #we added this code before in this function
    '''
    x1 = l_location_vector[l_potential_matrix[0].index(max(l_potential_matrix[0]))]
    if (x1 == 0):
        x1 = l_location_vector[l_potential_matrix[0].index(min(l_potential_matrix[0]))]
    x2 = l_location_vector[l_potential_matrix[1].index(max(l_potential_matrix[1]))]
    if (x2 == 0):
        x2 = l_location_vector[l_potential_matrix[1].index(min(l_potential_matrix[1]))]
        '''
    #print("x1: " + str(x1) + " x2: " + str(x2))
    #print("amplitude: " + str(first_peak_amplitude))
    #print("alfa: " + str(alfa_ratio))
    #print("folder to save is: " + str(l_folder_to_save))
    row_for_data_base.append(x1)
    row_for_data_base.append(x2)
    row_for_data_base.append(first_peak_amplitude)
    row_for_data_base.append(alfa_ratio)
    row_for_data_base.append(l_diffusion_dict)
    row_for_data_base.append(l_time_vector[0] / l_time_vector[1])
    row_for_data_base.append(1 / l_time_vector[1])
    ### according to what Gideon said:
    ### we insert by hand velocity =0 iff - the profile is constant on time - x1=x2 & alfa=1
    ###                                   - the peak is in the middle with no respect for the other parameters- x1=x2=0.5

    if (x1 == x2 and row_for_data_base[3] == 1):
        row_for_data_base.append(0)
        special_case = 1
    elif (x1 == x2 and x1 == 0.5):
        row_for_data_base.append(0)
        special_case = 1
    else:
        row_for_data_base.append(x.velocity_means[-1])

    return x.velocity_means[-1], l_folder_to_save, x.time_vec, row_for_data_base, x.css, list(x.potential_profile_mat[0]).index(max(list(x.potential_profile_mat[0]))), list(x.potential_profile_mat[1]).index(max(list(x.potential_profile_mat[1]))), special_case, x.steady_state_code


def create_costume_run_arguments_array(l_pot1, l_pot2):
    array = []
    pot_mat = [[], []]
    for t1_pot in l_pot1:  # go over the first potential profile
        for t2_pot in l_pot2:  # go over the second potential profile
            pot_mat[0] = []
            for pot_per_loc in t1_pot:  # initialize the first potential profile
                pot_mat[0].append(pot_per_loc)
            pot_mat[1] = []
            for pot_per_loc in t2_pot:  # initialize the first potential profile
                pot_mat[1].append(pot_per_loc)
            array.append(list(pot_mat))

    return array


# velocity_mat_dimention is actual minus 1 from the given value
def plot_velocity_per_potential_profile(velocity_array, velocity_mat_dimention, num_of_duty_cycles):
    velocity_per_loc_combination_mat = []  # np.zeros((velocity_mat_dimention-1, velocity_mat_dimention-1))  # the potential for every peak location
    local_velocity_array = velocity_array
    # creating the pot_per_loc_combination_mat
    for i in range(num_of_duty_cycles):
        velocity_per_loc_combination_mat = []
        dc = (i + 1) / (num_of_duty_cycles + 1)
        for j in range(velocity_mat_dimention - 1):
            velocity_per_loc_combination_mat.append(local_velocity_array[:velocity_mat_dimention - 1])
            local_velocity_array = local_velocity_array[velocity_mat_dimention - 1:]

        # plot the mean velocity per peak location 3D
        z = np.array(velocity_per_loc_combination_mat)
        fig = plt.figure()
        p = plt.imshow(z, extent=([0, 1, 0, 1]), origin='lower', cmap='bwr')
        plt.title("VELOCITY MAP [CM/S], D.C. = " + str(dc)[:4])
        plt.xlabel("second peak locaion in [u meter]")
        plt.ylabel("first peak locaion in [u meter]")

        plt.colorbar(p)
        str_dc = str(dc)
        file_name = "dc = " + str(dc) + ".png"
        path_new = os.path.join(get_save_dir(), file_name)
        plt.savefig(path_new)

    return velocity_per_loc_combination_mat


def running_time_and_delta_t(x1, x2, alfa_ratio, first_peak_amplitude,diffusion_coefficient, min_steps = 5):
    first_peak_long = max(x1, 1-x1)
    first_peak_short = min(x1, 1-x1)
    second_peak_long = max(x2, 1-x2)
    second_peak_short = min(x2, 1-x2)
    running_time = 0
    smallest_segment = min(first_peak_short, second_peak_short)

    first_large_incline = (abs(first_peak_amplitude)/first_peak_short)
    first_small_incline = (abs(first_peak_amplitude)/first_peak_long)
    second_large_incline = (abs(first_peak_amplitude*alfa_ratio))/second_peak_short
    second_small_incline = (abs(first_peak_amplitude*alfa_ratio))/second_peak_long

    smallest_incline = min(abs(first_small_incline), abs(second_small_incline))
    smallest_velocity = smallest_incline * diffusion_coefficient #V=E*diffusion_coefficient
    if (abs(first_small_incline) >= abs(second_small_incline)):
        running_time = first_peak_long / smallest_velocity
    elif (abs(first_small_incline) < abs(second_small_incline)):
        running_time = second_peak_long / smallest_velocity

    #delta_t = ((smallest_segment * 10**(-6)) / (max(first_large_incline, second_large_incline) * diffusion_coefficient)) #/ min_steps  #assuming the smallest segment pick's value is 1 [V]

    return  running_time*10**(-6)

def fill_data_base(array_for_data_base):
    with open('data_base.csv', 'a', newline='') as data_base:
        writer = csv.writer(data_base)
        for row in array_for_data_base:
            if (row == "unstable"): row[-1] == "unstable"
            writer.writerow(row)


def create_config_gui(root, gui_rachet_resolution, gui_location_of_x1, gui_location_of_x2,
                      qui_first_peak_amplitude,gui_number_of_first_peak_amplitude, gui_alpha,gui_number_of_alphas,
                      gui_period,gui_number_of_periods,gui_duty_cycle,gui_number_of_duty_cycles, gui_save_in_data_base):
    # Function to be called when the submit button is clicked
    def on_submit():
        start_4 = time.time()
        # Retrieve the values entered by the user
        gui_rachet_resolution.set(gui_rachet_resolution_entry.get())
        gui_location_of_x1.set(gui_location_of_x1.get())
        gui_location_of_x2.set(gui_location_of_x2.get())
        qui_first_peak_amplitude.set(gui_first_peak_amplitude_entry.get())
        gui_number_of_first_peak_amplitude.set(gui_number_of_first_peak_amplitude.get())
        gui_alpha.set(gui_alpha_entry.get())
        gui_number_of_alphas.set(gui_number_of_alphas.get())
        gui_period.set(gui_period_entry.get())
        gui_number_of_periods.set(gui_number_of_periods.get())
        gui_duty_cycle.set(gui_duty_cycle.get())
        gui_number_of_duty_cycles.set(gui_number_of_duty_cycles.get())
        gui_save_in_data_base.set(gui_save_in_data_base.get())
        # Close the window
        root.destroy()

    # Create the main window
    window = tk.Frame(root)
    window.pack()

    label = tk.Label(window, text='RIMS\nThis window will help you configure the parameter for the run\nfor parameters that followed by "number of.." it works as the following'
                                  ' example: first_peak_amplitude = 4\nif number_of_first_peak_amplitude =1 -> we will run with the chosen paramter\n'
                                  'if number_of_first_peak_amplitude=4 -> we will run with {0.25,0.5,0.75,1}  ')
    label.pack()

    # Add a label and text box for spline_function_argument
    gui_rachet_resolution_label = tk.Label(window, text="rachet_resolution- minimum of 4:")
    gui_rachet_resolution_entry = tk.Entry(window, textvariable=gui_rachet_resolution)
    gui_rachet_resolution_label.pack()
    gui_rachet_resolution_entry.pack()

    # Add a label and text box for location_of_x1
    gui_location_of_x1_label = tk.Label(window, text="location_of_x1- 0 -> all options, 1-rachet_resolution -> 1 spot:")
    gui_location_of_x1_entry = tk.Entry(window, textvariable=gui_location_of_x1)
    gui_location_of_x1_label.pack()
    gui_location_of_x1_entry.pack()

    # Add a label and text box for location_of_x2
    gui_location_of_x2_label = tk.Label(window, text="location_of_x2- 0 -> all options, 1-rachet_resolution -> 1 spot:")
    gui_location_of_x2_entry = tk.Entry(window, textvariable=gui_location_of_x2)
    gui_location_of_x2_label.pack()
    gui_location_of_x2_entry.pack()

    # Add a label and text box for the first_peak_amplitude
    gui_first_peak_amplitude_label = tk.Label(window, text="first_peak_amplitude:")
    gui_first_peak_amplitude_entry = tk.Entry(window, textvariable=gui_first_peak_amplitude)
    gui_first_peak_amplitude_label.pack()
    gui_first_peak_amplitude_entry.pack()

    # Add a label and text box for number_of_first_peak_amplitude
    gui_number_of_first_peak_amplitude_label = tk.Label(window, text="number_of_first_peak_amplitude:")
    gui_number_of_first_peak_amplitude_entry = tk.Entry(window, textvariable=gui_number_of_first_peak_amplitude)
    gui_number_of_first_peak_amplitude_label.pack()
    gui_number_of_first_peak_amplitude_entry.pack()

    # Add a label and text box for gui_alfa
    gui_alpha_label = tk.Label(window, text="alpha:")
    gui_alpha_entry = tk.Entry(window, textvariable=gui_alpha)
    gui_alpha_label.pack()
    gui_alpha_entry.pack()

    # Add a label and text box for gui_number_of_alfa
    gui_number_of_alphas_label = tk.Label(window, text="number_of_alpha:")
    gui_number_of_alphas_entry = tk.Entry(window, textvariable=gui_number_of_alphas)
    gui_number_of_alphas_label.pack()
    gui_number_of_alphas_entry.pack()

    # Add a label and text box for gui_period
    gui_period_label = tk.Label(window, text="period:")
    gui_period_entry = tk.Entry(window, textvariable=gui_period)
    gui_period_label.pack()
    gui_period_entry.pack()

    # Add a label and text box for gui_period
    gui_number_of_periods_label = tk.Label(window, text="number_of_period:")
    gui_number_of_periods_entry = tk.Entry(window, textvariable=gui_number_of_periods)
    gui_number_of_periods_label.pack()
    gui_number_of_periods_entry.pack()

    # Add a label and text box for gui_number_of_duty_cycles
    gui_duty_cycles_label = tk.Label(window, text="duty_cycle:")
    gui_duty_cycles_entry = tk.Entry(window, textvariable=gui_duty_cycle)
    gui_duty_cycles_label.pack()
    gui_duty_cycles_entry.pack()

    # Add a label and text box for gui_number_of_duty_cycles
    gui_number_of_duty_cycles_label = tk.Label(window, text="number_of_duty_cycles:")
    gui_number_of_duty_cycles_entry = tk.Entry(window, textvariable=gui_number_of_duty_cycles)
    gui_number_of_duty_cycles_label.pack()
    gui_number_of_duty_cycles_entry.pack()

    # Create a label and a checkbutton
    gui_save_in_data_base_label = tk.Label(window, text="save_in_data_base:")
    gui_save_in_data_base_checkbutton = tk.Checkbutton(window, variable=gui_save_in_data_base, onvalue="YES",
                                                       offvalue="NO")
    # Place the widgets in the GUI
    gui_save_in_data_base_label.pack()
    gui_save_in_data_base_checkbutton.pack()

    # Add a button to submit the form
    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack()

    # Return the window object
    return window

def parsing_gui_configuration(parameter_max, division_of_parameter, parameter_array):
    for i in range(division_of_parameter):
        parameter_array.append((i+1)*(parameter_max/division_of_parameter))
    return
def find_amplitude_and_alpha(pot_mat):
    amp_1 = max(pot_mat[0])
    if amp_1 == 0 :
        amp_1 = min(pot_mat[0])
    amp_2 = min(pot_mat[1])
    if amp_2 == 0:
        amp_2 = max(pot_mat[1])
    l_alpha = amp_2/amp_1
    return (amp_1,l_alpha)

def find_which_profile(old_pot_mat_array, x1_loc, x2_loc):
    max_or_min_1 = 0 # max
    max_or_min_2 = 1 # min
    new_pot_mat_array = []
    peak_1 = 0
    peak_2 = 0
    idx_1 = 0
    idx_2 = 0
    if x1_loc == 0 and x2_loc == 0: return old_pot_mat_array
    for i in old_pot_mat_array:
            peak_1 = max(i[0])
            if peak_1 == 0:
                peak_1 = min(i[0])
            peak_2 = min(i[1])
            if peak_2 == 0:
                peak_2 = max(i[1])
            idx_1 = i[0].index(peak_1)
            idx_2 = i[1].index(peak_2)
            if (idx_1 == x1_loc-1 and idx_2 == x2_loc-1) or (idx_1 == x1_loc-1 and 0 == x2_loc) or (0 == x1_loc and idx_2 == x2_loc-1):
                new_pot_mat_array.append(i)


    return new_pot_mat_array
if __name__ == '__main__':  # important to make resolution the same size as the x/p arrys
    # for GUI
    # Create the root window
    root = tk.Tk()
    root.geometry("1000x1000")

    # Create Tkinter variables for storing the configuration values
    gui_rachet_resolution = tk.IntVar(root)
    gui_location_of_x1 = tk.IntVar(root)
    gui_location_of_x2 = tk.IntVar(root)
    gui_first_peak_amplitude = tk.DoubleVar(root)
    gui_number_of_first_peak_amplitude = tk.IntVar(root)
    gui_alpha = tk.DoubleVar(root)
    gui_number_of_alphas = tk.IntVar(root)
    gui_period = tk.DoubleVar(root)
    gui_number_of_periods = tk.IntVar(root)
    gui_duty_cycle = tk.DoubleVar(root)
    gui_number_of_duty_cycles = tk.IntVar(root)
    gui_save_in_data_base = tk.StringVar(root)

    # Set the default values
    gui_rachet_resolution.set(4)
    gui_location_of_x1.set(0)
    gui_location_of_x2.set(0)
    gui_first_peak_amplitude.set(1)
    gui_number_of_first_peak_amplitude.set(2)
    gui_alpha.set(-1)
    gui_number_of_alphas.set(2)
    gui_period.set(1e-5)
    gui_number_of_periods.set(1)
    gui_duty_cycle.set(0.5)
    gui_number_of_duty_cycles.set(1)
    gui_save_in_data_base.set('YES')

    # Create the GUI
    window = create_config_gui(root, gui_rachet_resolution, gui_location_of_x1, gui_location_of_x2,
                               gui_first_peak_amplitude,gui_number_of_first_peak_amplitude,
                                gui_alpha,gui_number_of_alphas, gui_period,gui_number_of_periods,
                               gui_duty_cycle, gui_number_of_duty_cycles, gui_save_in_data_base)

    # Run the Tkinter event loop
    root.mainloop()

    # Print the configured values
    print(f"gui_rachet_resolution: {gui_rachet_resolution.get()}")
    print(f"gui_location_of_x1:{gui_location_of_x1.get()}")
    print(f"gui_location_of_x2:{gui_location_of_x2.get()}")
    print(f"gui_first_peak_amplitude: {gui_first_peak_amplitude.get()}")
    print(f"gui_number_of_first_peak_amplitude:{gui_number_of_first_peak_amplitude.get()}")
    print(f"gui_alpha: {gui_alpha.get()}")
    print(f"gui_number_of_alphas:{gui_number_of_alphas.get()}")
    print(f"gui_period: {gui_period.get()}")
    print(f"gui_number_of_periods:{gui_number_of_periods.get()}")
    print(f"gui_duty_cycle:{gui_duty_cycle.get()}")
    print(f"gui_number_of_duty_cycles: {gui_number_of_duty_cycles.get()}")
    print(f"gui_save_in_data_base: {gui_save_in_data_base.get()}")

    costume_run_parallel_array = []  # array of arguments for the costume_run_parallel function
    # parsing gui configuration

    # for spline function - location vector and potential profiles
    spline_function_argument = gui_rachet_resolution.get()  # number of costume_run runs =
                                                            # (spline_function_argument -1)^2, minimal value - 4

    first_peak_amplitude_array = []
    parsing_gui_configuration(gui_first_peak_amplitude.get(), gui_number_of_first_peak_amplitude.get(), first_peak_amplitude_array )

    alphas_array = []
    parsing_gui_configuration(gui_alpha.get(), gui_number_of_alphas.get(), alphas_array )

    length = 1  # in um
    resolution = 1000  # number of singular points in the profile (and the length)
    diffusion_dict = {"10^-5": 1.2 * 10 ** -5}  # dictionary of the different diffusion constants
    folder_to_save = "first_run"  # save folder, will be in the output\{date_time}\ path,\
    # handy for lots of simulations in single python run
    x1_array = []
    pot1_array = []
    x2_array = []
    pot2_array = []
    for i in range(gui_number_of_first_peak_amplitude.get()):
        for j in range(gui_number_of_alphas.get()):
            (x1, pot1, x2, pot2) = make_spline_func(spline_function_argument, first_peak_amplitude_array[i], alphas_array[j])
            # x1/2 is the location vector
            x1_array.append(x1)
            pot1_array.append(pot1)
            x2_array.append(x2)
            pot2_array.append(pot2)

    #creating potential matrix array- each object in the array is 1 potential matrix for the run
    temp_potential_mat_array = []
    potential_mat_array = []
    for i in range(len(pot1_array)):
        potential_mat = [[],[]]
        potential_mat  = (create_costume_run_arguments_array(pot1_array[i], pot2_array[i]))
        temp_potential_mat_array.append(potential_mat)
    for i in range(len(temp_potential_mat_array)):
        for j in range(len(temp_potential_mat_array[0])):
            potential_mat_array.append(temp_potential_mat_array[i][j])

    print("this is the old potential array:")
    print(potential_mat_array)
    potential_mat_array = find_which_profile(potential_mat_array, gui_location_of_x1.get(), gui_location_of_x2.get())
    print("this is the new potential array:")
    print(potential_mat_array)



    location_vector_array = []
    print(x1_array[0])
    location_vector = [0]
    for i in x1_array[0]:  # initialize the location vector
        location_vector.append(i)  # in um
    location_vector.append(1)
    print(location_vector)
    # the physical location of the potential listed, all the value at the first column
    # will be at x=location_vector[0], if the vector include x=0 but not x=length (or the other way around)
    # the code will force the start and the end to the same potential (not tested),
    # if neither listed both will be set on 0

    #running_time_and_delta_t(x1,x2,alfa_ratio,first_peak_amplitude, diffusion_dict)

    mean_velocity_dc_array = []  # [mean velocity, dc]

    array_for_data_base = []

    # duty cycle and flashing period
    periods_array = []
    parsing_gui_configuration(gui_period.get(), gui_number_of_periods.get(), periods_array)
    dc_array = []
    parsing_gui_configuration(gui_duty_cycle.get(), gui_number_of_duty_cycles.get(), dc_array)

    time_vector_array = []
    for dc in dc_array:
        for period in periods_array:
            time_vector_array.append([dc * period, period])

    run_index = 1

    print(potential_mat_array)
    for dc in dc_array:
        for period in periods_array:
            for pot_mat in potential_mat_array:
                    folder_to_save = ("run_number_" + str(run_index))
                    run_index += 1
                    temp_pot_mat = copy.deepcopy(pot_mat)
                    temp_pot_mat[0].insert(0, 0)
                    temp_pot_mat[0].append(0)
                    temp_pot_mat[1].insert(0, 0)
                    temp_pot_mat[1].append(0)
                    time_vector=[]
                    time_vector.append(dc * period)
                    time_vector.append(period)
                    amplitude, alpha = find_amplitude_and_alpha(temp_pot_mat)
                    costume_run_parallel_array.append(
                        tuple((list(temp_pot_mat), time_vector, location_vector, length, resolution, diffusion_dict,
                               folder_to_save, amplitude, alpha)))

    print(len(costume_run_parallel_array))
    for i in costume_run_parallel_array:
        print(i)
    # run parallel

    with Pool() as pool:
        # call the same function with different data in parallel
        for mean_velocity_and_dc in pool.starmap(costume_run_parallel, costume_run_parallel_array):
            # report the value to show progress
            print(mean_velocity_and_dc)
            if (mean_velocity_and_dc[-2] == 1):  # special case
                mean_velocity_dc_array.append((mean_velocity_and_dc[0]))  # for the hit map
                array_for_data_base.append(mean_velocity_and_dc[3])
            else:
                if (mean_velocity_and_dc[-1] != 3):
                    mean_velocity_dc_array.append((mean_velocity_and_dc[0]))  # for the hit map
                    array_for_data_base.append(mean_velocity_and_dc[3])
                else:
                    mean_velocity_dc_array.append(np.nan)  # for the hit map
                    mean_velocity_and_dc[3][-1] = "unstable"
                    array_for_data_base.append(mean_velocity_and_dc[3])

    # print (array_for_data_base)
    print(plot_velocity_per_potential_profile(mean_velocity_dc_array, spline_function_argument, len(costume_run_parallel_array)))
    if (gui_save_in_data_base.get() == "YES"):
        print("saving in data base")
        fill_data_base(array_for_data_base)


