from defines import *
import os

"""
interface.py

Extraction of data from user and external files
"""

'''----------------------------------------------------------------------
                            IMPLEMENTATION
----------------------------------------------------------------------'''

RUN_NUMBER = 0

print("-------------------------------------------------------")
print("     RIMS - Ratchet based Ion Movement Simulator")
print("     version " + VERSION)
print("-------------------------------------------------------")

def headline_panel():
    global RUN_NUMBER
    print("     run " + str(RUN_NUMBER))
    RUN_NUMBER += 1
    return


def simulation_over_panel():
    print("\n-------------------------------------------------------")
    print("                 Simulation over")
    print("-------------------------------------------------------")


def execution_rerun_panel():
    rerun = input("\nPress y and ENTER to run an new simulation, otherwise press ENTER...\n")
    if rerun == 'y':
        return True
    return False


def exit_panel():
    print("\n-------------------------------------------------------")
    print("                 Exiting RIMS")
    print("-------------------------------------------------------")
    input("\npress ENTER to exit...\n")
    exit()


def extract_enable_video():
    print("Add video output?\n1)Yes\n2)No")
    vid_num = input_check_int("Select 1 or 2 = ", [1, 2])
    if vid_num == 1:
        return True
    return False


def extract_data_from_interface():
    """
    Extracts the ratchet parameters from the user as a list in form:
    potential_profile = [L, x, t_vec, potential_mat]
    L = ratchet length x axis
    x = x axis vector
    t_vec = timing vector. t_vec[last] = T
    potential_mat = matrix of the potential profiles. rows-different profiles, cols-potential at location x
    """
    print("-------------------------------------------------------")
    print("             Step 2- Configure the ratchet")
    print("-------------------------------------------------------")

    print("\nEnter ratchet function:\n\t0)Load from csv sheet\n\t1)Saw wave\n\t2)Double Sin\n")
    ratchet_number = input_check_int("Ratchet function number = ", [0, 1, 2])
    if ratchet_number != 0:
        if ratchet_number == 1:
            '''In console ratchet functions'''
            print("Please describe the potential profile applied on the system.\n")
            print("                    _          ")
            print(r"        / |\         |         ")
            print(r"      /   | \        | A[v]    ")
            print(r"    /     |  \       |         ")
            print(r"  /       |   \     _|         ")
            print(r"                               ")
            print(r"  \______/\____/               ")
            print("    a[um]   b[um]            \n")

            a = input_check_float("\ta[um] = ") * pow(10, -4)
            b = input_check_float("\tb[um] = ") * pow(10, -4)
            L = a + b
            amp = input_check_float("\tA[v] = ")
            x = np.linspace(0, a + b, num=load_one_setting(settings_filename, 'RESOLUTION'))
            f1 = amp * np.divide(x, a)
            f2 = amp * np.divide((x - (a + b)), (-b))
            step = np.heaviside(x - a, 1)
            pos = f1 - step * f1 + step * f2

        else:
            print("Please enter ratchet sin wave parameters.\n")
            print("                    _          ")
            print(r"        / |\        _| a2[v]   ")
            print(r"      /   | \        |         ")
            print(r"    /     |  \       | a1[v]   ")
            print(r"  /       |   \     _|         ")
            print(r"                               ")
            print(r"  \____________/               ")
            print("        L[um]                \n")
            print("qV(x) = a1 * sin(2pi * x / L) + a2 * sin(4pi * x / L)\n")
            L = input_check_float("\tL[um] = ") * pow(10, -4)
            a1 = input_check_float("\ta1[v] = ")
            a2 = input_check_float("\ta2[v] = ")

            x = np.linspace(0, L, num=load_one_setting(settings_filename, 'RESOLUTION'))
            pos = a1 * np.sin(2 * np.pi * x / L) + a2 * np.sin(4 * np.pi * x / L)
        print("Select relative amplitude for the negative profile.\nMeaning V- = V+ * -ALPHA")
        print("Enter ALPHA as positive float or integer")
        alpha = -abs(input_check_float("\tALPHA = "))
        neg = np.multiply(pos, alpha)
        potential_mat = np.vstack((pos, neg))

        print("\nEnter ratchet flashing frequency in Hz: (can use factors of K,M,G)")
        flash_frequency = -1
        tries = 0
        while flash_frequency <= 0:
            if tries > 0:
                print("\tfrequency takes values equal or larger than 1")
            try:
                raw_input = input("Ratchet frequency [Hz] = ")
                converted_to_int = raw_input
                converted_to_int = converted_to_int.replace('k', '').replace('K', '').replace('M', '').replace('G', '')
                flash_frequency = int(converted_to_int)
                if 'k' in raw_input or 'K' in raw_input:
                    flash_frequency *= 1000
                if 'M' in raw_input:
                    flash_frequency *= 1000000
                if 'G' in raw_input:
                    flash_frequency *= 1000000000

            except ValueError:
                print("\tPlease enter an integer as specified above")
                tries += 1
                continue
            tries += 1
        print("\nEnter ratchet duty cycle from 0-1:")
        dc = -1
        tries = 0
        while dc < 0 or dc > 1:
            if tries > 0:
                pass  # pass  # warning_to_console('DC takes float values from (0-1)')
            try:
                dc = float(input("DC = "))
            except ValueError:
                tries += 1
                continue
            tries += 1
        T = 1 / flash_frequency
        t_vec = np.array([dc * T, T])

    else:  # data from csv
        L, t_vec, potential_mat = select_csv_file()
        x = np.linspace(0, L, num=potential_mat.shape[1])

    potential_profile = [L, x, t_vec, potential_mat]

    print("\n-------------------------------------------------------")
    print("             Starting simulation")
    print("-------------------------------------------------------")
    print("\nBuilding Monte Carlo calculation system")

    print("Initial location of every ion is uniformly randomized over [0," + str(L * pow(10, 4)) + "um]\n")
    return potential_profile


def ion_selection_panel():
    """
    Extracts ions & diffusion coefficient dictionary from the user
    """
    print("-------------------------------------------------------")
    print("             Step 1- Configure the system")
    print("-------------------------------------------------------")

    print("\nSelect an ion to be simulated from the following list:")
    print("\t0) Manually insert diffusion coefficient")
    for i, ion_tup in enumerate(diffusion_coefficient_dict.items(), 1):
        print('\t' + str(i) + ') ' + ion_tup[0])
    print("\nFor multiple ions, type comma between inputs")

    input_valid = False
    ions_for_simulation_dict = {}
    while not input_valid:
        manual_counter = 1
        user_input = input("Enter your selection:")

        '''Parse arguments'''
        if ',' in user_input:
            user_input_split = user_input.split(',')
        else:
            user_input_split = [user_input]

        for arg in user_input_split:
            arg = arg.strip(' ')
            try:
                int_arg = int(arg)
                num_options = len(diffusion_coefficient_dict.items())
                if int_arg > num_options:
                    # warning_to_console(
                    #     str(int_arg) + ' is out of range. There are only ' + str(num_options) + ' options.')
                    # print(arg + ' not included.')
                    continue
            except ValueError:
                # warning_to_console("Please enter numbers separated by commas only.")
                print('\tobject ' + str(arg) + ' not comprehensible and not included')
                continue

            if arg == '0':  # manual entry
                clear = False
                diff = 0
                while not clear:
                    try:
                        diff = float(input("Enter diffusion coefficient [cm^2/sec]:"))
                        if diff >= 0.0001:
                            # warning_to_console("Diffusion coefficient must be under 0.0001 [cm^2/sec]")
                            continue
                        clear = True
                    except ValueError:
                        # warning_to_console("Please enter a float as specified above")
                        continue
                ion_arg = 'manual_input' + str(manual_counter)
                manual_counter += 1
            else:
                ion_arg, diff = list(diffusion_coefficient_dict.items())[int(arg) - 1]

            ions_for_simulation_dict[ion_arg] = diff

        if len(list(ions_for_simulation_dict.items())) > 0:
            input_valid = True

    '''Print ions to be simulated'''
    print('\nIons and diffusion coefficients to be simulated:')
    for key, value in ions_for_simulation_dict.items():
        print(key, ' : ', value, '[cm^2,sec]')
    return ions_for_simulation_dict


def input_check_int(msg, desired_range):
    val = 'blank'
    while val not in desired_range:
        try:
            val = int(input(msg))
        except ValueError:
            # warning_to_console("Please enter an integer as specified above")
            continue
    return val


def input_check_float(msg):
    val = 'blank'
    clear = False
    while not clear:
        try:
            val = float(input(msg))
            clear = True
        except ValueError:
            # warning_to_console("Please enter an integer or a float as specified above")
            continue
    return val


def get_time_stamp(time_start):
    ts = str(time_start.strftime("%x")).replace('/', '-') + '_' \
         + str(time_start.strftime("%X")).replace(':', '')
    return ts


def select_csv_file():
    folder = r'potential profile sources/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    items = os.listdir(folder)
    valid_files = []
    print("\nPotential profiles csv/txt files found in folder '" + folder + "':")
    for i, entry in enumerate(items, 1):
        if os.path.isfile(os.path.join(folder, entry)) and entry.endswith('csv') or entry.endswith('txt'):
            print(str(i) + ') ' + entry)
            valid_files.append(entry)
    if len(valid_files) == 0:
        # warning_to_console("No .csv or .txt files found in " + folder)
        print("Please refer to README for more details on the csv ratchet parameter file")
        print("Save a csv parameter file to " + folder + " and relaunch RIMS")
        exit_panel()

    print("Select the number of the profile you wish load:")
    file = valid_files[input_check_int("file number: ", range(1, len(valid_files) + 1)) - 1]
    scalar_x, vec_t, mat_v = load_data_from_csv(folder + file)
    return scalar_x, vec_t, mat_v


def load_data_from_csv(csv_file_path):
    """
    :param csv_file_path - path to a ratchet profile setting file set as follows
    Loads csv pre-saved ratchet parameter file. Refer to README for more details on the csv format
    """
    vec_header = np.loadtxt(csv_file_path, max_rows=1, delimiter=',')  # holds the first data row
    scalar_x = vec_header[0] * pow(10, -4)  # width of the profile
    vec_t = vec_header[1:]  # timings vector
    vec_t = np.array([t * pow(10, -6) for t in vec_t])
    mat_v = np.loadtxt(csv_file_path, skiprows=1, delimiter=',')  # potential profiles
    '''In case only one profile was detected in the file'''
    if mat_v.ndim == 1:
        print("only 1 profile was detected. adding a second profile such that V2= " + str(ALPHA) + "*V1")
        mat_v = np.vstack((mat_v, ALPHA * mat_v))

    return scalar_x, vec_t, mat_v
