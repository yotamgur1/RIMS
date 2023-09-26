import csv
from datetime import datetime
import os
import pandas as pd
import numpy as np
import scipy.constants as constants

# region settings and file support
def get_root_folder():
    """
    the only place in the code with direct folder, use for change saves and read folder
    """
    return os.path.join(os.getcwd())


SettingsFilePath = os.path.join(get_root_folder(), 'settings.csv')
SettingsDict = None
SimDataDict = None
save_dir = None


def load_setting_dict():
    global SettingsDict
    SettingsDict = {}
    with open(SettingsFilePath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0][0] in ['#', '\n']:
                continue
            value = row[1]
            if value == 'true' or value == 'True':
                value = True
            elif value == 'false' or value == 'False':
                value = False
            else:
                try:
                    value = float(row[1])
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
            SettingsDict[row[0]] = value


def get_setting(setting_name):
    if SettingsDict is None:
        load_setting_dict()
    if setting_name in SettingsDict:
        return SettingsDict[setting_name]


def micron_to_cm(micron):
    return micron/10000


def get_save_dir():
    global save_dir
    if save_dir is None:
        save_dir = os.path.join(get_root_folder(), 'output', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(save_dir)
    return save_dir


def reset_save_dir():
    global save_dir
    save_dir = None


def get_max_vec_by_run(run_folder):
    """
    get run max velocity by run folder
    the func is too specific, need to fix?
    """
    column = "last velocity means"
    path = os.path.join(get_root_folder(), 'output', run_folder)
    file = "run_data.csv"
    df = pd.read_csv(os.path.join(path, file))
    return np.max(np.abs(df[column]))

# endregion

# region Calculations nnd main run support


def get_electric_field_vec(potential_profile, L, res):
    """
    Derives the electric field from the potential, E(x,t) saves it as attribute
    """
    dx = L / res
    electric_field_vec = -np.gradient(potential_profile, dx)
    '''plot E & V'''
    return electric_field_vec


def get_electric_velocity(x, gamma, ef, L):
    index_in_array = (x * len(ef) / L).astype(int)
    electric_field = ef[index_in_array]
    return electric_field / gamma


e = constants.e


def betta():
    return 1/(8.617*10**-5*get_setting("TEMPERATURE"))

# endregion
