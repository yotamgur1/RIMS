import matplotlib.pyplot as plt
import numpy as np

from general_class import *
from interface import *
from rims_main import Rims


def set_prams_dc_sawtooth(diffusion, length, xc, dc, v_max, alpha, flash_frequency, z=1):
    length_cm = micron_to_cm(length)
    a = length_cm * xc
    b = length_cm - a
    x = np.linspace(0, length_cm, num=get_setting('RESOLUTION'))
    f1 = v_max * np.divide(x, a)
    f2 = v_max * np.divide((x - length_cm), (-b))
    step = np.heaviside(x - a, 1)
    pos = f1 - step * f1 + step * f2
    neg = np.multiply(pos, alpha)
    potential_mat = np.vstack((pos, neg))

    T = 1 / flash_frequency
    t_vec = np.array([dc * T, T])
    potential_profile = [length_cm, x, t_vec, potential_mat]
    ion_dict = {f'Custom, {np.round(diffusion,2)}': diffusion}
    return [ion_dict, potential_profile]


def run_dc_sawtooth_vel_check(alpha, intervals,  max_v, run_excel_file=None, ss_criteria=0):
    if run_excel_file is None:
        run_excel_file = []
    dc_list = np.linspace(0, 1, intervals)
    sim_list = []
    v_0 = max_v * 1 * 10 ** -5 * betta() * 1 / 10 ** -4
    flash_frequency = v_0/0.0001  # ?????
    for dc in dc_list:
        run_id = f"{np.round(alpha,2)} {np.round(dc,2)}"
        ion_selection_dict, potential_profile = set_prams_dc_sawtooth(diffusion=1.2 * 10 ** -5, length=1, xc=0.7,
                                                                      dc=dc, v_max=max_v, alpha=alpha,
                                                                      flash_frequency=flash_frequency)
        r = Rims(ion_selection_dict, potential_profile, ss_criteria)
        sim_list.append(r)
        ss_code = sim_list[-1].run_rims()
        rims_save_dir = sim_list[-1].save_data(run_id)
        rims_parameters = sim_list[-1].get_class_str_parameters()
        if len(run_excel_file) == 0:
            titles = ["alpha", "dc", "last velocity means", *rims_parameters[0], "dir"]
            run_excel_file.append(titles)
        run_excel_file.append([alpha, dc, sim_list[-1].velocity_means[-1], *rims_parameters[1], f'=HYPERLINK("{rims_save_dir}")'])
    vecs = []
    stdes = []
    for sim in sim_list:
        # vec_data_from_sim = sim.get_vec_and_err_to_plot()
        # stdes.append(np.sqrt(np.mean(np.square((vec_data_from_sim[2][-5:])))))
        vec_data_from_sim = sim.get_vec_to_plot()
        vecs.append(np.mean(vec_data_from_sim[0][-5:]))
        stdes.append(np.std(vec_data_from_sim[0][-5:]))

    vecs = np.array(vecs)
    stdes = np.array(stdes)

    return dc_list, vecs / v_0, stdes/v_0


def full_fig_6_reconstruct(alpha_int, intervals, max_v, ss_criteria_by_last_run=False):
    reset_save_dir()
    t = datetime.now()
    ss_criteria = 0
    if ss_criteria_by_last_run:
        ss_criteria = get_max_vec_by_run("2022-06-08_15-18-16")
    alphas = np.linspace(0, -1, int(1/np.abs(alpha_int))+1)
    run_xl_file = []
    fig_vectors = []
    fig_x_vector = np.round(alphas, 3).astype(str)
    fig, ax = plt.subplots()
    for alpha in alphas:
        x, y, err_y = run_dc_sawtooth_vel_check(alpha, intervals,  max_v, run_xl_file, ss_criteria)
        fig_vectors.append(np.hstack([y, err_y]))
        ax.errorbar(x, y, yerr=err_y)
    df_run_data = pd.DataFrame(run_xl_file[1:], columns=run_xl_file[0])
    df_run_data.to_csv(os.path.join(get_save_dir(), "run_data.csv"))
    fig_mat = np.array(fig_vectors)
    df_fig_vectors = pd.DataFrame(fig_mat.T, columns=fig_x_vector)
    df_fig_vectors.to_csv(os.path.join(get_save_dir(), "fig_vectors.csv"))
    t2 = datetime.now() - t
    ax.legend(np.round(alphas, 3).astype(str))
    ax.grid(True)
    print(f"simulation time (fig 6 reconstruct): {t2}")
    fig.savefig(os.path.join(get_save_dir(), "run_data"))
    # fig.show()


def full_fig_3_reconstract(hz_int_count):
    """
    dor - need to rewrite
    """
    t = datetime.now()
    ds = np.array([1.2 * 10 ** -5, 2 * 10 ** -5])
    frequencies = np.linspace(40, 120, hz_int_count) * 1000
    plots = []
    for d in ds:
        vecs = []
        for hz in frequencies:
            ion_selection_dict, potential_profile = set_prams_dc_sawtooth(diffusion=d, length=1, xc=0.7,
                                                                          dc=0.25, v_max=2.5, alpha=-0.5,
                                                                          flash_frequency=hz)
            sim = Rims(ion_selection_dict, potential_profile)
            sim.run_rims()
            vecs.append(np.mean(sim.get_vec_to_plot()[0][-5:]))
        plots.append([frequencies, vecs])
    for p in plots:
        plt.plot(*p)
    plt.legend(ds.astype(str))
    plt.grid()
    t2 = datetime.now() - t
    print(f"{t2}")
    # plt.show()


# region custom run

def custom_run(potential_matrix, time_vector, location_vector, length, resolution, diffusion_dict, folder_to_save, new_running_time):

    ion_selection_dict, potential_profile2 = set_prams_dc_sawtooth(diffusion=1.2 * 10 ** -5, length=1, xc=0.7,
                                                                  dc=0.79, v_max=1, alpha=-0.8,
                                                                  flash_frequency=3.8)

    potential_profile = custom_run_build(potential_matrix, time_vector, location_vector, length, resolution)
    rims = Rims(diffusion_dict, potential_profile , new_running_time)
    rims.run_rims(new_running_time)
    #rims.save_data(folder_to_save)
    return rims


def custom_run_build(potential_matrix, time_vector, location_vector, length, resolution):
    new_potential_matrix, new_location_vector = interpolate_potential(potential_matrix,
                                                                      location_vector, length, resolution)
    potential_profile = [micron_to_cm(length), micron_to_cm(np.array(new_location_vector)),
                         np.array(time_vector), np.array(new_potential_matrix)]
    return potential_profile


def interpolate_potential(potential_matrix, location_vector, length, resolution):
    x_for_int = np.linspace(0, length, resolution)
    new_profile_mat = []
    location_vector_temp = []
    for profile in potential_matrix:
        location_vector_temp = location_vector.copy()
        if 0 not in location_vector_temp and length not in location_vector_temp:
            location_vector_temp.insert(0, 0)
            profile.insert(0, 0)
            location_vector_temp.append(length)
            profile.append(0)
        elif 0 not in location_vector_temp:
            location_vector_temp.insert(0, 0)
            profile.insert(0, profile[-1])
        elif length not in location_vector_temp:
            location_vector_temp.append(length)
            profile.append(profile[0])
        new_profile_mat.append(np.interp(x_for_int, location_vector_temp, profile))
    return new_profile_mat, x_for_int


# endregion

# region main helpers
def fast_start(ion_lst, csv_num):
    ions_for_simulation_dict = {}
    for i in ion_lst:
        ion_arg, diff = list(diffusion_coefficient_dict.items())[i - 1]
        ions_for_simulation_dict[ion_arg] = diff

    L, t_vec, potential_mat = fast_select_csv_file(csv_num)
    x = np.linspace(0, L, num=potential_mat.shape[1])

    potential_profile = [L, x, t_vec, potential_mat]
    return ions_for_simulation_dict, potential_profile


def fast_select_csv_file(k):
    folder = r'potential profile sources/'

    items = os.listdir(folder)
    valid_files = []
    for i, entry in enumerate(items, 1):
        if os.path.isfile(os.path.join(folder, entry)) and entry.endswith('csv') or entry.endswith('txt'):
            valid_files.append(entry)
    file = valid_files[k]
    scalar_x, vec_t, mat_v = load_data_from_csv(folder + file)
    return scalar_x, vec_t, mat_v
# endregion