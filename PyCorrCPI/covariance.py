from .data import *
from .helpers_numba import *
from .imports import *
from .output_functions import *

## Constants

u = 1.660538921e-27
e = 1.60217663e-19
k = 8.9875517923e9
hbar = 1.0546e-34
kb = 1.380649e-23
cm_to_J = 1.986e-23
c_cm_s = 2.997e10
hbar = 1.054571817e-34
bohr_radius = 5.29177210903e-11
p_au_fac = hbar / bohr_radius  # factor for atomic units of momentum
p_au_KE_eV_fac = (p_au_fac) ** 2 / (u * e)  # factor for converting from atomic units of momentum to KE in eV
cm_to_hartree = 1 / 219474.6
hartree_to_eV = 27.211396132
u_to_amu = 1 / 5.4857990943e-4
bohr_to_angstrom = 0.529177211


output_function_dict = {
    "beta_psum_KER": beta_psum_KER,
    "output_format_TfAcAc": output_format_TfAcAc,
}

param_dict = {
    "spinewidth": 3,
    "linewidth": 2,
    "ticklength": 6,
    "tickwidth": 3,
    "ticklabelsize": 16,
    "axislabelsize": 20,
}


def neaten_plot(ax, param_dict):
    """Function used to neaten up covariance plots"""
    ax.tick_params(
        labelsize=param_dict["ticklabelsize"],
        length=param_dict["ticklength"],
        width=param_dict["tickwidth"],
    )
    ax.xaxis.get_label().set_fontsize(param_dict["axislabelsize"])
    ax.yaxis.get_label().set_fontsize(param_dict["axislabelsize"])
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(param_dict["spinewidth"])


def generate_contingent_covariance(covariance_output_list, contingent_filters, example_covariance):
    cont_covariance_output = ContCovariance(contingent_filters, covariance_output_list)

    cont_covariance_output.get_details(example_covariance)
    cont_covariance_output.calc_cont_covar()

    return cont_covariance_output


def calc_covariance(
    dataset,
    ion_list,
    dim_list,
    bin_list,
    store_coincs=False,
    update_dataset=True,
    verbose=False,
    remove_autovariance=True,
    custom_function=False,
    n_EiEj=1,
    n_EiEjEk=1,
    n_EijEk=1,
    n_EiEjEkEl=1,
    n_EijEkl=1,
    n_EijkEl=1,
    n_EiEjEkl=1,
    filter_function=False,
    filter_max=np.inf,
    max_shot=None,
    only_coincidence=False,
    contingent=False,
    contingent_filters=None,
):
    """Function for initializing an instance of the Covariance class

    :param ion_list: list of n ions to be used in the nfold covariance calculation
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    :param store_coincs: NOT USED CURENTLY
    :param update_dataset: NOT USED CURRENTLY
    :param verbose: print more details during the calculation, default=False
    :param remove_autovariance: If True, automatically removes autovariance contributions if two ions
        the same. Default=True
    :param custom_function: Custom function for formatting the covariance output. If False,
        a default function is used. Default=False
    :param filter_function: Custom function used for filtering coincidences. If False, the default
        (absolute magnitude of summed momenta calculation) is used
    :param filter_max: Max value of the output of the filter function for an acceptable coincidence.
        Default is np.inf to ignore filtering
    :param max_shot: Maximum shot used in the calculation. Default=None
    :param only_coincidence: If True, only calculate the leading term (the true coincidences). Default = False
    :param n_EiEj: number of times to calculate the <A><B> term in two-fold covariance
    :param n_EiEjEk: number of times to calculate the <A><B><C> term in three-fold covariance
    :param n_EijEk: number of times to calculate the <AB><C> type terms in three-fold covariance
    :param n_EiEjEkEl: number of times to calculate the <A><B><C><D> term in four-fold covariance
    :param n_EijEkl: number of times to calculate the <AB><CD> type terms in four-fold covariance
    :param n_EijkEl: number of times to calculate the <ABC><D> type terms in four-fold covariance
    :param n_EiEjEkl: number of times to calculate the <A><B><CD> type terms in four-fold covariance
    :param contingent: is the covariance calculation contingent? default=False
    :param contingent_filter: filter object used in the calculation if contingent. default=None
    """

    if not contingent:
        covariance_output = Covariance(
            dataset,
            ion_list,
            dim_list,
            bin_list,
            store_coincs=store_coincs,
            update_dataset=update_dataset,
            verbose=verbose,
            remove_autovariance=remove_autovariance,
            custom_function=custom_function,
            n_EiEj=n_EiEj,
            n_EiEjEk=n_EiEjEk,
            n_EijEk=n_EijEk,
            n_EiEjEkEl=n_EiEjEkEl,
            n_EijEkl=n_EijEkl,
            n_EijkEl=n_EijkEl,
            n_EiEjEkl=n_EiEjEkl,
            filter_function=filter_function,
            filter_max=filter_max,
            max_shot=max_shot,
            only_coincidence=only_coincidence,
            contingent=contingent,
        )

        covariance_output.calc_covariance()

        return covariance_output

    else:
        # cont_covariance_output=ContCovariance(contingent_filters)

        covariance_output_list = []

        for contingent_filter in contingent_filters:
            covariance_output = Covariance(
                dataset,
                ion_list,
                dim_list,
                bin_list,
                store_coincs=store_coincs,
                update_dataset=update_dataset,
                verbose=verbose,
                remove_autovariance=remove_autovariance,
                custom_function=custom_function,
                n_EiEj=n_EiEj,
                n_EiEjEk=n_EiEjEk,
                n_EijEk=n_EijEk,
                n_EiEjEkEl=n_EiEjEkEl,
                n_EijEkl=n_EijEkl,
                n_EijkEl=n_EijkEl,
                n_EiEjEkl=n_EiEjEkl,
                filter_function=filter_function,
                filter_max=filter_max,
                max_shot=max_shot,
                only_coincidence=only_coincidence,
                contingent=contingent,
                contingent_filter=contingent_filter,
            )

            covariance_output.calc_covariance()

            # cont_covariance_output.add_covariance(covariance_output)

            covariance_output_list.append(covariance_output.output_array)

        cont_covariance_output = generate_contingent_covariance(
            covariance_output_list, contingent_filters, covariance_output
        )

        del covariance_output
        # cont_covariance_output.calc_cont_covar()

        return cont_covariance_output


@njit()
def calculate_indexes(A_idx_dict, shot_array_A, shot_array, A_arr):
    """Finds the first and last index in an ion data array corresponding to each laser shot
    that will be used in the covariance calculation. Precalculating this dictionary speeds up
    the calculation, particularly for higher fold covariances and/or if recalculating terms
    multiple times.
    !This function assumes that the data is sorted by ascending shot number!


    :param A_idx_dict:  the (empty) dictionary which will be filled with (shot, initial_index, final_index)
    :param shot_array_A:  array of laser shots used in the covariance calculation, for which ion A is detected
    :param shot_array: array of total laser shots used in the covariance calculation
    :param A_arr: the ion A data array ([])
    """

    last_index_A_old = 0
    A_shots = A_arr[:, 3]
    found = False
    for shot in shot_array:
        found, first_index_A = find_first(shot, A_shots[last_index_A_old:])
        if found:
            last_index_A = find_last(shot, A_shots[first_index_A + last_index_A_old :])
            abs_first_index_A = last_index_A_old + first_index_A
            abs_last_index_A = last_index_A_old + first_index_A + last_index_A
            dict_arr = np.asarray([abs_first_index_A, abs_last_index_A + 1], dtype="float64")
            A_idx_dict[shot] = np.asarray([abs_first_index_A, abs_last_index_A + 1], dtype="float64")
            last_index_A_old += first_index_A + last_index_A
        else:
            # No ion A in this shot
            A_idx_dict[shot] = np.asarray([0.0, 0.0], dtype="float64")


@njit(parallel=True, fastmath=True)
def compute_2fold_covariance_term_iterate(output_arr, dim_list):
    """calculate 2fold covariance from its consistuent terms"""
    for x in prange(dim_list[0]):
        for y in prange(dim_list[1]):
            for z in prange(dim_list[2]):
                output_arr[-1, x, y, z] = output_arr[0, x, y, z] - output_arr[1, x, y, z]
    return output_arr


@njit(parallel=True, fastmath=True)
def compute_3fold_covariance_term_iterate(output_arr, dim_list):
    """calculate 3fold covariance from its consistuent terms"""
    for x in prange(dim_list[0]):
        for y in prange(dim_list[1]):
            for z in prange(dim_list[2]):
                output_arr[-1, x, y, z] = (
                    output_arr[0, x, y, z]
                    - output_arr[1, x, y, z]
                    - output_arr[2, x, y, z]
                    - output_arr[3, x, y, z]
                    + 2 * output_arr[4, x, y, z]
                )
    return output_arr


@njit(parallel=True, fastmath=True)
def compute_4fold_covariance_term_iterate(output_arr, dim_list):
    """calculate 4fold covariance from its consistuent terms"""
    for x in prange(dim_list[0]):
        for y in prange(dim_list[1]):
            for z in prange(dim_list[2]):
                output_arr[-1, x, y, z] = (
                    output_arr[0, x, y, z]
                    - (
                        output_arr[1, x, y, z]
                        + output_arr[2, x, y, z]
                        + output_arr[3, x, y, z]
                        + output_arr[4, x, y, z]
                    )
                    - (output_arr[5, x, y, z] + output_arr[6, x, y, z] + output_arr[7, x, y, z])
                    + 2
                    * (
                        output_arr[8, x, y, z]
                        + output_arr[9, x, y, z]
                        + output_arr[10, x, y, z]
                        + output_arr[11, x, y, z]
                        + output_arr[12, x, y, z]
                        + output_arr[13, x, y, z]
                    )
                    - 6 * output_arr[14, x, y, z]
                )
    return output_arr


@njit
def calc_Cab(
    output_array,
    output_function,
    shot_array,
    ion_array_list,
    idx_dict_list,
    mass_list,
    dim_list,
    bin_list,
    shift_num_list,
    n_calc,
    n_shots,
    term_counter,
    autovariance_array,
    use_filter_function,
    filter_function,
    filter_max,
):
    """Find double coincidences, transform into output frame and sum.
    Acts in place on the general output array. This function calculates
    all terms of the covariance expression, with appropriate values for the shot
    shifting index.

    :param ouput_array: array which stores output of covariance calculation
    :param output_function: the function which takes the coincidence and converts into the observables of interest
    :param shot_array: array of the unique shots in the dataset over which the covariance is calculated
    :param ion_array_list: data arrays of the three ions, [px,py,pz,shot,pmag] format
    :param idx_dict_list: list of the precalculated dictionaries of shot indices for each ion
    :param mass_list: mass of each ion in amu. Passed to the output function
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    :param shift_num_list: list of shot shift numbers applied to the ions beyond ion A
    :param n_calc: number of times the term will be recalculated.
    :param n_shots: number of shots covariance is calculated over.
    :param term_counter: index of the term in the epression to be calculated.
    :param autovariance_array: array of pairs of ions which are the same to exclude autovariance/trivial coincidence
    :param use_filter_function: 1 if filtering on some criteria of coincidences
    :param filter_function: function used for filtering, takes in (vec_list,mag_list,mass_list)
    :param filter_max: if output of filter_function is greater than filter_max, coincidence is ignored
    """

    ## as another speedup, we now precalculate the normalization needed to the term
    ## to save on doing extra operations on the entire array later
    addval = 1.0 / (n_shots * n_calc)

    ## do the shot shifting
    A_shot_array = shot_array
    B_shot_array = np.roll(shot_array, -shift_num_list[0])

    A_ion_array, B_ion_array = ion_array_list
    A_idx_dict, B_idx_dict = idx_dict_list
    A_mass, B_mass = mass_list

    x_pixels, y_pixels, z_pixels = dim_list

    coinc_counter = 0

    len_autovariance_array = len(autovariance_array)

    # print(x_pixels, y_pixels, z_pixels)

    ## now thanks to the shot dictionaries we can easily iterate over all sets of events in a laser shot
    for A_shot, B_shot in zip(A_shot_array, B_shot_array):
        A_idx_i, A_idx_f = A_idx_dict[A_shot]
        B_idx_i, B_idx_f = B_idx_dict[B_shot]

        # For each 'A' ion in the 'shot'
        for i in prange(int(A_idx_i), int(A_idx_f)):
            ## n.b. now that we're using our own implementation of .dot,
            ## don't need to convert these arrays to contigious for speedup
            A_vec = A_ion_array[i, :3]
            A_mag = A_ion_array[i, 4]
            A_KE = p_au_KE_eV_fac * (A_mag) ** 2 / (2 * A_mass)

            # For each 'B' ion in the 'shot'
            for j in prange(int(B_idx_i), int(B_idx_f)):
                skip_coinc = 0
                if len_autovariance_array:
                    iter_index_array = np.array([i, j])
                    # check if any indices match
                    iter_index_set = set(iter_index_array)
                    # print(len(iter_index_set), len(iter_index_array))
                    if len(iter_index_set) != len(iter_index_array):
                        for autovariance_pair in autovariance_array:
                            if iter_index_array[autovariance_pair[0]] == iter_index_array[autovariance_pair[1]]:
                                skip_coinc = 1
                            else:
                                skip_coinc = 0
                if skip_coinc:
                    continue

                B_vec = B_ion_array[j, :3]
                B_mag = B_ion_array[j, 4]
                #B_KE = p_au_KE_eV_fac * (B_mag) ** 2 / (2 * B_mass)

                vec_list = [A_vec, B_vec]
                mag_list = [A_mag, B_mag]

                if use_filter_function:
                    filter_val = filter_function(vec_list, mag_list, mass_list)
                    if filter_val > filter_max:
                        skip_coinc = 1
                    if skip_coinc:
                        continue

                ## calculate the output from the set of 3 ion momenta vectors
                x_out_list, y_out_list, z_out_list = output_function(vec_list, mag_list, mass_list, bin_list, dim_list)

                for x_out, y_out, z_out in zip(x_out_list, y_out_list, z_out_list):
                    ## if in bounds add to array
                    if 0 <= x_out < x_pixels and 0 <= y_out < y_pixels and 0 <= z_out < z_pixels:
                        coinc_counter += 1
                        output_array[term_counter, x_out, y_out, z_out] += addval

    # print(coinc_counter)


@njit
def calc_Cabc(
    output_array,
    output_function,
    shot_array,
    ion_array_list,
    idx_dict_list,
    mass_list,
    dim_list,
    bin_list,
    shift_num_list,
    n_calc,
    n_shots,
    term_counter,
    autovariance_array,
    use_filter_function,
    filter_function,
    filter_max,
):
    """Find triple coincidences, transform into output frame and sum.
    Acts in place on the general output array. This function calculates
    all terms of the covariance expression, with appropriate values for the shot
    shifting index.

    :param ouput_array: array which stores output of covariance calculation
    :param output_function: the function which takes the coincidence and converts into the observables of interest
    :param shot_array: array of the unique shots in the dataset over which the covariance is calculated
    :param ion_array_list: data arrays of the three ions, [px,py,pz,shot,pmag] format
    :param idx_dict_list: list of the precalculated dictionaries of shot indices for each ion
    :param mass_list: mass of each ion in amu. Passed to the output function
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    :param shift_num_list: list of shot shift numbers applied to the ions beyond ion A
    :param n_calc: number of times the term will be recalculated.
    :param n_shots: number of shots covariance is calculated over.
    :param term_counter: index of the term in the epression to be calculated.
    :param autovariance_array: array of pairs of ions which are the same to exclude autovariance/trivial coincidence
    :param use_filter_function: 1 if filtering on some criteria of coincidences
    :param filter_function: function used for filtering, takes in (vec_list,mag_list,mass_list)
    :param filter_max: if output of filter_function is greater than filter_max, coincidence is ignored
    """

    ## as another speedup, we now precalculate the normalization needed to the term
    ## to save on doing extra operations on the entire array later
    addval = 1.0 / (n_shots * n_calc)

    ## do the shot shifting
    A_shot_array = shot_array
    B_shot_array = np.roll(shot_array, -shift_num_list[0])
    C_shot_array = np.roll(shot_array, -shift_num_list[1])

    A_ion_array, B_ion_array, C_ion_array = ion_array_list
    A_idx_dict, B_idx_dict, C_idx_dict = idx_dict_list
    A_mass, B_mass, C_mass = mass_list

    x_pixels, y_pixels, z_pixels = dim_list

    coinc_counter = 0

    len_autovariance_array = len(autovariance_array)

    ## now thanks to the shot dictionaries we can easily iterate over all sets of events in a laser shot
    for A_shot, B_shot, C_shot in zip(A_shot_array, B_shot_array, C_shot_array):
        A_idx_i, A_idx_f = A_idx_dict[A_shot]
        B_idx_i, B_idx_f = B_idx_dict[B_shot]
        C_idx_i, C_idx_f = C_idx_dict[C_shot]

        # For each 'A' ion in the 'shot'
        for i in prange(int(A_idx_i), int(A_idx_f)):
            ## n.b. now that we're using our own implementation of .dot,
            ## don't need to convert these arrays to contigious for speedup
            A_vec = A_ion_array[i, :3]
            A_mag = A_ion_array[i, 4]
            A_KE = p_au_KE_eV_fac * (A_mag) ** 2 / (2 * A_mass)

            # For each 'B' ion in the 'shot'
            for j in prange(int(B_idx_i), int(B_idx_f)):
                B_vec = B_ion_array[j, :3]
                B_mag = B_ion_array[j, 4]
                B_KE = p_au_KE_eV_fac * (B_mag) ** 2 / (2 * B_mass)

                for k in prange(int(C_idx_i), int(C_idx_f)):
                    # # print(len(autovariance_list))
                    skip_coinc = 0
                    if len_autovariance_array:
                        iter_index_array = np.array([i, j, k])
                        # check if any indices match
                        iter_index_set = set(iter_index_array)
                        # print(len(iter_index_set), len(iter_index_array))
                        if len(iter_index_set) != len(iter_index_array):
                            for autovariance_pair in autovariance_array:
                                if iter_index_array[autovariance_pair[0]] == iter_index_array[autovariance_pair[1]]:
                                    skip_coinc = 1
                                else:
                                    skip_coinc = 0
                    if skip_coinc:
                        continue

                    C_vec = C_ion_array[k, :3]
                    C_mag = C_ion_array[k, 4]
                    #C_KE = p_au_KE_eV_fac * (C_mag) ** 2 / (2 * C_mass)

                    vec_list = [A_vec, B_vec, C_vec]
                    mag_list = [A_mag, B_mag, C_mag]

                    if use_filter_function:
                        filter_val = filter_function(vec_list, mag_list, mass_list)
                        if filter_val > filter_max:
                            skip_coinc = 1
                        if skip_coinc:
                            continue

                    ## calculate the output from the set of 3 ion momenta vectors
                    x_out_list, y_out_list, z_out_list = output_function(
                        vec_list, mag_list, mass_list, bin_list, dim_list
                    )

                    for x_out, y_out, z_out in zip(x_out_list, y_out_list, z_out_list):
                        ## if in bounds add to array
                        if 0 <= x_out < x_pixels and 0 <= y_out < y_pixels and 0 <= z_out < z_pixels:
                            coinc_counter += 1
                            output_array[term_counter, x_out, y_out, z_out] += addval


@njit
def calc_Cabcd(
    output_array,
    output_function,
    shot_array,
    ion_array_list,
    idx_dict_list,
    mass_list,
    dim_list,
    bin_list,
    shift_num_list,
    n_calc,
    n_shots,
    term_counter,
    autovariance_array,
    use_filter_function,
    filter_function,
    filter_max,
):
    """Find quadruple coincidences, transform into output frame and sum.
    Acts in place on the general output array. This function calculates
    all terms of the covariance expression, with appropriate values for the shot
    shifting index.

    :param ouput_array: array which stores output of covariance calculation
    :param output_function: the function which takes the coincidence and converts into the observables of interest
    :param shot_array: array of the unique shots in the dataset over which the covariance is calculated
    :param ion_array_list: data arrays of the three ions, [px,py,pz,shot,pmag] format
    :param idx_dict_list: list of the precalculated dictionaries of shot indices for each ion
    :param mass_list: mass of each ion in amu. Passed to the output function
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    :param shift_num_list: list of shot shift numbers applied to the ions beyond ion A
    :param n_calc: number of times the term will be recalculated.
    :param n_shots: number of shots covariance is calculated over.
    :param term_counter: index of the term in the epression to be calculated.
    :param autovariance_array: array of pairs of ions which are the same to exclude autovariance/trivial coincidence
    :param use_filter_function: 1 if filtering on some criteria of coincidences
    :param filter_function: function used for filtering, takes in (vec_list,mag_list,mass_list)
    :param filter_max: if output of filter_function is greater than filter_max, coincidence is ignored
    """

    ## as another speedup, we now precalculate the normalization needed to the term
    ## to save on doing extra operations on the entire array later
    addval = 1.0 / (n_shots * n_calc)

    ## do the shot shifting
    A_shot_array = shot_array
    B_shot_array = np.roll(shot_array, -shift_num_list[0])
    C_shot_array = np.roll(shot_array, -shift_num_list[1])
    D_shot_array = np.roll(shot_array, -shift_num_list[2])

    A_ion_array, B_ion_array, C_ion_array, D_ion_array = ion_array_list
    A_idx_dict, B_idx_dict, C_idx_dict, D_idx_dict = idx_dict_list
    A_mass, B_mass, C_mass, D_mass = mass_list

    x_pixels, y_pixels, z_pixels = dim_list

    coinc_counter = 0

    len_autovariance_array = len(autovariance_array)

    ## now thanks to the shot dictionaries we can easily iterate over all sets of events in a laser shot
    for A_shot, B_shot, C_shot, D_shot in zip(A_shot_array, B_shot_array, C_shot_array, D_shot_array):
        A_idx_i, A_idx_f = A_idx_dict[A_shot]
        B_idx_i, B_idx_f = B_idx_dict[B_shot]
        C_idx_i, C_idx_f = C_idx_dict[C_shot]
        D_idx_i, D_idx_f = D_idx_dict[D_shot]

        # For each 'A' ion in the 'shot'
        for i in prange(int(A_idx_i), int(A_idx_f)):
            ## n.b. now that we're using our own implementation of .dot,
            ## don't need to convert these arrays to contigious for speedup
            A_vec = A_ion_array[i, :3]
            A_mag = A_ion_array[i, 4]
            #A_KE = p_au_KE_eV_fac * (A_mag) ** 2 / (2 * A_mass)

            # For each 'B' ion in the 'shot'
            for j in prange(int(B_idx_i), int(B_idx_f)):
                B_vec = B_ion_array[j, :3]
                B_mag = B_ion_array[j, 4]
                #B_KE = p_au_KE_eV_fac * (B_mag) ** 2 / (2 * B_mass)

                for k in prange(int(C_idx_i), int(C_idx_f)):
                    C_vec = C_ion_array[k, :3]
                    C_mag = C_ion_array[k, 4]
                    #C_KE = p_au_KE_eV_fac * (C_mag) ** 2 / (2 * C_mass)

                    for l in prange(int(D_idx_i), int(D_idx_f)):
                        # # print(len(autovariance_list))
                        skip_coinc = 0
                        if len_autovariance_array:
                            iter_index_array = np.array([i, j, k, l])
                            # check if any indices match
                            iter_index_set = set(iter_index_array)
                            # print(len(iter_index_set), len(iter_index_array))
                            if len(iter_index_set) != len(iter_index_array):
                                for autovariance_pair in autovariance_array:
                                    if iter_index_array[autovariance_pair[0]] == iter_index_array[autovariance_pair[1]]:
                                        skip_coinc = 1
                                    else:
                                        skip_coinc = 0

                        if skip_coinc:
                            continue

                        D_vec = D_ion_array[l, :3]
                        D_mag = D_ion_array[l, 4]
                        #D_KE = p_au_KE_eV_fac * (D_mag) ** 2 / (2 * D_mass)

                        vec_list = [A_vec, B_vec, C_vec, D_vec]
                        mag_list = [A_mag, B_mag, C_mag, D_mag]

                        if use_filter_function:
                            filter_val = filter_function(vec_list, mag_list, mass_list)
                            if filter_val > filter_max:
                                skip_coinc = 1
                            if skip_coinc:
                                continue

                        ## calculate the output from the set of 3 ion momenta vectors
                        x_out_list, y_out_list, z_out_list = output_function(
                            vec_list, mag_list, mass_list, bin_list, dim_list
                        )

                        for x_out, y_out, z_out in zip(x_out_list, y_out_list, z_out_list):
                            ## if in bounds add to array
                            if 0 <= x_out < x_pixels and 0 <= y_out < y_pixels and 0 <= z_out < z_pixels:
                                coinc_counter += 1
                                output_array[term_counter, x_out, y_out, z_out] += addval

class Covariance:
    """Class for storing covariance outputs and performing covariance calculation

    :param ion_list: list of n ions to be used in the nfold covariance calculation
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    :param store_coincs: NOT USED CURENTLY
    :param update_dataset: NOT USED CURRENTLY
    :param verbose: print more details during the calculation, default=False
    :param remove_autovariance: If True, automatically removes autovariance contributions if two ions
        the same. Default=True
    :param custom_function: Custom function for formatting the covariance output. If False,
        a default function is used. Default=False
    :param filter_function: Custom function used for filtering coincidences. If False, the default
        (absolute magnitude of summed momenta calculation) is used
    :param filter_max: Max value of the output of the filter function for an acceptable coincidence.
        Default is np.inf to ignore filtering
    :param max_shot: Maximum shot used in the calculation. Default=None
    :param only_coincidence: If True, only calculate the leading term (the true coincidences). Default = False
    :param n_EiEj: number of times to calculate the <A><B> term in two-fold covariance
    :param n_EiEjEk: number of times to calculate the <A><B><C> term in three-fold covariance
    :param n_EijEk: number of times to calculate the <AB><C> type terms in three-fold covariance
    :param n_EiEjEkEl: number of times to calculate the <A><B><C><D> term in four-fold covariance
    :param n_EijEkl: number of times to calculate the <AB><CD> type terms in four-fold covariance
    :param n_EijkEl: number of times to calculate the <ABC><D> type terms in four-fold covariance
    :param n_EiEjEkl: number of times to calculate the <A><B><CD> type terms in four-fold covariance
    :param contingent: is the covariance calculation contingent? default=False
    :param contingent_filter: filter object used in the calculation if contingent. default=None

    """

    def __init__(
        self,
        dataset,
        ion_list,
        dim_list,
        bin_list,
        store_coincs=False,
        update_dataset=True,
        verbose=False,
        remove_autovariance=True,
        custom_function=False,
        max_shot=None,
        n_EiEj=1,
        n_EiEjEk=1,
        n_EijEk=1,
        n_EiEjEkEl=1,
        n_EijEkl=1,
        n_EijkEl=1,
        n_EiEjEkl=1,
        filter_function=False,
        filter_max=np.inf,
        only_coincidence=False,
        contingent=False,
        contingent_filter=None,
    ):
        self.dataset = dataset
        self.shot_array = self.dataset.shot_array
        self.n_shots = len(self.shot_array)
        self.ion_list = ion_list
        self.nfold = len(ion_list)
        self.dim_list = nb.typed.List(dim_list)
        self.bin_list = nb.typed.List(bin_list)
        self.store_coincs = store_coincs
        self.update_dataset = update_dataset
        self.verbose = verbose
        self.remove_autovariance = remove_autovariance
        self.custom_function = custom_function
        self.n_EiEj = n_EiEj
        self.n_EiEjEk = n_EiEjEk
        self.n_EijEk = n_EijEk
        self.n_EiEjEkEl = n_EiEjEkEl
        self.n_EijEkl = n_EijEkl
        self.n_EijkEl = n_EijkEl
        self.n_EiEjEkl = n_EiEjEkl
        self.max_shot = max_shot
        self.filter_function = filter_function
        self.filter_max = filter_max
        self.only_coincidence = only_coincidence
        self.contingent = contingent
        self.contingent_filter = contingent_filter

    def apply_contingent_filter(self):
        """To do contingent covariance, all we do is filter on the shot array used in the covariance calculation"""

        # Firstly, (if necessary) generate a datasetdataframe which contains just the first entry for each shot

        try:
            shot_df = self.dataset.shot_df
        except:
            self.dataset.generate_shot_df()
            shot_df = self.dataset.shot_df

        # Convert this dataframe into a dict

        self.contingent_filter_dict = dict(zip(shot_df.shot, shot_df[self.contingent_filter.filter_param]))

        new_shot_list = [
            index
            for index, value in self.contingent_filter_dict.items()
            if (value > self.contingent_filter.filter_min)
            & (value <= self.contingent_filter.filter_max)
        ]
        self.shot_array = np.array(new_shot_list)
        self.n_shots = len(self.shot_array)

        if self.verbose:
            print(
                f"After filtering {self.contingent_filter.filter_param} between {self.contingent_filter.filter_min} and {self.contingent_filter.filter_max}: {self.n_shots} shots"
            )

    def check(self):
        """Run various checks before moving onto covariance calculation"""

        # try:
        #     shot_array = self.dataset.shot_array
        # except:
        #     raise ValueError("Shot array not in dataset")

        if len(self.dim_list) != len(self.bin_list):
            raise ValueError("dim_list and bin_list must be same length")

        ndims = len(self.dim_list)
        if ndims > 4:
            raise ValueError("Output with greater than 4 dimensions is not supported")

    def setup_output(self):
        """Create the output array which stores results of covariance calculation"""
        pixels_x, pixels_y, pixels_z = self.dim_list
        ## determine how many ions to be correlated, and their identity
        if self.nfold > 4:
            raise ValueError("Correlations above 4fold are not implemented yet!")
        elif self.nfold == 1:
            raise ValueError("Need more than 1 ion to correlate!")
        elif self.nfold == 2:
            self.ion_A, self.ion_B = self.ion_list
            output_array = np.zeros((3, pixels_x, pixels_y, pixels_z), dtype=np.float64)
        elif self.nfold == 3:
            output_array = np.zeros((6, pixels_x, pixels_y, pixels_z), dtype=np.float64)
            self.ion_A, self.ion_B, self.ion_C = self.ion_list
        elif self.nfold == 4:
            output_array = np.zeros((16, pixels_x, pixels_y, pixels_z), dtype=np.float64)
            self.ion_A, self.ion_B, self.ion_C, self.ion_D = self.ion_list
        self.output_array = output_array

        if self.verbose:
            print(f"Correlating {self.nfold} ions")
            print(
                "Ions to correlate: "
                + " ,".join([f"{ion.label} " for ion in self.ion_list])
            )

    def setup_terms(self):
        """Setup the number of times each term is to be calculated, and the shot-shifting which must be used."""

        ## setup shift_val_list, term_name_list, ncalc_list

        if self.nfold == 2:
            self.shift_val_list = [[0], [1]]
            self.term_name_list = ["<AB>", "<A><B>"]
            self.ncalc_list = [1, 0] if self.only_coincidence else [1, self.n_EiEj]
        elif self.nfold == 3:
            self.shift_val_list = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]

            self.term_name_list = [
                "<ABC>",
                "<AB><C>",
                "<AC><B>",
                "<BC><A>",
                "<A><B><C>",
            ]
            self.ncalc_list = (
                [1, 0, 0, 0, 0]
                if self.only_coincidence
                else [
                    1,
                    self.n_EijEk,
                    self.n_EijEk,
                    self.n_EijEk,
                    self.n_EiEjEk,
                ]
            )
        elif self.nfold == 4:
            self.shift_val_list = [
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 2],
                [1, 0, 2],
                [1, 2, 0],
                [1, 1, 2],
                [1, 2, 1],
                [2, 1, 1],
                [1, 2, 3],
            ]

            self.term_name_list = [
                "<ABCD>",
                "<A><BCD>",
                "<B><ACD>",
                "<C><ABD>",
                "<D><ABC>",
                "<AB><CD>",
                "<AC><BD>",
                "<AD><BC>",
                "<AB><C><D>",
                "<AC><B><D>",
                "<AD><B><C>",
                "<BC><A><D>",
                "<BD><A><C>",
                "<CD><A><B>",
                "<A><B><C><D>",
            ]

            self.ncalc_list = (
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                if self.only_coincidence
                else [
                    1,
                    self.n_EijkEl,
                    self.n_EijkEl,
                    self.n_EijkEl,
                    self.n_EijkEl,
                    self.n_EijEkl,
                    self.n_EijEkl,
                    self.n_EijEkl,
                    self.n_EiEjEkl,
                    self.n_EiEjEkl,
                    self.n_EiEjEkl,
                    self.n_EiEjEkl,
                    self.n_EiEjEkl,
                    self.n_EiEjEkl,
                    self.n_EiEjEkEl,
                ]
            )

    def get_from_ions(self):
        """Get the idx_dict, ion_array and ion_mass from each ion used in calculation"""
        self.idx_dict_list = []
        self.ion_array_list = []
        self.ion_mass_list = []
        for ion in self.ion_list:
            try:
                self.idx_dict_list.append(ion.idx_dict)
                if self.verbose:
                    print("%s dict already present" % ion.label)
            except:
                if self.verbose:
                    print("%s dict not present, generating..." % ion.label)
                ion.get_idx_dict(self.dataset.shot_array)
                self.idx_dict_list.append(ion.idx_dict)
            self.ion_array_list.append(ion.data_array)
            self.ion_mass_list.append(ion.mass)

        self.idx_dict_list = nb.typed.List(self.idx_dict_list)
        self.ion_array_list = nb.typed.List(self.ion_array_list)
        self.ion_mass_list = nb.typed.List(self.ion_mass_list)

    def setup_autovariance_correction(self):
        """Identify potential autovariance issues to be corrected for."""
        autovariance_counter = 0
        self.autovariance_array = np.zeros((30, 2), dtype="int64")
        if self.remove_autovariance:
            ## identify identical ions
            for i, ion1 in enumerate(self.ion_list):
                for j, ion2 in enumerate(self.ion_list):
                    if ion1 == ion2 and i != j:
                        if self.verbose:
                            print("Found duplicate ion to correct for")
                        self.autovariance_array[autovariance_counter, :] = [i, j]
                        autovariance_counter += 1
        self.autovariance_array = self.autovariance_array[0:autovariance_counter, :]

    def setup_filter(self):
        """Assign the filter function used in the covariance calculation."""
        if self.filter_function:
            self.use_filter_function = 1
            if isinstance(self.filter_function, str) and self.filter_function == "psum":
                # If the string is "psum", assign the corresponding function
                self.filter_function = calc_psum_abs
            elif callable(self.filter_function):
                # If it's already a callable function, keep it as is
                pass
            else:
                # Handle cases where filter_function is not "psum" and not callable
                raise ValueError("Invalid filter function provided. Must be 'psum' or a callable function.")
        else:
            # No filter function was provided, use default
            self.use_filter_function = 0
            self.filter_function = calc_psum_abs

    def setup_output_function(self):
        """Assign the output function used in the covariance calculation."""
        if self.custom_function:
            ## see if string is key in dictionary first
            try:
                self.output_function = output_function_dict[self.custom_function]
            except:
                self.output_function = self.custom_function
        else:
            ## if no specified output, select default based on number of ions to correlated
            if len(self.ion_list) == 2:
                self.output_function = output_format_Newton_2fold
            elif len(self.ion_list) == 3:
                self.output_function = output_format_Newton_3fold
            elif len(self.ion_list) == 4:
                self.output_function = output_format_Newton_4fold
        if not callable(self.output_function):
            raise ValueError("Invalid output function")

    def clip_max_shot(self):
        """Cut data based on max_shot."""
        if self.max_shot:
            self.shot_array = self.shot_array[self.shot_array < self.max_shot]

    def compute_covariance_term(self):
        """Calculate the 2/3/4fold covariance fromthe individual terms."""
        if self.nfold == 2:
            self.output_array = compute_2fold_covariance_term_iterate(self.output_array, self.dim_list)
        elif self.nfold == 3:
            self.output_array = compute_3fold_covariance_term_iterate(self.output_array, self.dim_list)
        elif self.nfold == 4:
            self.output_array = compute_4fold_covariance_term_iterate(self.output_array, self.dim_list)

    def calc_2fold_coinc(self):
        """Call the numba function for twofold coincidences"""

        calc_Cab(
            self.output_array,
            self.output_function,
            self.shot_array,
            self.ion_array_list,
            self.idx_dict_list,
            self.ion_mass_list,
            self.dim_list,
            self.bin_list,
            self.use_shiftnum_array,
            self.ncalc,
            self.n_shots,
            self.term_counter,
            self.autovariance_array,
            self.use_filter_function,
            self.filter_function,
            self.filter_max,
        )

    def calc_3fold_coinc(self):
        """Call the numba function for twofold coincidences"""

        calc_Cabc(
            self.output_array,
            self.output_function,
            self.shot_array,
            self.ion_array_list,
            self.idx_dict_list,
            self.ion_mass_list,
            self.dim_list,
            self.bin_list,
            self.use_shiftnum_array,
            self.ncalc,
            self.n_shots,
            self.term_counter,
            self.autovariance_array,
            self.use_filter_function,
            self.filter_function,
            self.filter_max,
        )

    def calc_4fold_coinc(self):
        """Call the numba function for twofold coincidences"""

        calc_Cabcd(
            self.output_array,
            self.output_function,
            self.shot_array,
            self.ion_array_list,
            self.idx_dict_list,
            self.ion_mass_list,
            self.dim_list,
            self.bin_list,
            self.use_shiftnum_array,
            self.ncalc,
            self.n_shots,
            self.term_counter,
            self.autovariance_array,
            self.use_filter_function,
            self.filter_function,
            self.filter_max,
        )

    def check_data_array(self):
        """Check that each ion has a data_array, if not then create it."""
        for ion in self.ion_list:
            if not hasattr(ion, "data_array"):
                ion.dataframe_to_arr()

    def calc_covariance(self):
        """General function which calls all the necessary setup/checks before covariance calculation,
        and then proceeds with the general logic."""
        self.start_time = time.time()
        if self.contingent:
            self.apply_contingent_filter()
        self.check()
        self.setup_output()
        self.setup_output_function()
        self.setup_terms()
        self.setup_filter()
        self.check_data_array()
        self.get_from_ions()
        self.setup_autovariance_correction()
        self.clip_max_shot()

        ## calculate each term of covariance
        self.term_counter = 0

        self.calc_shiftnum_array = np.zeros(self.nfold - 1, dtype="int64")
        self.use_shiftnum_array = np.zeros(self.nfold - 1, dtype="int64")

        ## check that the ions have data_arrays, force them to be made if not

        for term_name, shift_vals, ncalc in zip(self.term_name_list, self.shift_val_list, self.ncalc_list):
            self.ncalc = ncalc
            if self.verbose:
                print(f"Calculating {term_name}")
                print(f"Calculating {ncalc} times")
                print(
                    f"Current time: {round(time.time() - self.start_time, 1)} s",
                    flush=True,
                )
            for _ in range(self.ncalc):
                for cs in range(self.nfold - 1):
                    shiftnum = int(np.random.randint(len(self.shot_array) - 1)) + 1
                    self.calc_shiftnum_array[cs] = shiftnum

                ### todo here, catch the edge case of repeated shift numbers
                for cs in range(self.nfold - 1):
                    shiftnum = shift_vals[cs]
                    self.use_shiftnum_array[cs] = 0 if shiftnum == 0 else self.calc_shiftnum_array[shiftnum - 1]
                if self.nfold == 2:
                    self.calc_2fold_coinc()
                elif self.nfold == 3:
                    self.calc_3fold_coinc()
                elif self.nfold == 4:
                    self.calc_4fold_coinc()
            self.term_counter += 1

        self.compute_covariance_term()

        if self.verbose:
            print(f"Return at {round(time.time() - self.start_time, 1)} seconds")

    def plot_2DHist(
        self,
        proj_list,
        param_dict=param_dict,
        term=-1,
        label_list=None,
        arrow=False,
        vfac=1,
        cmap="bwr",
        axis_centered_on_zero=True,
        colors_centered_on_zero=True,
        **kwargs
    ):
        """Plot 2D histogram of calculated covariance

        :param proj_list: list of axes to project over in the plotting
        :param param_dict: param_dict used to neaten the plot
        :param term: which term of output array to plot. Default is -1 (covariance)
        :param label_list: list of labels of each axes of output array. Default is ['p$_x$','p$_y$','p$_z$']
        :param arrow: If True, plot an arrow for the reference label
        :param vfac: Used to choose max for colourscale. Default=1
        :param cmap: Colormap for plotting. Default='bwr'
        :param axis_centered_on_zero: Are the axes symmetric about zero (as in Newton plot)?
            If False, axis start at zero. then Default=True
        :param colors_centered_on_zero: Do you want to plot with a colormap that is symmetric about zero?
            If False, then vmin=0. Default=True
        """
        if label_list is None:
            label_list = ["p$_x$", "p$_y$", "p$_z$"]
        self.label_list = label_list
        for i in proj_list:
            if i == 0:
                bin_x = self.bin_list[1]
                dim_x = self.dim_list[1]
                label_x = self.label_list[1]
                bin_y = self.bin_list[2]
                dim_y = self.dim_list[2]
                label_y = self.label_list[2]
            elif i == 1:
                bin_x = self.bin_list[0]
                dim_x = self.dim_list[0]
                label_x = self.label_list[0]
                bin_y = self.bin_list[2]
                dim_y = self.dim_list[2]
                label_y = self.label_list[2]
            elif i == 2:
                bin_x = self.bin_list[0]
                dim_x = self.dim_list[0]
                label_x = self.label_list[0]
                bin_y = self.bin_list[1]
                dim_y = self.dim_list[1]
                label_y = self.label_list[1]
            covar = self.output_array[term, :, :, :]
            term_im = np.sum(covar, axis=i)

            fig, ax = plt.subplots(figsize=(8, 8))
            if axis_centered_on_zero:
                extent = (
                    -(bin_x * dim_x) / 2,
                    (bin_x * dim_x) / 2,
                    -(bin_y * dim_y) / 2,
                    (bin_y * dim_y) / 2,
                )
            else:
                extent = (0, bin_x * dim_x, 0, bin_y * dim_y)

            vmax = np.max(term_im) / vfac
            vmin = -vmax if colors_centered_on_zero else 0
            ax.imshow(
                term_im.T,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
                extent=extent,
                interpolation="none",
                origin="lower",
                aspect="auto",
                **kwargs
            )

            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)

            if arrow:
                ref_label = self.ion_list[0].label
                ### add the reference arrow
                if i == 2:
                    ax.arrow(
                        0.5,
                        0.5,
                        0,
                        0.25,
                        head_width=0.04,
                        color="k",
                        transform=ax.transAxes,
                    )
                    ax.text(
                        0.5,
                        0.85,
                        ref_label,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=24,
                    )
                if i == 0:
                    ax.arrow(
                        0.5,
                        0.5,
                        0.25,
                        0,
                        head_width=0.04,
                        color="k",
                        transform=ax.transAxes,
                    )
                    ax.text(
                        0.85,
                        0.5,
                        ref_label,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=24,
                    )
                ### add the reference arrow

            title_string = f'({" ".join(str(ion.label) for ion in self.ion_list)})'
            ax.set_title(title_string, fontsize=28)
            neaten_plot(ax, param_dict)
            plt.show()


class ContCovarianceFilter:
    """Class for containing filters used in contingent calculation

    :param filter_param: the name (string) of the parameter being filtered on in the dataframes
    :param filter_min: minimum value this parameter can take
    :param filter_max: maximum value this parameter can take

    """

    def __init__(self, filter_param, filter_min, filter_max):
        self.filter_param = filter_param
        self.filter_min = filter_min
        self.filter_max = filter_max


class ContCovariance(Covariance):
    """
    Class for contingent covariance calculations.
    Contains information for filtering data on the contingent paramter (e.g. FEL pulse energy), and collects the individual covariance objects.

    :param filter_list: list of filters used to calculate the contingent covariance
    :param covariances: list of covariances which are summed to make the contingent
    """

    def __init__(self, cont_filter_list, covariance_outputs):
        self.cont_filter_list = cont_filter_list
        self.covariance_outputs = covariance_outputs

    def get_details(self, example_covariance):
        for k, v in vars(example_covariance).items():
            if k not in ["dataset", "contingent_filter"]:
                try:
                    setattr(self, k, v)
                except:
                    pass  # non-pickelable stuff wasn't needed

    def __iter__(self, *args, **kwargs):
        return self.covariance_outputs.__iter__(*args, **kwargs)

    def calc_cont_covar(self):
        """Sum up each indivdual covariance output to get the contingent output"""
        for i, output_array in enumerate(self.covariance_outputs):
            if i == 0:
                self.output_array = output_array
            else:
                self.output_array += output_array

    def add_covariance_outputs(self, covariance):
        """Append a covariance object to the collection"""
        self.covariance_outputs.append(covariance_output)
