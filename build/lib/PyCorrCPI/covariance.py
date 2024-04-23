from .imports import *
from .helpers_numba import *
from .data import *
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
p_au_fac = hbar/bohr_radius # factor for atomic units of momentum
p_au_KE_eV_fac = (p_au_fac)**2/(u*e) # factor for converting from atomic units of momentum to KE in eV
cm_to_hartree = 1 / 219474.6 
hartree_to_eV = 27.211396132
u_to_amu = 1 / 5.4857990943e-4
bohr_to_angstrom = 0.529177211


output_function_dict = {'beta_psum_KER':beta_psum_KER,
                        'output_format_TfAcAc':output_format_TfAcAc}

param_dict = {'spinewidth':3,
              'linewidth':2,
              'ticklength':6,
              'tickwidth':3,
              'ticklabelsize':16,
              'axislabelsize':20}


def neaten_plot(ax, param_dict):
    ax.tick_params(labelsize=param_dict['ticklabelsize'],length=param_dict['ticklength'],width=param_dict['tickwidth'])
    ax.xaxis.get_label().set_fontsize(param_dict['axislabelsize'])
    ax.yaxis.get_label().set_fontsize(param_dict['axislabelsize'])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(param_dict['spinewidth'])


def calc_covariance(dataset, ion_list, dim_list, bin_list,
                       store_coincs=False,
                       update_dataset=True,
                       verbose=False,
                       remove_autovariance=True,
                       custom_function=False,
                       n_EiEj=1,
                       n_EiEjEk=1, n_EijEk=1,
                       n_EiEjEkEl=1, n_EijEkl=1, n_EijkEl=1, n_EiEjEkl=1,
                       filter_function=False,
                       filter_max=np.inf,
                       max_shot=None,
                       only_coincidence=None):
    """Wrapper function for initializing an instance of the Covariance class
    """


    covariance_output = Covariance(dataset, ion_list, dim_list, bin_list,
                       store_coincs=store_coincs,
                       update_dataset=update_dataset,
                       verbose=verbose,
                       remove_autovariance=remove_autovariance,
                       custom_function=custom_function,
                       n_EiEj=n_EiEj,
                       n_EiEjEk=n_EiEjEk, n_EijEk=n_EijEk,
                       n_EiEjEkEl=n_EiEjEkEl, n_EijEkl=n_EijEkl, n_EijkEl=n_EijkEl, n_EiEjEkl=n_EiEjEkl,
                       filter_function=filter_function,
                       filter_max=filter_max,
                       max_shot=max_shot,
                       only_coincidence=only_coincidence)

    covariance_output.calc_covariance()
    return(covariance_output)



@njit()
def calculate_indexes(A_idx_dict,shot_array_A,shot_array,A_arr):
    """Finds the first and last index in an ion data array corresponding to each laser shot
    that will be used in the covariance calculation. Precalculating this dictionary speeds up
    the calculation, particularly for higher fold covariances and/or if recalculating terms
    multiple times. 
    !This function assumes that the data is sorted by ascending shot number!

    Parameters:
    A_idx_dict -- the (empty) dictionary which will be filled with (shot, initial_index, final_index)
    shot_array_A -- array of laser shots used in the covariance calculation, for which ion A is detected
    shot_array -- array of total laser shots used in the covariance calculation
    A_arr - the ion A data array ([])
    """
    
    last_index_A_old=0
    A_shots = A_arr[:,3]
    found=False
    for shot in shot_array:
        found, first_index_A = find_first(shot,A_shots[last_index_A_old:])
        if found:
            last_index_A = find_last(shot,A_shots[first_index_A+last_index_A_old:])
            abs_first_index_A = last_index_A_old+first_index_A
            abs_last_index_A = last_index_A_old+first_index_A+last_index_A
            dict_arr = np.asarray([abs_first_index_A,abs_last_index_A+1],dtype='float64')
            A_idx_dict[shot] = np.asarray([abs_first_index_A,abs_last_index_A+1],dtype='float64')
            last_index_A_old += first_index_A+last_index_A
        else:
            # No ion A in this shot
            A_idx_dict[shot]= np.asarray([0.,0.],dtype='float64')
            

@njit(parallel=True,fastmath=True)
def compute_2fold_covariance_term_iterate(output_arr, dim_list):
    """calculate 2fold covariance from its consistuent terms"""
    for x in prange(dim_list[0]):
        for y in prange(dim_list[1]):
            for z in prange(dim_list[2]):
                output_arr[-1,x,y,z] = output_arr[0,x,y,z] - output_arr[1,x,y,z]
    return(output_arr)

@njit(parallel=True,fastmath=True)
def compute_3fold_covariance_term_iterate(output_arr, dim_list):
    """calculate 3fold covariance from its consistuent terms"""
    for x in prange(dim_list[0]):
        for y in prange(dim_list[1]):
            for z in prange(dim_list[2]):
                output_arr[-1,x,y,z] = output_arr[0,x,y,z] - \
                output_arr[1,x,y,z] - output_arr[2,x,y,z] - output_arr[3,x,y,z] + \
                2*output_arr[4,x,y,z]
    return(output_arr)

@njit(parallel=True,fastmath=True)
def compute_4fold_covariance_term_iterate(output_arr, dim_list):
    """calculate 4fold covariance from its consistuent terms"""
    for x in prange(dim_list[0]):
        for y in prange(dim_list[1]):
            for z in prange(dim_list[2]):
                output_arr[-1,x,y,z] = output_arr[0,x,y,z] - \
                (output_arr[1,x,y,z]+output_arr[2,x,y,z]+output_arr[3,x,y,z]+output_arr[4,x,y,z]) - \
                (output_arr[5,x,y,z]+output_arr[6,x,y,z]+output_arr[7,x,y,z])+\
                2*(output_arr[8,x,y,z]+output_arr[9,x,y,z]+output_arr[10,x,y,z]+output_arr[11,x,y,z]+output_arr[12,x,y,z]+output_arr[13,x,y,z]) - \
                6*output_arr[14,x,y,z]
    return(output_arr)



@njit
def calc_Cab(output_array, output_function,
                    shot_array,
                    ion_array_list, 
                    idx_dict_list,
                    mass_list, 
                    dim_list, 
                    bin_list,
                    shift_num_list,
                    n_calc, n_shots, term_counter,
                    autovariance_array,
                    use_filter_function,
                    filter_function,
                    filter_max):
    
    """Find double coincidences, transform into output frame and sum.
    Acts in place on the general output array. This function calculates
    all terms of the covariance expression, with appropriate values for the shot
    shifting index.
    
    Keyword arguments: 
    -- ouput_array: the output array for the covariance calculation
    -- output_function: the function which takes the coincidence and converts into the observables of interest
    -- shot_array: array of the unique shots in the dataset over which the covariance is calculated
    -- ion_array_list, data arrays of the three ions, [px,py,pz,shot,pmag] format
    -- idx_dict_list: the precalculated dictionaries of shot indices for each ion
    -- mass_list: mass of each ion in amu. Passed to the output function
    -- dim_list: number of pixels in each dimension of covariance output
    -- bin_list: bin size in each dimension for outputting covariance histogram
    -- shift_num_list: list of shift numbers applied to the ions beyond ion A
    -- n_calc: number of times the term will be recalculated. needed for proper weighting/normalization
    -- n_shots: number of shots covariance is calculated over. needed for proper weighting/normalization
    -- term_counter: index of the term in the epression to be calculated.
    -- autovariance_array: array of pairs of ions which are the same to exclude autovariance/trivial coincidence
    -- use_filter_function: 1 if filtering on some criteria of coincidences
    -- filter_function: function used for filtering, takes in (vec_list,mag_list,mass_list)
    -- filter_max: if output of filter_function is greater than filter_max, coincidence is ignored
    """
    
    ## as another speedup, we now precalculate the normalization needed to the term
    ## to save on doing extra operations on the entire array later
    addval = 1./(n_shots*n_calc)
    
    ## do the shot shifting
    A_shot_array = shot_array
    B_shot_array = np.roll(shot_array,-shift_num_list[0])

    A_ion_array,B_ion_array=ion_array_list
    A_idx_dict,B_idx_dict = idx_dict_list
    A_mass,B_mass = mass_list

    x_pixels,y_pixels,z_pixels=dim_list

    coinc_counter=0

    len_autovariance_array=len(autovariance_array)

    ## now thanks to the shot dictionaries we can easily iterate over all sets of events in a laser shot
    for A_shot,B_shot in zip(A_shot_array,B_shot_array):
        
        A_idx_i, A_idx_f = A_idx_dict[A_shot]
        B_idx_i, B_idx_f = B_idx_dict[B_shot]

        # For each 'A' ion in the 'shot'
        for i in prange(int(A_idx_i), int(A_idx_f)):

            ## n.b. now that we're using our own implementation of .dot,
            ## don't need to convert these arrays to contigious for speedup
            A_vec = A_ion_array[i,:3]
            A_mag = A_ion_array[i, 4]
            A_KE = p_au_KE_eV_fac*(A_mag)**2/(2*A_mass)

            # For each 'B' ion in the 'shot'
            for j in prange(int(B_idx_i), int(B_idx_f)):
                skip_coinc=0
                if len_autovariance_array:
                    iter_index_array= np.array([i,j])
                    # check if any indices match
                    iter_index_set = set(iter_index_array)
                    # print(len(iter_index_set), len(iter_index_array))
                    if len(iter_index_set)!=len(iter_index_array):
                        for autovariance_pair in autovariance_array:
                            if iter_index_array[autovariance_pair[0]]==iter_index_array[autovariance_pair[1]]:
                                skip_coinc=1
                            else:
                                skip_coinc=0
                if skip_coinc:
                    continue

                B_vec = B_ion_array[j,:3]
                B_mag = B_ion_array[j, 4]
                B_KE = p_au_KE_eV_fac*(B_mag)**2/(2*B_mass)

                vec_list = [A_vec,B_vec]
                mag_list = [A_mag,B_mag]

                if use_filter_function:
                    filter_val = filter_function(vec_list,mag_list,mass_list)
                    if filter_val>filter_max:
                        skip_coinc=1
                    if skip_coinc:
                        continue


                ## calculate the output from the set of 3 ion momenta vectors
                x_out_list,y_out_list,z_out_list = output_function(vec_list,mag_list,mass_list,bin_list,dim_list)

                for x_out,y_out,z_out in zip(x_out_list,y_out_list,z_out_list):
                    ## if in bounds add to array
                    if 0 <= z_out < x_pixels:
                        if 0 <= y_out < y_pixels:
                            if 0 <= z_out < z_pixels:
                                coinc_counter+=1
                                output_array[term_counter,x_out, y_out, z_out] += addval

    # print(coinc_counter)

@njit
def calc_Cabc(output_array, output_function,
                    shot_array,
                    ion_array_list, 
                    idx_dict_list,
                    mass_list, 
                    dim_list, 
                    bin_list,
                    shift_num_list,
                    n_calc, n_shots, term_counter,
                    autovariance_array,
                    use_filter_function,
                    filter_function,
                    filter_max):
    
    """Find double coincidences, transform into output frame and sum.
    Acts in place on the general output array. This function calculates
    all terms of the covariance expression, with appropriate values for the shot
    shifting index.
    
    Keyword arguments: 
    -- ouput_array: the output array for the covariance calculation
    -- output_function: the function which takes the coincidence and converts into the observables of interest
    -- shot_array: array of the unique shots in the dataset over which the covariance is calculated
    -- ion_array_list, data arrays of the three ions, [px,py,pz,shot,pmag] format
    -- idx_dict_list: the precalculated dictionaries of shot indices for each ion
    -- mass_list: mass of each ion in amu. Passed to the output function
    -- dim_list: number of pixels in each dimension of covariance output
    -- bin_list: bin size in each dimension for outputting covariance histogram
    -- shift_num_list: list of shift numbers applied to the ions beyond ion A
    -- n_calc: number of times the term will be recalculated. needed for proper weighting/normalization
    -- n_shots: number of shots covariance is calculated over. needed for proper weighting/normalization
    -- term_counter: index of the term in the epression to be calculated.
    -- autovariance_array: array of pairs of ions which are the same to exclude autovariance/trivial coincidence
    -- filter_function: function used for filtering, takes in (vec_list,mag_list,mass_list)
    -- filter_max: if output of filter_function is greater than filter_max, coincidence is ignored
    """
    
    ## as another speedup, we now precalculate the normalization needed to the term
    ## to save on doing extra operations on the entire array later
    addval = 1./(n_shots*n_calc)
    
    ## do the shot shifting
    A_shot_array = shot_array
    B_shot_array = np.roll(shot_array,-shift_num_list[0])
    C_shot_array = np.roll(shot_array,-shift_num_list[1])

    A_ion_array,B_ion_array,C_ion_array=ion_array_list
    A_idx_dict,B_idx_dict,C_idx_dict = idx_dict_list
    A_mass,B_mass,C_mass = mass_list

    x_pixels,y_pixels,z_pixels=dim_list

    coinc_counter=0

    len_autovariance_array = len(autovariance_array)

    ## now thanks to the shot dictionaries we can easily iterate over all sets of events in a laser shot
    for A_shot,B_shot,C_shot in zip(A_shot_array,B_shot_array,C_shot_array):
        
        A_idx_i, A_idx_f = A_idx_dict[A_shot]
        B_idx_i, B_idx_f = B_idx_dict[B_shot]
        C_idx_i, C_idx_f = C_idx_dict[C_shot]

        # For each 'A' ion in the 'shot'
        for i in prange(int(A_idx_i), int(A_idx_f)):

            ## n.b. now that we're using our own implementation of .dot,
            ## don't need to convert these arrays to contigious for speedup
            A_vec = A_ion_array[i,:3]
            A_mag = A_ion_array[i, 4]
            A_KE = p_au_KE_eV_fac*(A_mag)**2/(2*A_mass)

            # For each 'B' ion in the 'shot'
            for j in prange(int(B_idx_i), int(B_idx_f)):

                B_vec = B_ion_array[j,:3]
                B_mag = B_ion_array[j, 4]
                B_KE = p_au_KE_eV_fac*(B_mag)**2/(2*B_mass)

                for k in prange(int(C_idx_i), int(C_idx_f)):
                    # # print(len(autovariance_list))
                    skip_coinc=0
                    if len_autovariance_array:
                        iter_index_array= np.array([i,j,k])
                        # check if any indices match
                        iter_index_set = set(iter_index_array)
                        # print(len(iter_index_set), len(iter_index_array))
                        if len(iter_index_set)!=len(iter_index_array):
                            for autovariance_pair in autovariance_array:
                                if iter_index_array[autovariance_pair[0]]==iter_index_array[autovariance_pair[1]]:
                                    skip_coinc=1
                                else:
                                    skip_coinc=0
                    if skip_coinc:
                        continue

                    C_vec = C_ion_array[k,:3]
                    C_mag = C_ion_array[k, 4]
                    C_KE = p_au_KE_eV_fac*(C_mag)**2/(2*C_mass)

                    vec_list = [A_vec,B_vec,C_vec]
                    mag_list = [A_mag,B_mag,C_mag]

                    if use_filter_function:
                        filter_val = filter_function(vec_list,mag_list,mass_list)
                        if filter_val>filter_max:
                            skip_coinc=1
                        if skip_coinc:
                            continue


                    ## calculate the output from the set of 3 ion momenta vectors
                    x_out_list,y_out_list,z_out_list = output_function(vec_list,mag_list,mass_list,bin_list,dim_list)

                    for x_out,y_out,z_out in zip(x_out_list,y_out_list,z_out_list):
                        ## if in bounds add to array
                        if 0 <= x_out < x_pixels:
                            if 0 <= y_out < y_pixels:
                                if 0 <= z_out < z_pixels:
                                    coinc_counter+=1
                                    output_array[term_counter,x_out, y_out, z_out] += addval

@njit
def calc_Cabcd(output_array, output_function,
                    shot_array,
                    ion_array_list, 
                    idx_dict_list,
                    mass_list, 
                    dim_list, 
                    bin_list,
                    shift_num_list,
                    n_calc, n_shots, term_counter,
                    autovariance_array,
                    use_filter_function,
                    filter_function,
                    filter_max):
    
    """Find double coincidences, transform into output frame and sum.
    Acts in place on the general output array. This function calculates
    all terms of the covariance expression, with appropriate values for the shot
    shifting index.
    
    Keyword arguments: 
    -- ouput_array: the output array for the covariance calculation
    -- output_function: the function which takes the coincidence and converts into the observables of interest
    -- shot_array: array of the unique shots in the dataset over which the covariance is calculated
    -- ion_array_list, data arrays of the three ions, [px,py,pz,shot,pmag] format
    -- idx_dict_list: the precalculated dictionaries of shot indices for each ion
    -- mass_list: mass of each ion in amu. Passed to the output function
    -- dim_list: number of pixels in each dimension of covariance output
    -- bin_list: bin size in each dimension for outputting covariance histogram
    -- shift_num_list: list of shift numbers applied to the ions beyond ion A
    -- n_calc: number of times the term will be recalculated. needed for proper weighting/normalization
    -- n_shots: number of shots covariance is calculated over. needed for proper weighting/normalization
    -- term_counter: index of the term in the epression to be calculated.
    -- autovariance_array: array of pairs of ions which are the same to exclude autovariance/trivial coincidence
    -- filter_function: function used for filtering, takes in (vec_list,mag_list,mass_list)
    -- filter_max: if output of filter_function is greater than filter_max, coincidence is ignored
    """
    
    ## as another speedup, we now precalculate the normalization needed to the term
    ## to save on doing extra operations on the entire array later
    addval = 1./(n_shots*n_calc)
    
    ## do the shot shifting
    A_shot_array = shot_array
    B_shot_array = np.roll(shot_array,-shift_num_list[0])
    C_shot_array = np.roll(shot_array,-shift_num_list[1])
    D_shot_array = np.roll(shot_array,-shift_num_list[2])

    A_ion_array,B_ion_array,C_ion_array,D_ion_array=ion_array_list
    A_idx_dict,B_idx_dict,C_idx_dict,D_idx_dict = idx_dict_list
    A_mass,B_mass,C_mass,D_mass = mass_list

    x_pixels,y_pixels,z_pixels=dim_list

    coinc_counter=0

    len_autovariance_array = len(autovariance_array)

    ## now thanks to the shot dictionaries we can easily iterate over all sets of events in a laser shot
    for A_shot,B_shot,C_shot,D_shot in zip(A_shot_array,B_shot_array,C_shot_array,D_shot_array):
        
        A_idx_i, A_idx_f = A_idx_dict[A_shot]
        B_idx_i, B_idx_f = B_idx_dict[B_shot]
        C_idx_i, C_idx_f = C_idx_dict[C_shot]
        D_idx_i, D_idx_f = D_idx_dict[D_shot]

        # For each 'A' ion in the 'shot'
        for i in prange(int(A_idx_i), int(A_idx_f)):

            ## n.b. now that we're using our own implementation of .dot,
            ## don't need to convert these arrays to contigious for speedup
            A_vec = A_ion_array[i,:3]
            A_mag = A_ion_array[i, 4]
            A_KE = p_au_KE_eV_fac*(A_mag)**2/(2*A_mass)

            # For each 'B' ion in the 'shot'
            for j in prange(int(B_idx_i), int(B_idx_f)):

                B_vec = B_ion_array[j,:3]
                B_mag = B_ion_array[j, 4]
                B_KE = p_au_KE_eV_fac*(B_mag)**2/(2*B_mass)

                for k in prange(int(C_idx_i), int(C_idx_f)):

                    C_vec = C_ion_array[k,:3]
                    C_mag = C_ion_array[k, 4]
                    C_KE = p_au_KE_eV_fac*(C_mag)**2/(2*C_mass)

                    for l in prange(int(D_idx_i), int(D_idx_f)):
                        # # print(len(autovariance_list))
                        skip_coinc=0
                        if len_autovariance_array:
                            iter_index_array= np.array([i,j,k,l])
                            # check if any indices match
                            iter_index_set = set(iter_index_array)
                            # print(len(iter_index_set), len(iter_index_array))
                            if len(iter_index_set)!=len(iter_index_array):
                                for autovariance_pair in autovariance_array:
                                    if iter_index_array[autovariance_pair[0]]==iter_index_array[autovariance_pair[1]]:
                                        skip_coinc=1
                                    else:
                                        skip_coinc=0


                        if skip_coinc:
                            continue

                        D_vec = D_ion_array[l,:3]
                        D_mag = D_ion_array[l, 4]
                        D_KE = p_au_KE_eV_fac*(D_mag)**2/(2*D_mass)

                        vec_list = [A_vec,B_vec,C_vec,D_vec]
                        mag_list = [A_mag,B_mag,C_mag,D_mag]

                        if use_filter_function:
                            filter_val = filter_function(vec_list,mag_list,mass_list)
                            if filter_val>filter_max:
                                skip_coinc=1
                            if skip_coinc:
                                continue

                        ## calculate the output from the set of 3 ion momenta vectors
                        x_out_list,y_out_list,z_out_list = output_function(vec_list,mag_list,mass_list,bin_list,dim_list)

                        for x_out,y_out,z_out in zip(x_out_list,y_out_list,z_out_list):
                            ## if in bounds add to array
                            if 0 <= x_out < x_pixels:
                                if 0 <= y_out < y_pixels:
                                    if 0 <= z_out < z_pixels:
                                        coinc_counter+=1
                                        output_array[term_counter,x_out, y_out, z_out] += addval


class Covariance:
    """Obect for storing covariance outputs and performing covariance calculation"""
    def __init__(self, dataset, ion_list, dim_list, bin_list,
                       store_coincs=False,
                       update_dataset=True,
                       verbose=False,
                       remove_autovariance=True,
                       custom_function=False,
                       max_shot=None,
                       n_EiEj=1,
                       n_EiEjEk=1, n_EijEk=1,
                       n_EiEjEkEl=1, n_EijEkl=1, n_EijkEl=1, n_EiEjEkl=1,
                       filter_function=False,
                       filter_max=np.inf,
                       only_coincidence=False):
        
        self.dataset=dataset
        self.shot_array=self.dataset.shot_array
        self.n_shots = len(self.shot_array)
        self.ion_list=ion_list
        self.nfold=len(ion_list)
        self.dim_list=nb.typed.List(dim_list)
        self.bin_list=nb.typed.List(bin_list)
        self.store_coincs=store_coincs
        self.update_dataset=update_dataset
        self.verbose=verbose
        self.remove_autovariance=remove_autovariance
        self.custom_function=custom_function
        self.n_EiEj=n_EiEj
        self.n_EiEjEk=n_EiEjEk
        self.n_EijEk=n_EijEk
        self.n_EiEjEkEl=n_EiEjEkEl
        self.n_EijEkl=n_EijEkl
        self.n_EijkEl=n_EijkEl
        self.n_EiEjEkl=n_EiEjEkl
        self.max_shot=max_shot
        self.filter_function=filter_function
        self.filter_max=filter_max
        self.only_coincidence=only_coincidence


    def check(self):
        """Run various checks before moving onto covariance calculation"""

        try:
            shot_array = self.dataset.shot_array
        except:
            raise ValueError('Shot array not in dataset')
        
        if len(self.dim_list)!=len(self.bin_list):
            raise ValueError('dim_list and bin_list must be same length')
            
        ndims = len(self.dim_list)
        if ndims>4:
            raise ValueError('Output with greater than 4 dimensions is not supported')

    def setup_output(self):
        pixels_x,pixels_y,pixels_z = self.dim_list
        ## determine how many ions to be correlated, and their identity
        if self.nfold>4:
            raise ValueError('Correlations above 4fold are not implemented yet!')
        elif self.nfold==1:
            raise ValueError('Need more than 1 ion to correlate!')
        elif self.nfold==2:
            self.ion_A,self.ion_B=self.ion_list
            output_array = np.zeros((3,pixels_x,pixels_y,pixels_z), dtype=np.float64)
        elif self.nfold==3:
            output_array = np.zeros((6,pixels_x,pixels_y,pixels_z), dtype=np.float64)
            self.ion_A,self.ion_B,self.ion_C=self.ion_list
        elif self.nfold==4:
            output_array = np.zeros((16,pixels_x,pixels_y,pixels_z), dtype=np.float64)
            self.ion_A,self.ion_B,self.ion_C,self.ion_D=self.ion_list
        self.output_array=output_array

        if self.verbose:
            print("Correlating %s ions" % self.nfold)
            print("Ions to correlate: " + " ,".join(['%s ' % (ion.label,) for ion in self.ion_list]))
    
    
    def setup_terms(self):

        ## setup shift_val_list, term_name_list, ncalc_list
        
        if self.nfold==2:
            self.shift_val_list = [[0],
                      [1]]
            self.term_name_list = ['<AB>', '<AB>']
            if self.only_coincidence:
                self.ncalc_list = [1,0]
            else:
                self.ncalc_list = [1,self.n_EiEj]
        elif self.nfold==3:
            self.shift_val_list = [[0,0],
                              [0,1],
                              [1,0],
                              [1,1],
                              [1,2]]

            self.term_name_list = ['<ABC>',
                              '<AB><C>',
                             '<AC><B>',
                             '<BC><A>',
                             '<A><B><C>']
            if self.only_coincidence:
                self.ncalc_list = [1,0,0,0,0]
            else:
                self.ncalc_list = [1, self.n_EijEk, self.n_EijEk, self.n_EijEk, self.n_EiEjEk]
            
        elif self.nfold==4:     
            
            self.shift_val_list = [[0,0,0],
                              [1,1,1],
                              [1,0,0],
                              [0,1,0],
                              [0,0,1],
                              [0,1,1],
                              [1,0,1],
                              [1,1,0],
                              [0,1,2],
                              [1,0,2],
                              [1,2,0],
                              [1,1,2],
                              [1,2,1],
                              [2,1,1],
                              [1,2,3]]

            self.term_name_list = ['<ABCD>', 
                              '<A><BCD>', '<B><ACD>', '<C><ABD>', '<D><ABC>',
                             '<AB><CD>', '<AC><BD>','<AD><BC>',
                             '<AB><C><D>', '<AC><B><D>', '<AD><B><C>', '<BC><A><D>', '<BD><A><C>', '<CD><A><B>',
                             '<A><B><C><D>']
            
            if self.only_coincidence:
                self.ncalc_list = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            else:
                self.ncalc_list = [1,
                             self.n_EijkEl,self.n_EijkEl,self.n_EijkEl,self.n_EijkEl,
                             self.n_EijEkl,self.n_EijEkl,self.n_EijEkl,
                             self.n_EiEjEkl,self.n_EiEjEkl,self.n_EiEjEkl,self.n_EiEjEkl,self.n_EiEjEkl,self.n_EiEjEkl,
                             self.n_EiEjEkEl]

    def get_from_ions(self):
        self.idx_dict_list=[]
        self.ion_array_list=[]
        self.ion_mass_list=[]
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
        ## identify any autovariance issues
        autovariance_counter=0
        self.autovariance_array = np.zeros((30,2), dtype='int64')
        if self.remove_autovariance:
            ## identify identical ions
            for i, ion1 in enumerate(self.ion_list):
                for j, ion2 in enumerate(self.ion_list):
                    if i!=j:
                        if ion1==ion2:
                            if self.verbose:
                                print('Found duplicate ion to correct for')
                            self.autovariance_array[autovariance_counter,:] = [i,j]
                            autovariance_counter+=1
        self.autovariance_array = self.autovariance_array[0:autovariance_counter,:]
        print(self.autovariance_array)

    def setup_filter(self):

        if self.filter_function:
            self.use_filter_function=1
            if self.filter_function=='psum':
                self.filter_function=calc_psum_abs
        else:
            self.use_filter_function=0
            self.filter_function=calc_psum_abs

    def setup_output_function(self):
        if self.custom_function:
            ## see if string is key in dictionary first
            try:
                self.output_function=output_function_dict[self.custom_function]
            except:
                self.output_function=self.custom_function
        else:
            ## if no specified output, select default based on number of ions to correlated
            if len(self.ion_list)==2:
                self.output_function=output_format_Newton_2fold
            elif len(self.ion_list)==3:
                self.output_function=output_format_Newton_3fold
            elif len(self.ion_list)==4:
                self.output_function=output_format_Newton_4fold
        if not callable(self.output_function):
            raise ValueError("Invalid output function")


    def clip_max_shot(self):
        if self.max_shot:
            self.shot_array=self.shot_array[self.shot_array<self.max_shot]

        
    def compute_covariance_term(self):
        if self.nfold==2:
            self.output_array=compute_2fold_covariance_term_iterate(self.output_array, self.dim_list)
        elif self.nfold==3:
            self.output_array=compute_3fold_covariance_term_iterate(self.output_array, self.dim_list)
        elif self.nfold==4:
            self.output_array=compute_4fold_covariance_term_iterate(self.output_array, self.dim_list)

    def calc_2fold_coinc(self):
        ### call numba fn

        calc_Cab(self.output_array, self.output_function,
                self.shot_array,self.ion_array_list,self.idx_dict_list,
                self.ion_mass_list, 
                self.dim_list, self.bin_list,self.use_shiftnum_array,
                self.ncalc, self.n_shots, self.term_counter,
                self.autovariance_array, 
                self.use_filter_function, self.filter_function, self.filter_max)

    def calc_3fold_coinc(self):
        ### call numba fn

        calc_Cabc(self.output_array, self.output_function,
                self.shot_array,self.ion_array_list,self.idx_dict_list,
                self.ion_mass_list, 
                self.dim_list, self.bin_list,self.use_shiftnum_array,
                self.ncalc, self.n_shots, self.term_counter,
                self.autovariance_array, 
                self.use_filter_function,self.filter_function, self.filter_max)

    def calc_4fold_coinc(self):
        ### call numba fn

        calc_Cabcd(self.output_array, self.output_function,
                self.shot_array,self.ion_array_list,self.idx_dict_list,
                self.ion_mass_list, 
                self.dim_list, self.bin_list,self.use_shiftnum_array,
                self.ncalc, self.n_shots, self.term_counter,
                self.autovariance_array, 
                self.use_filter_function, self.filter_function, self.filter_max)


    def calc_covariance(self):

        self.start_time = time.time()
        self.check()
        self.setup_output()
        self.setup_output_function()
        self.setup_terms()
        self.setup_filter()
        self.get_from_ions()
        self.setup_autovariance_correction()
        self.clip_max_shot()


        ## calculate each term of covariance
        self.term_counter = 0
        
        self.calc_shiftnum_array = np.zeros(self.nfold-1, dtype='int64')
        self.use_shiftnum_array = np.zeros(self.nfold-1,dtype='int64')

        for term_name, shift_vals, ncalc in zip(self.term_name_list,self.shift_val_list, self.ncalc_list):
            self.ncalc=ncalc
            if self.verbose:
                print('Calculating ' + term_name)
                print('Calculating %s times' % ncalc)  
                print('Current time: %s s' % round((time.time()-self.start_time),1), flush=True)
            for n in range(self.ncalc):
                for cs in range(self.nfold-1):
                    shiftnum=int(np.random.randint(len(self.shot_array)-1))+1
                    self.calc_shiftnum_array[cs]=shiftnum
                
                ### todo here, catch the edge case of repeated shift numbers
                for cs in range(self.nfold-1):
                    shiftnum = shift_vals[cs]
                    if shiftnum==0:
                        self.use_shiftnum_array[cs]=0
                    else:
                        self.use_shiftnum_array[cs]=self.calc_shiftnum_array[shiftnum-1]  
                if self.nfold==2:                    
                    self.calc_2fold_coinc()
                elif self.nfold==3:
                    self.calc_3fold_coinc()
                elif self.nfold==4:
                    self.calc_4fold_coinc()
            self.term_counter+=1

        self.compute_covariance_term()

        if self.verbose:
            print("Return at %s seconds" % round((time.time() - self.start_time), 1))


    def plot_2DHist(self, proj_list, param_dict=param_dict, term=-1, 
                label_list = ['p$_x$','p$_y$','p$_z$'],arrow=False,
                vfac=1,cmap='bwr', axis_centered_on_zero=True):
        self.label_list = label_list
        for i in proj_list:
            if i==0:
                bin_x = self.bin_list[1]
                dim_x = self.dim_list[1]
                label_x = self.label_list[1]
                bin_y = self.bin_list[2]
                dim_y = self.dim_list[2]
                label_y = self.label_list[2]
            if i==1:
                bin_x = self.bin_list[0]
                dim_x = self.dim_list[0]
                label_x = self.label_list[0]
                bin_y = self.bin_list[2]
                dim_y = self.dim_list[2]
                label_y = self.label_list[2]
            if i==2:
                bin_x = self.bin_list[0]
                dim_x = self.dim_list[0]
                label_x = self.label_list[0]
                bin_y = self.bin_list[1]
                dim_y = self.dim_list[1]
                label_y = self.label_list[1]
            covar = self.output_array[term,:,:,:]
            term_im = np.sum(covar,axis=i)
            
            fig,ax=plt.subplots(figsize=(8,8))
            if axis_centered_on_zero:
                extent = (-(bin_x*dim_x)/2, (bin_x*dim_x)/2, -(bin_y*dim_y)/2, (bin_y*dim_y)/2)
            else:
                extent = (0, bin_x*dim_x, 0, bin_y*dim_y)
            ax.imshow(term_im.T, cmap=cmap, vmax=np.max(term_im)/vfac, vmin=-np.max(term_im)/vfac,
                     extent = extent, interpolation='none',
                     origin='lower', aspect='auto')
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)

            if arrow:
                ref_label=self.ion_list[0].label
                ### add the reference arrow
                if i==2:
                    ax.arrow(0.5, 0.5, 0, 0.25, head_width=0.04,color='k', transform=ax.transAxes)
                    ax.text(0.5,0.85,ref_label, ha='center', va='center',transform=ax.transAxes, fontsize=24)
                if i==0:
                    ax.arrow(0.5, 0.5, 0.25, 0, head_width=0.04,color='k', transform=ax.transAxes)
                    ax.text(0.85,0.5,ref_label, ha='center', va='center',transform=ax.transAxes, fontsize=24)
                ### add the reference arrow

            title_string = f'({" ".join(str(ion.label) for ion in self.ion_list)})'
            ax.set_title(title_string,fontsize=28)
            neaten_plot(ax,param_dict)
            plt.show()


