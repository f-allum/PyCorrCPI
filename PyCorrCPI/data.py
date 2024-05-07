from .imports import *
from .helpers_numba import *
from .covariance import *




class Ion:
    """Class used to define the parameters to extract data from different ions from a dataset

    :param label: (str) name used to label the ion
    :param filter_i: lower bound of the filter used to select ion data from Dataset
    :param filter_f: upper bound of the filter used to select ion data from Dataset
    :param dataset: Dataset object associated with the ion
    :param filter_param: column of the Dataset's dataframe to filter on to select ion's data
    :param center: ion's (x,y) center
    :param center_t: ion's t center
    :param mass: ion's mass
    :param charge: ion's charge
    :param shot_array_method: method for determining and ion's array of shots for the 
        covariance calculation. Default is 'range'. Options: 'range', 'unique'. 'Range'
        should work unless the dataset is missing certain laser shots between the first and last.
    :param use_for_mass_cal: If True, this ion can be used to do a m/z calibration
        in an IonCollection. Default = True
    """
    def __init__(self, label, filter_i, filter_f, dataset=None, filter_param='t',
        center=None, center_t=None, mass=None, charge=None, shot_array_method='range',
        use_for_mass_calib=True):
        self.label = label
        self.filter_i = filter_i
        self.filter_f = filter_f
        self.filter_param=filter_param
        self.shot_array_method = shot_array_method
        self.mass=mass
        self.use_for_mass_calib=use_for_mass_calib
        self.charge=charge

        try:
            self.mz = self.mass/self.charge
        except:
            print("Unable to determine ion mass-to-charge ratio")

        self.center=center

        if center_t:
            self.center_t = center_t

        # else:
        #     self.center_t = (self.Ti+self.Tf)/2

        if dataset:
            self.assign_dataset(dataset)


            
    def assign_dataset(self, dataset):
        """Assign Dataset object to the ion"""
        self.grab_data(dataset)
        self.get_shot_array()

    def print_details(self):
        """Print out attributes of the object excluding large arrays/dataframes"""
        for key,value in self.__dict__.items():
            if key not in ['data_df','data_array', 'shot_array']:
                print(f"'{key}':{value}")

    def grab_data(self,dataset):
        """Gets data corresponding to ion from dataset based on the inputted filter"""
        try:
            self.data_df = dataset.sep_by_custom(self.filter_i,self.filter_f, self.filter_param)
        except:
            raise Exception("filter_param is not found in dataframe!")

    def dataframe_to_arr(self):
        """Converts necessary dataframe columns for covariance calculation to an array"""
        try:
            self.data_array = self.data_df[["px", "py", "pz", "shot", "pmag"]].to_numpy()
        except:
            self.data_array = self.data_df[["px_AU", "py_AU", "pz_AU", "shot", "pmag_AU"]].to_numpy()

    def get_shot_array(self):
        """Find array of shots in dataset which contain this ion"""
        if self.shot_array_method=='range':
            # print(np.min(self.data_df.shot))
            self.shot_array = np.arange(np.min(self.data_df.shot), np.max(self.data_df.shot))
        elif self.shot_array_method=='unique':
            self.shot_array = np.array(np.unique(self.data_df.shot))
        else:
            raise Exception("Invalid shot_array_method inputted!")

    def get_idx_dict(self, shot_array_total):
        """Create dictionary of indices of rows in dataset corresponding to this ion. Needed for covariance"""
        idx_dict = Dict.empty(
                key_type=float_single,
                value_type=float_array)

        calculate_indexes(idx_dict,self.shot_array,shot_array_total,self.data_array)
        self.idx_dict=idx_dict

    def calc_mz_cal(self, coeffs_tof_sqmz):
        """Calculated expected m/z of central ToF from calibration toefficients"""
        self.cal_mz = (self.center_t*coeffs_tof_sqmz[0] + coeffs_tof_sqmz[1])**2

    def calc_t_absolute(self, t0):
        """Calculate absolute t by subtracting t0 (start of mass spectrum)"""
        self.t0=t0
        self.data_df['t_absolute']=self.data_df['t']-self.t0

    def calc_t_centered(self):
        """Calculate t relative to t_center. Used in converting to 3D momentum."""
        self.data_df['t_centered']=self.data_df['t']-self.center_t

        
    def manual_center(self):
        """Manually center data in x,y using a user-given center."""
        if self.center:
            self.data_df['xcorr_manual'] = self.data_df['x']-center[0]
            self.data_df['ycorr_manual'] = self.data_df['y']-center[1]
        else:
            print("Can't manually center - center not given")

    def apply_jet_correction(self, jet_offset, jet_velocity):
        """Center data using jet offsets and velocity in x and y. If this isn't working, it may be a sign error"""
        self.jet_offset=jet_offset
        self.jet_velocity=jet_velocity
        self.data_df['xcorr_jet'] = (self.data_df['x']-jet_offset[0])-(self.data_df['t_absolute']*jet_velocity[0])
        self.data_df['ycorr_jet'] = (self.data_df['y']-jet_offset[1])-(self.data_df['t_absolute']*jet_velocity[1])

    def adjust_jet_correction(self, jet_adjust):
        """Take centers produced from the jet correction and further adjust these manually"""
        self.jet_adjust = jet_adjust
        self.data_df['xcorr_jet_adjust'] = self.data_df['xcorr_jet']+jet_adjust[0]
        self.data_df['ycorr_jet_adjust'] = self.data_df['ycorr_jet']+jet_adjust[1]



    def correct_centers(self, method=None):
        """Re-center data in x and y.

        :param method: Options: ['jet', 'manual','jet_adjust']
            'jet' - center by subtracting jet offsets/velocities obtained by previous calibration
            'manual' - center by subtracting manual user defined centers
            'jet_adjust' - center by subtracting jet offsets/velicocities, and then further adjust
            by user-defined parameters
            """
        if method in ['jet', 'manual', 'jet_adjust']:
            if method=='jet':
                self.data_df['x_centered'] = self.data_df['xcorr_jet']
                self.data_df['y_centered'] = self.data_df['ycorr_jet']
            elif method=='manual':
                self.data_df['x_centered'] = self.data_df['xcorr_manual']
                self.data_df['y_centered'] = self.data_df['ycorr_manual']
            elif method=='jet_adjust':
                self.data_df['x_centered'] = self.data_df['xcorr_jet_adjust']
                self.data_df['y_centered'] = self.data_df['ycorr_jet_adjust']
        else:
            print("Using default centers")
            self.data_df['x_centered'] = self.data_df['x']
            self.data_df['y_centered'] = self.data_df['y']

        self.centered=True


    def apply_momentum_calibration(self, C_xy, C_z, C_total=None, center_method='manual'):
        """Convert (centered data to 3D momenta). For now this assumes that images are round
        (i.e. that scaling parameter in x and y are the same). This function automatically
        converts the ion's dataframe to array for future covariance calculation.

        :param C_xy: linear scaling factor from x/t or y/t to velocity
        :param C_z: linear scaling factor from (t-tcenter)*charge/mass to velocity
        :param center_method: method used to adjust x/y centers prior to calibration. See
            correct_centers
        """
        if C_xy:
            self.C_xy = C_xy
        if C_z:
            self.C_z = C_z
        
            
        self.correct_centers(method=center_method)
  
        
        
        self.data_df['t_absolute'] = self.data_df['t']-self.t0
        self.data_df['t_relative'] = self.data_df[ 't']-self.center_t
        
        self.data_df['vx'] = C_xy*(self.data_df['x_centered']/self.data_df['t_absolute'])
        self.data_df['vy'] = C_xy*(self.data_df['y_centered']/self.data_df['t_absolute'])
        self.data_df['vz'] = (C_z*self.charge*(self.data_df['t_centered']))/self.mass

        if C_total:
            self.data_df['vx']*=C_total
            self.data_df['vy']*=C_total
            self.data_df['vz']*=C_total

        self.data_df['px'] = self.data_df['vx'] * self.mass
        self.data_df['py'] = self.data_df['vy'] * self.mass
        self.data_df['pz'] = self.data_df['vz'] * self.mass
        
        self.data_df['pmag'] = np.sqrt((self.data_df['px']**2+self.data_df['py']**2+self.data_df['pz']**2))
        self.data_df['vmag'] = np.sqrt((self.data_df['vx']**2+self.data_df['vy']**2+self.data_df['vz']**2))
        
        self.cal_mom=True
        self.dataframe_to_arr()
        



class IonCollection:
    """Class for groups of Ions, which can be used for mass calibrations, jet calibrations etc.
    Can iterate over the Ion objects in an IonCollection

    :param filter_param: filter_param used for defining Ions in the group
    :param allow_auto_mass_charge: not implemented currently
    :param shot_array_method: shot_array_method used for defining Ions in the group
    """
    def __init__(self, filter_param=None, allow_auto_mass_charge=False, shot_array_method=None):
        self.data = list()
        self.filter_param = filter_param
        self.allow_auto_mass_charge = allow_auto_mass_charge
        self.shot_array_method = shot_array_method

        optional_kwargs = dict()
        if self.filter_param:
            optional_kwargs["filter_param"] = self.filter_param
        if self.allow_auto_mass_charge:
            optional_kwargs["allow_auto_mass_charge"] = self.allow_auto_mass_charge
        if self.shot_array_method:
            optional_kwargs["shot_array_method"] = self.shot_array_method

        self.ion_class = partial(Ion, **optional_kwargs)

    def mz_calibration(self):
        """Autoamtically perform m/z calibration based on the Ions in the collection which have specified
        center_t, mass and charge. 

        Output first-order polynomial coefficients are stored as self.coeffs_sqmz_tof and self.coeffs_tof_sqmz
        The t0 is stored as self.cal_t0"""
        tof_list = []
        mz_list = []
        for ion in self.data:
            if (ion.use_for_mass_calib):
                if (ion.center_t) and (ion.mass) and (ion.charge):
                    tof_list.append(ion.center_t)
                    mz_list.append(ion.mass/ion.charge)

        mz_arr = np.array(mz_list)
        tof_arr = np.array(tof_list)


        # Using np.polyfit to do the linear fitting
        coeffs_sqmz_tof = np.polyfit(np.sqrt(mz_arr[mz_arr>0]), tof_arr[mz_arr>0], 1)

        # Using IPython.Display to print LaTeX
        display(Math(r"t = %.2f\sqrt{\frac{m}{z}} + %.2f" % (coeffs_sqmz_tof[0], coeffs_sqmz_tof[1])))

        coeffs_tof_sqmz = np.polyfit(tof_arr[mz_arr>0], np.sqrt(mz_arr[mz_arr>0]),1)

        display(Math(r"\sqrt{\frac{m}{z}} = %.4ft + %.4f" % (coeffs_tof_sqmz[0], coeffs_tof_sqmz[1])))
        
        self.cal_t0 = coeffs_sqmz_tof[1]
        
        self.coeffs_sqmz_tof = coeffs_sqmz_tof
        self.coeffs_tof_sqmz = coeffs_tof_sqmz
        
        self.calc_cal_mz_ions()


    def calc_cal_mz_ions(self):
        """Calculates the calibrated ion m/z for each ion in the IonCollection."""
        for ion in self.data:
            ion.calc_mz_cal(self.coeffs_tof_sqmz)

    def __getitem__(self, index):
        return self.data[index]
    
    def __iter__(self, *args, **kwargs):
        return self.data.__iter__(*args, **kwargs)
    
    def __len__(self):
        return len(self.data)
        
    def __str__(self):
        return str([ion.label for ion in self.data])
    
    def __repr__(self):
        return f"Collection with {len(self.data)} ions:\n{str(self)}"
    
    
    @wraps(Ion)
    def add_ion(self, *args, **kwargs):
        """Create Ion and append it to IonCollection"""
        self.data.append(self.ion_class(*args, **kwargs))
        
    def assign_dataset(self, dataset):
        """Assign Dataset obect to each Ion in IonCollection."""
        for ion in self.data:
            ion.assign_dataset(dataset)





class Dataset:
    """Class used for handling datasets of charged particle imaging data, based around a single
    Pandas Dataframe.

    :param data_df: the Pandas dataframe
    :param shot_array_method: method used for calculating total array of shots in the dataset
    """
    def __init__(self, data_df, shot_array_method = 'range'):
        # print(data_df)
        self.data_df = data_df
        self.columns = list(self.data_df.columns)
        self.cal_mom = False
        self.shot_array_method = shot_array_method

        
        # assert 'x' in self.columns, "Input dataframe is missing 'x'"
        # assert 'y' in self.columns, "Input dataframe is missing 'y'"
        # assert 't' in self.columns, "Input dataframe is missing 't'"
        assert 'shot' in self.columns, "Input dataframe is missing 'shot'"

        self.get_shot_array()
        
    def sep_by_custom(self, lim1, lim2, param):
        """Separate data_df by some parameter.

        :param lim1: lower limit to filter on
        :param lim2: upper limit to filter on
        :param param: data_df column name to filter on

        :return: filtered dataframe
        """
        data_df_filt = self.data_df[(self.data_df[param]>=lim1)&(self.data_df[param]<lim2)].copy()
        return(data_df_filt)

    def sep_by_tof(self, Ti, Tf):
        """Separate data_df by t between Ti and Tf.

        :return: filtered_dataframe
        """
        data_df_filt = sep_by_custom(Ti,Tf,'t')
        return(data_df_filt)

    def get_shot_array(self):
        """Get array of shots within the dataset, and store in self.shot_array"""
        if self.shot_array_method=='range':
            self.shot_array = np.arange(np.min(self.data_df.shot), np.max(self.data_df.shot))
        elif self.shot_array_method=='unique':
            self.shot_array = np.array(np.unique(self.data_df.shot))
        else:
            raise Exception("Invalid shot_array_method inputted!")

    def apply_mz_calibration(self, coeffs_tof_sqmz):
        """Apply m/z calibration for all data in data_df.

        :param coeffs_tof_sqmz: calibration coefficients from t to sqrt(m/z)
        """
        self.data_df['cal_mz'] = (self.data_df['t']*coeffs_tof_sqmz[0] + coeffs_tof_sqmz[1])**2