from .imports import *
from .helpers_numba import *
from .covariance import *




class Ion:
    """Class used to define the parameters to extract data from different ions from a dataset"""
    def __init__(self, label, filter_i, filter_f, dataset, filter_param='tof',
        center_x=None, center_y=None, center_t=None, mass=None, charge=None, shot_array_method='range'):
        self.label = label
        self.filter_i = filter_i
        self.filter_f = filter_f
        self.filter_param=filter_param
        self.shot_array_method = shot_array_method
        self.mass=mass

        if center_x:
            self.center_x=center_x
        else:
            self.center_x=0
        if center_y:
            self.center_y=center_y
        else:
            self.center_y=0
        
        if not [i for i in (center_x,center_y) if i is None]:
            self.center_given=True
        else:
            self.center_given=False
        if center_t:
            self.center_t = center_t
        # else:
        #     self.center_t = (self.Ti+self.Tf)/2

        self.grab_data(dataset)
        self.get_shot_array()

            
    def print_details(self):
        for key,value in self.__dict__.items():
            if key not in ['data_df','data_array', 'shot_array']:
                print(f"'{key}':{value}")

    def grab_data(self,dataset):
        """Gets data corresponding to ion from dataset based on the inputted filter"""
        try:
            self.data_df = dataset.sep_by_custom(self.filter_i,self.filter_f, self.filter_param)
        except:
            raise Exception("filter_param is not found in dataframe!")
        self.dataframe_to_arr()

    def dataframe_to_arr(self):
        """Converts necessary dataframe columns for covariance calculation to an array"""
        self.data_array = self.data_df[["px_AU", "py_AU", "pz_AU", "shot", "pmag_AU"]].to_numpy()

    def get_shot_array(self):
        """Find array of shots in dataset which contain this ion"""
        if self.shot_array_method=='range':
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



class Dataset:
    def __init__(self, data_df, C_xy = None, C_z = None, shot_array_method = 'range'):
        # print(data_df)
        self.data_df = data_df
        self.columns = list(self.data_df.columns)
        self.cal_mom = False
        self.C_xy = C_xy
        self.C_z = C_z
        self.shot_array_method = shot_array_method

        
        # assert 'x' in self.columns, "Input dataframe is missing 'x'"
        # assert 'y' in self.columns, "Input dataframe is missing 'y'"
        # assert 't' in self.columns, "Input dataframe is missing 't'"
        assert 'shot' in self.columns, "Input dataframe is missing 'shot'"

        self.get_shot_array()
        
        
    # def sep_by_tof(self, Ti, Tf):
    #     data_df_filt = self.data_df[(self.data_df['t']>=Ti)&(self.data_df['t']<Tf)].copy()
    #     return(data_df_filt)

    def sep_by_tof(self, Ti, Tf):
        data_df_filt = self.data_df[(self.data_df['tof']>=Ti)&(self.data_df['tof']<Tf)].copy()
        return(data_df_filt)

    def sep_by_custom(self, lim1, lim2, param):
        data_df_filt = self.data_df[(self.data_df[param]>=lim1)&(self.data_df[param]<lim2)].copy()
        return(data_df_filt)
    
    def calibrate_all_momenta(self, ion_list):
        for ion in ion_list:
            self.calibrate_momenta(ion)

    def get_shot_array(self):
        if self.shot_array_method=='range':
            self.shot_array = np.arange(np.min(self.data_df.shot), np.max(self.data_df.shot))
        elif self.shot_array_method=='unique':
            self.shot_array = np.array(np.unique(self.data_df.shot))
        else:
            raise Exception("Invalid shot_array_method inputted!")
        
        
    #### need to move this function into ion class...
    def calibrate_momenta(self, ion, C_xy=None, C_z=None, fit_center=False):
        ion_mask = self.data_df[(self.data_df['t']>=ion.Ti)&(self.data_df['t']<ion.Tf)]
        data_df_ion = self.data_df[ion_mask]
        
        if C_xy:
            self.C_xy = C_xy
        if C_z:
            self.C_z = C_z
        
        if fit_center:
            data_df_ion['x_centered'] = data_df_ion-ion.centre_x_fit
            data_df_ion['y_centered'] = data_df_ion-ion.centre_y_fit
        else:
            data_df_ion['x_centered'] = data_df_ion-ion.centre_x
            data_df_ion['y_centered'] = data_df_ion-ion.centre_y
            
        # if this is the first time the dataset has had momentum calibration run on it, populate new columns
        # in the dataframe
        if self.cal_mom==False:
            self.data_df_ion['ion'] = np.nan
            self.data_df_ion['t_relative'] = np.nan
            self.data_df_ion['vx'] = np.nan
            self.data_df_ion['vy'] = np.nan
            self.data_df_ion['vz'] = np.nan
            self.data_df_ion['px'] = np.nan
            self.data_df_ion['py'] = np.nan
            self.data_df_ion['pz'] = np.nan
            self.data_df_ion['vmag'] = np.nan
            self.data_df_ion['pmag'] = np.nan
            self.cal_mom=True
        
        self.data_df['t_absolute'] = self.data_df_ion['t']-self.t0
        self.data_df.loc[ion_mask, 't_relative'] = self.data_df.loc[ion_mask, 't']-ion.centre_t
        
        self.data_df.loc[ion_mask,'vx'] = C_xy*(self.data_df.loc[ion_mask,'x_centered']/self.data_df.loc[ion_mask,'tcorr'])
        self.data_df.loc[ion_mask,'vy'] = C_xy*(self.data_df.loc[ion_mask,'y_centered']/self.data_df.loc[ion_mask,'tcorr'])
        self.data_df.loc[ion_mask,'px'] = self.data_df.loc[ion_mask,'vx'] * ion.mass
        self.data_df.loc[ion_mask,'py'] = self.data_df.loc[ion_mask,'vy'] * ion.mass
        self.data_df.loc[ion_mask,'vz'] = (C_z*charge*(self.data_df.loc[ion_mask,'t_centered']))/ion.mass
        self.data_df.loc[ion_mask,'pz'] = self.data_df.loc[ion_mask,'vz'] * ion.mass
        
        self.data_df.loc[ion_mask,'pmag'] = np.sqrt((self.data_df.loc[ion_mask,'px']**2+self.data_df.loc[ion_mask,'py']**2+self.data_df.loc[ion_mask,'pz']**2))
        self.data_df.loc[ion_mask,'vmag'] = np.sqrt((self.data_df.loc[ion_mask,'vx']**2+self.data_df.loc[ion_mask,'vy']**2+self.data_df.loc[ion_mask,'self.data_df.loc[ion_mask,vz']**2))
        
        return(data_df_ion)
    
    ### should be moved out of class
    def mz_calibration(self, ion_list):
        tof_list = []
        mz_list = []
        for ion in ion_list:
            if (ion.mass&ion.charge):
                tof_list.append(ion.centre_t)
                mz_list.append(ion.mass/ion.charge)

        mz_arr = np.array(mz_list)
        tof_arr = np.array(tof_list)

        # Using np.polyfit to do the linear fitting
        coeffs_sqmz_tof = np.polyfit(np.sqrt(mz_arr[mz_arr>0]), tof_arr[mz_arr>0], 1)

        # Using IPython.Display to print LaTeX
        display(Math(r"t = %.2f\sqrt{\frac{m}{z}} + %.2f" % (coeffs_sqmz_tof[0], coeffs_sqmz_tof[1])))

        coeffs_tof_sqmz = np.polyfit(tof_arr[z_arr>0], np.sqrt(z_arr[z_arr>0]),1)

        display(Math(r"\sqrt{\frac{m}{z}} = %.4ft + %.4f" % (coeffs_tof_sqmz[0], coeffs_tof_sqmz[1])))
        
        self.t0 = coeffs_sqmz_tof[1]
        
        self.coeffs_sqmz_tof = coeffs_sqmz_tof
        self.coeffs_tof_sqmz = coeffs_tof_sqmz
        
        self.apply_mz_calibration()

        return(coeffs_sqmz_tof, coeffs_tof_sqmz)
    
    def apply_mz_calibration(self):
        self.data_df['cal_mz'] = self.data_df['t']*self.coeffs_tof_sqmz[0] + self.coeffs_tof_sqmz[1]
        
    def apply_jet_correction(self):
        self.data_df['xcorr'] = self.data_df['x']
        self.data_df['ycorr'] = self.data_df['y']