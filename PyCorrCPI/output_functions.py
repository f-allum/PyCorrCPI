from .imports import *
from .helpers_numba import *
from .data import *

@njit
def beta_psum_KER(vec_list, mag_list, mass_list, bin_list, dim_list):
    """Represent covariance as a function of the relative recoil angle between them,
    the sums of their absolute momentun, and the sums of their KEs. 

    :param vec_list: list of 3D momentum vectors of each ion
    :param mag_list: list of momentum magnitudes of each ion
    :param mass_list: list of masses of each ion
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    """
    x_bin,y_bin,z_bin = bin_list
    
    pmag_sum=0
    KE_sum=0
    for vec, mag, mass in zip(vec_list,mag_list,mass_list):
        pmag_sum+=mag
        KE = ((mag**2)/(2*mass))* p_au_KE_eV_fac
        KE_sum+=KE

    A_vec = vec_list[0]
    B_vec = vec_list[1]
    A_mag = mag_list[0]
    B_mag = mag_list[1]
    beta = np.arccos(dot3D(A_vec,B_vec)/(A_mag*B_mag))*(180/np.pi)

    pmag_sum_binned = int(pmag_sum/y_bin+0.5)
    KE_sum_binned = int(KE_sum/z_bin+0.5)
    beta_binned = int(beta/x_bin+0.5)

    return([beta_binned],[pmag_sum_binned], [KE_sum_binned])


@njit
def output_format_Newton_2fold(vec_list, mag_list, mass_list, bin_list, dim_list):
    """Represent covariance as 2fold Newton plot. Ion A is used as reference, and we plot the 3D momentum of 
    Ion B relative to this. Note, third dimension is not used.

    :param vec_list: list of 3D momentum vectors of each ion
    :param mag_list: list of momentum magnitudes of each ion
    :param mass_list: list of masses of each ion
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]"""
    A_vec,B_vec = vec_list
    A_mag,B_mag = mag_list
    x_bin,y_bin,z_bin = bin_list
    x_pixels,y_pixels, z_pixels = dim_list

    ## use momn conservation to find 'missing' momentum
    C_vec = -(A_vec+B_vec)
    C_mag = norm3D(C_vec)
    
    angle1 = np.arccos(dot3D(A_vec,B_vec)/(A_mag*B_mag))
    angle2 = np.arccos(dot3D(A_vec,C_vec)/(A_mag*C_mag))
    Bproj_par = B_mag*np.cos(angle1)
    Bproj_perp = B_mag*np.sin(angle1)
    Cproj_par = C_mag*np.cos(angle2)
    Cproj_perp = -C_mag*np.sin(angle2)  
    Aproj_par = A_mag
    Aproj_perp=0
    
    Aproj_par = int(Aproj_par/x_bin+0.5+x_pixels/2)
    Bproj_par = int(Bproj_par/x_bin+0.5+x_pixels/2)
    Cproj_par = int(Cproj_par/x_bin+0.5+x_pixels/2)
    Cproj_perp = int(Cproj_perp/y_bin+0.5+y_pixels/2)
    Aproj_perp = int(Aproj_perp/y_bin+0.5+y_pixels/2)
    Bproj_perp = int(Bproj_perp/y_bin+0.5+y_pixels/2)
    
    return([Bproj_perp],[Bproj_par], [0,0])

@njit
def output_format_Newton_3fold(vec_list, mag_list, mass_list, bin_list, dim_list):
    """Represent covariance as 3fold Newton plot. Ion A defines a reference vector (y), with A and B defining (xy) plane.
    Ion C is plotted in this frame.

    :param vec_list: list of 3D momentum vectors of each ion
    :param mag_list: list of momentum magnitudes of each ion
    :param mass_list: list of masses of each ion
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]"""

    z_vec_norm = np.array([0.,0.,1.])
    rot_matrix = np.array([[0.,0.,0.],
      [0.,0.,0.],
      [0.,0.,0.]], dtype='float64')
    
    normal = np.zeros(3, dtype='float64')
    ax_rot = np.zeros(3, dtype='float64')

    A_vec,B_vec,C_vec = vec_list
    A_mag,B_mag,C_mag = mag_list
    x_bin,y_bin,z_bin = bin_list
    x_pixels,y_pixels, z_pixels = dim_list

    A_vec_norm = A_vec/A_mag
    B_vec_norm = B_vec/B_mag
    
    ## find the vector normal to the A,B plane through cross product
    cross3D(B_vec_norm,A_vec_norm,normal)
    normal_vec_norm = normal/norm3D(normal) #this is unecessary as cross product was on normalized vectors(?)

    ## rotate ions into plane
    alpha = np.arccos(normal_vec_norm[2])
    cross3D(normal_vec_norm,z_vec_norm,ax_rot)
    rotation_matrix(ax_rot, -alpha, rot_matrix)

    C_vec_rot = dot3D(C_vec,rot_matrix)
    B_vec_rot = dot3D(B_vec,rot_matrix)
    A_vec_rot = dot3D(A_vec,rot_matrix)

    ## rotate ions in plane to align the A+ vector
    A_vec_norm_rot = dot3D(A_vec_norm,rot_matrix)
    A_vec_norm_rot_norm = A_vec_norm_rot/norm3D(A_vec_norm_rot)

    beta = np.arctan2(A_vec_norm_rot_norm[0],A_vec_norm_rot_norm[1])
    rotation_matrix(z_vec_norm, -beta, rot_matrix)

    C_vec_rot2 = dot3D(C_vec_rot,rot_matrix)
    B_vec_rot2 = dot3D(B_vec_rot,rot_matrix)
    A_vec_rot2 = dot3D(A_vec_rot,rot_matrix)          

    ## bin the C momentum vector in this frame and return
    C_bin_x = int(C_vec_rot2[0]/x_bin+0.5+x_pixels/2)
    C_bin_y = int(C_vec_rot2[1]/y_bin+0.5+y_pixels/2)
    C_bin_z = int(C_vec_rot2[2]/z_bin+0.5+z_pixels/2)


    ## bin the B momentum vector in this frame and return
    B_bin_x = int(B_vec_rot2[0]/x_bin+0.5+x_pixels/2)
    B_bin_y = int(B_vec_rot2[1]/y_bin+0.5+y_pixels/2)
    B_bin_z = int(B_vec_rot2[2]/z_bin+0.5+z_pixels/2)

    # return(([B_bin_x,C_bin_x],[B_bin_y,C_bin_y],[B_bin_z,C_bin_z]))
    return(([C_bin_x],[C_bin_y],[C_bin_z]))


@njit
def output_format_Newton_2fold_3rdbody(vec_list, mag_list, mass_list, bin_list, dim_list):
    """Represent covariance as 2fold Newton plot. Ion A is used as reference, and we plot the 3D momentum of 
    Ion B relative to this. Note, third dimension is not used.

    :param vec_list: list of 3D momentum vectors of each ion
    :param mag_list: list of momentum magnitudes of each ion
    :param mass_list: list of masses of each ion
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]"""
    A_vec,B_vec = vec_list
    A_mag,B_mag = mag_list
    x_bin,y_bin,z_bin = bin_list
    x_pixels,y_pixels, z_pixels = dim_list

    ## use momn conservation to find 'missing' momentum
    C_vec = -(A_vec+B_vec)
    C_mag = norm3D(C_vec)
    
    angle1 = np.arccos(dot3D(A_vec,B_vec)/(A_mag*B_mag))
    angle2 = np.arccos(dot3D(A_vec,C_vec)/(A_mag*C_mag))
    Bproj_par = B_mag*np.cos(angle1)
    Bproj_perp = B_mag*np.sin(angle1)
    Cproj_par = C_mag*np.cos(angle2)
    Cproj_perp = -C_mag*np.sin(angle2)  
    Aproj_par = A_mag
    Aproj_perp=0
    
    Aproj_par = int(Aproj_par/x_bin+0.5+x_pixels/2)
    Bproj_par = int(Bproj_par/x_bin+0.5+x_pixels/2)
    Cproj_par = int(Cproj_par/x_bin+0.5+x_pixels/2)
    Cproj_perp = int(Cproj_perp/y_bin+0.5+y_pixels/2)
    Aproj_perp = int(Aproj_perp/y_bin+0.5+y_pixels/2)
    Bproj_perp = int(Bproj_perp/y_bin+0.5+y_pixels/2)
    
    return([Bproj_perp, Cproj_perp],[Bproj_par, Cproj_par], [0,0])

@njit
def output_format_Newton_4fold(vec_list, mag_list, mass_list, bin_list, dim_list):
    """Represent covariance as 4fold Newton plot. Ion A defines a reference vector (y), with A and B defining (xy) plane.
    Ion C and D are plotted in this frame.

    :param vec_list: list of 3D momentum vectors of each ion
    :param mag_list: list of momentum magnitudes of each ion
    :param mass_list: list of masses of each ion
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    """

    z_vec_norm = np.array([0.,0.,1.])
    rot_matrix = np.array([[0.,0.,0.],
      [0.,0.,0.],
      [0.,0.,0.]], dtype='float64')
    
    normal = np.zeros(3, dtype='float64')
    ax_rot = np.zeros(3, dtype='float64')

    A_vec,B_vec,C_vec,D_vec = vec_list
    A_mag,B_mag,C_mag,D_mag = mag_list
    x_bin,y_bin,z_bin = bin_list
    x_pixels,y_pixels, z_pixels = dim_list

    A_vec_norm = A_vec/A_mag
    B_vec_norm = B_vec/B_mag
    
    ## find the vector normal to the A,B plane through cross product
    cross3D(B_vec_norm,A_vec_norm,normal)
    normal_vec_norm = normal/norm3D(normal) #this is unecessary as cross product was on normalized vectors(?)

    ## rotate ions into plane
    alpha = np.arccos(normal_vec_norm[2])
    cross3D(normal_vec_norm,z_vec_norm,ax_rot)
    rotation_matrix(ax_rot, -alpha, rot_matrix)

    D_vec_rot = dot3D(D_vec,rot_matrix)
    C_vec_rot = dot3D(C_vec,rot_matrix)
    B_vec_rot = dot3D(B_vec,rot_matrix)
    A_vec_rot = dot3D(A_vec,rot_matrix)

    ## rotate ions in plane to align the A+ vector
    A_vec_norm_rot = dot3D(A_vec_norm,rot_matrix)
    A_vec_norm_rot_norm = A_vec_norm_rot/norm3D(A_vec_norm_rot)

    beta = np.arctan2(A_vec_norm_rot_norm[0],A_vec_norm_rot_norm[1])
    rotation_matrix(z_vec_norm, -beta, rot_matrix)

    D_vec_rot2 = dot3D(D_vec_rot,rot_matrix)
    C_vec_rot2 = dot3D(C_vec_rot,rot_matrix)
    B_vec_rot2 = dot3D(B_vec_rot,rot_matrix)
    A_vec_rot2 = dot3D(A_vec_rot,rot_matrix)          

    ## bin the C momentum vector in this frame and return
    C_bin_x = int(C_vec_rot2[0]/x_bin+0.5+x_pixels/2)
    C_bin_y = int(C_vec_rot2[1]/y_bin+0.5+y_pixels/2)
    C_bin_z = int(C_vec_rot2[2]/z_bin+0.5+z_pixels/2)

    D_bin_x = int(D_vec_rot2[0]/x_bin+0.5+x_pixels/2)
    D_bin_y = int(D_vec_rot2[1]/y_bin+0.5+y_pixels/2)
    D_bin_z = int(D_vec_rot2[2]/z_bin+0.5+z_pixels/2)

    return(([C_bin_x, D_bin_x],[C_bin_y, D_bin_y],[C_bin_z, D_bin_z]))


@njit()
def output_format_TfAcAc(vec_list, mag_list, mass_list, bin_list, dim_list):
    """Fourfold Newton plot where the sum of A and B make a reference vector and plane.
    The momentum of C is restricted to have px<0 in this frame. D is plotted.

    :param vec_list: list of 3D momentum vectors of each ion
    :param mag_list: list of momentum magnitudes of each ion
    :param mass_list: list of masses of each ion
    :param dim_list: list of the size of the output covariance dimensions [n_x,n_y,n_z]
    :param bin_list: binning factor used for computing the covariance histograms [b_x,b_y,b_z]
    """

    z_vec_norm = np.array([0.,0.,1.])
    rot_matrix = np.array([[0.,0.,0.],
      [0.,0.,0.],
      [0.,0.,0.]], dtype='float64')
    
    normal = np.zeros(3, dtype='float64')
    ax_rot = np.zeros(3, dtype='float64')
    
    A_vec,B_vec,C_vec,D_vec = vec_list
    A_mag, B_mag, C_mag, D_mag= mag_list
    x_bin,y_bin,z_bin = bin_list
    x_pixels,y_pixels, z_pixels = dim_list
    
    A_vec_norm = A_vec/A_mag
    B_vec_norm = B_vec/B_mag
    ABsum_vec = A_vec+B_vec
    ABsum_vec_norm = ABsum_vec/norm3D(ABsum_vec)
    
    cross3D(A_vec_norm,B_vec_norm,normal)

    normal_vec_norm = normal/norm3D(normal)

    alpha = np.arccos(normal_vec_norm[2])
    ax_rot = np.cross(normal_vec_norm,z_vec_norm)
    
    rotation_matrix(ax_rot, -alpha, rot_matrix)

    A_vec_rot = dot3D(A_vec,rot_matrix)
    B_vec_rot = dot3D(B_vec,rot_matrix)
    C_vec_rot = dot3D(C_vec,rot_matrix)
    D_vec_rot = dot3D(D_vec,rot_matrix)
    
    ABsum_vec_rot = A_vec_rot+B_vec_rot
    
    ABsum_vec_rot_norm = ABsum_vec_rot/norm3D(ABsum_vec_rot)
    
    beta = np.arctan2(ABsum_vec_rot_norm[0],ABsum_vec_rot_norm[1])

    rotation_matrix(z_vec_norm, -beta, rot_matrix)

    A_vec_rot2 = dot3D(A_vec_rot,rot_matrix)
    B_vec_rot2 = dot3D(B_vec_rot,rot_matrix)
    C_vec_rot2 = dot3D(C_vec_rot,rot_matrix)
    D_vec_rot2 = dot3D(D_vec_rot,rot_matrix)
    
    rotation_matrix(np.array([0,1,0], dtype='float64'), -np.pi, rot_matrix)

    if C_vec_rot2[0]>0:
        A_vec_rot2 = dot3D(A_vec_rot2, rot_matrix)
        B_vec_rot2 = dot3D(B_vec_rot2, rot_matrix)
        C_vec_rot2 = dot3D(C_vec_rot2, rot_matrix)
        D_vec_rot2 = dot3D(D_vec_rot2, rot_matrix)

    D_bin_x = int(D_vec_rot2[0]/x_bin+0.5+x_pixels/2)
    D_bin_y = int(D_vec_rot2[1]/y_bin+0.5+y_pixels/2)
    D_bin_z = int(D_vec_rot2[2]/z_bin+0.5+z_pixels/2)

    return(([D_bin_x],[D_bin_y],[D_bin_z]))



