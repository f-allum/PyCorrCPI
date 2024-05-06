from .imports import *


@njit
def find_first(item, vec):
    """return the index of the first occurence of item in vec
    :param item: item to search for
    :param vec: array to search in
    :return: two values - the first is boolean to indicate whether search was succesful or not.
    the second is the index of the first occurence."""
    for i in range(len(vec)):
        ## maximum indexes to search before giving up. in principle could cause issues
        ## if set too low
        if i>1000:
            return((0,0))
        if item == vec[i]:
            return (1,i)
    return(0,0)

@njit
def find_last(item, vec):
    """return the index of the first occurence of item in vec
    :param item: item to search for
    :param vec: array to search in
    """
    for i in range(len(vec)):
        if item != vec[i]:
            return i-1
    return len(vec)-1




@njit
def dot3D(vec1, vec2):
    """ Calculate the dot product of two 3d vectors. """
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]

@njit
def dot2D(vec1, vec2):
    """ Calculate the dot product of two 2d vectors. """
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]

@njit
def norm2D(vec):
    """ Calculate the norm of a 2d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

@njit
def norm3D(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

@njit
def cross3D(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. Modifies results in place """
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1

@njit((nb.float64[:],nb.float64,nb.float64[:,:]))
def rotation_matrix(axis, theta, rot_matrix):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. Modifies rot_matrix in place.
    """
    axis = axis / np.sqrt(dot3D(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    

    rot_matrix[0,0]=aa + bb - cc - dd
    rot_matrix[0,1]=2 * (bc + ad)
    rot_matrix[0,2]=2 * (bd - ac)
    rot_matrix[1,0]=2 * (bc - ad)
    rot_matrix[1,1]=aa + cc - bb - dd
    rot_matrix[1,2]=2 * (cd + ab)
    rot_matrix[2,0]=2 * (bd + ac)
    rot_matrix[2,1]=2 * (cd - ab)
    rot_matrix[2,2]=aa + dd - bb - cc
        

@njit
def calc_psum_abs(vec_list,mag_list,mass_list):
    """Calculate the magnitude of the sum of vectors.
    Used for filtering on psum of a coincidence."""
    vec_list_sum = np.zeros(3, dtype='float64')
    for vec in vec_list:
        vec_list_sum+=vec
    vec_list_sum_mag = norm3D(vec_list_sum)
    return(vec_list_sum_mag)
