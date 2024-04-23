import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from numba import jit
from numba.typed import Dict
from numba.core import types
from numba import prange, int64, double
import numba as nb
import math
import time


float_array = types.float64[:]
float_single = types.float64

### Just constants
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