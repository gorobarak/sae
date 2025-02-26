import math

import numpy as np
import torch.nn as nn

from .butterfly import Butterfly

def bitreversal_permutation(n):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return perm.squeeze(0)



def calc_k_function(n1,n2):

    k1 = max(n1 // 5 , (int(np.log2(n1)) ** 2) + 1)
    k2 = max(n2 // 5 , (int(np.log2(n2)) ** 2) + 1)
    return k1, k2

# Definition of BF

def BF(input_dim,output_dim):

    n1,n2 = input_dim, output_dim
    k1,k2 = calc_k_function(n1,n2)
    first_gadget = Butterfly(in_size=n1, out_size=k1, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
    second_gadget = nn.Linear(k1,k2,bias=False)
    third_gadget = Butterfly(in_size=k2, out_size = n2, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
    
    return nn.Sequential(first_gadget,second_gadget,third_gadget)