# Simple Python implementation of DHWT
import numpy as np

def transform(array):
    """
    Simple Discrete Haar Wavelet Transform implementation
    """
    n = len(array)
    if n % 2 != 0:
        # Pad with last element if odd length
        array = np.append(array, array[-1])
        n = len(array)
    
    new_dim = n // 2
    wavelet = np.zeros(new_dim)
    coeffs = np.zeros(new_dim)
    
    for i in range(new_dim):
        avg = (array[2*i] + array[2*i + 1]) / 2.0
        diff = (array[2*i] - array[2*i + 1]) / 2.0
        wavelet[i] = avg
        coeffs[i] = diff
    
    return wavelet, coeffs

def inverse_transform(wavelet, coeffs):
    """
    Inverse Discrete Haar Wavelet Transform
    """
    n = len(wavelet)
    array = np.zeros(2 * n)
    
    for i in range(n):
        array[2*i] = wavelet[i] + coeffs[i]
        array[2*i + 1] = wavelet[i] - coeffs[i]
    
    return array
