#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:10:13 2024

@author: hk
"""

import numpy as np
from math import factorial, sin, cos, sqrt

# Load and parse the file (assuming file is already read into file_content)
file_path = '/Users/hk/RAMathUBC/N14_M960_IcoC1.dat'

# Reading the file
with open(file_path, 'r') as file:
    file_content = file.readlines()

# Skip the first two lines (meta information) and process the rest
data_lines = file_content[2:]

# Extract points and weights
points = []
weights = []

# Each line contains 9 coordinates followed by a weight
for line in data_lines:
    values = list(map(float, line.split()))
    points.append(values[:-1])  # First 9 values are the points
    weights.append(values[-1])  # Last value is the weight

# Convert lists to numpy arrays for better handling
points = np.array(points)
weights = np.array(weights)

def euler_from_matrix(mat):
    """
    Convert a 3x3 rotation matrix into Euler angles (ZYZ convention).
    
    Parameters:
    mat (array): A 3x3 rotation matrix.

    Returns:
    tuple: Euler angles (alpha, beta, gamma).
    """
    # Extract the Euler angles from the rotation matrix assuming ZYZ convention
    if mat[2, 2] < 1:
        if mat[2, 2] > -1:
            beta = np.arccos(mat[2, 2])
            alpha = np.arctan2(mat[1, 2], mat[0, 2])
            gamma = np.arctan2(mat[2, 1], -mat[2, 0])
        else:
            beta = np.pi
            alpha = -np.arctan2(-mat[1, 0], mat[1, 1])
            gamma = 0
    else:
        beta = 0
        alpha = np.arctan2(-mat[1, 0], mat[1, 1])
        gamma = 0
    
    return alpha, beta, gamma

def small_d_matrix(l, m_prime, m, beta):
    """
    Calculate the Wigner small d-matrix element d^l_{m', m}(\beta).

    Parameters:
    l (int): The total angular momentum.
    m_prime (int): The m' index.
    m (int): The m index.
    beta (float): The angle.

    Returns:
    float: The Wigner small d-matrix element d^l_{m', m}(\beta).
    """
    if m_prime < -l or m_prime > l or m < -l or m > l:
        return 0

    # Check for the symmetry of the small d-matrix
    if m_prime < m:
        return (-1)**(m - m_prime) * small_d_matrix(l, m, m_prime, beta)

    # Compute the d-matrix element
    prefactor = sqrt(factorial(l + m) * factorial(l - m) / (factorial(l + m_prime) * factorial(l - m_prime)))
    
    # The sine and cosine terms
    sine_term = sin(beta / 2) ** (m_prime - m) * cos(beta / 2) ** (m + m_prime)
    
    return prefactor * sine_term

def wigner_D_l(l, alpha, beta, gamma):
    """
    Generate the Wigner D^l matrix for given l and Euler angles (alpha, beta, gamma).

    Parameters:
    l (int): Degree of the Wigner matrix.
    alpha (float): First Euler angle.
    beta (float): Second Euler angle.
    gamma (float): Third Euler angle.

    Returns:
    numpy.ndarray: The Wigner D^l matrix of shape (2l+1, 2l+1).
    """
    size = 2 * l + 1
    D_matrix = np.zeros((size, size), dtype=complex)

    for m_prime in range(-l, l + 1):
        for m in range(-l, l + 1):
            d_l_mpm = small_d_matrix(l, m_prime, m, beta)  # Wigner small d-matrix element
            D_matrix[m_prime + l, m + l] = (
                np.exp(-1j * m_prime * alpha) * d_l_mpm * np.exp(-1j * m * gamma)
            )

    return D_matrix


def quadrature_integration_D(l, m_prime, m):
    """
    Perform numerical integration of D^l_{m',m} over SO(3).

    Parameters:
    l (int): Degree of the Wigner matrix.
    m_prime (int): The m' index.
    m (int): The m index.

    Returns:
    complex: The result of the integration.
    """
    result = 0.0 + 0.0j

    # For each quadrature point (9D point), evaluate D^l_{m',m}
    for i, point in enumerate(points):
        # Reshape the 9 coordinates into a 3x3 matrix
        mat = np.array(point).reshape((3, 3))
        
        # Extract Euler angles (alpha, beta, gamma) from the rotation matrix
        alpha, beta, gamma = euler_from_matrix(mat)
        
        # Get the Wigner D^l matrix for these angles
        D_matrix = wigner_D_l(l, alpha, beta, gamma)
        
        # Extract the specific D^l_{m', m} element
        d_l_mpm = D_matrix[m_prime + l, m + l]
        
        # Multiply by the weight and accumulate
        result += d_l_mpm * weights[i]

    return result

# Example: Integrating D^1_{1, 0} over SO(3)
l = 9  # Degree of the Wigner D matrix
m_prime = 8  # m' index
m = -8  # m index
result = quadrature_integration_D(l, m_prime, m)
print(f"Integration result for D^{l}_{{{m_prime}, {m}}} over SO(3):", result)
