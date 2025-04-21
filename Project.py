#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 13:10:53 2025

@author: hk
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y
from sympy.physics.wigner import clebsch_gordan
from sympy import S
import jax 
jax.config.update("jax_enable_x64", True)
import s2fft
from tqdm import tqdm


def sample_spheres_uniform(N, d):
    phi = np.random.uniform(0, 2 * np.pi, size=(N, d))
    
    cos_theta = np.random.uniform(-1, 1, size=(N, d))
    theta = np.arccos(cos_theta)  

    points = np.stack((phi, theta), axis=-1)
    return points


def twod_coeff_all_vectorized(degree):
    l1_vals = np.arange(degree + 1)
    pairs = [(l1, l2) for l1 in l1_vals for l2 in range(degree - l1 + 1)]

    result = []

    for l1, l2 in pairs:
        m1_vals = np.arange(-l1, l1 + 1)
        m2_vals = np.arange(-l2, l2 + 1)

        m1, m2 = np.meshgrid(m1_vals, m2_vals, indexing='ij')
        m1 = m1.flatten()
        m2 = m2.flatten()

        pair_array = np.stack([
            np.column_stack((np.full_like(m1, l1), m1)),
            np.column_stack((np.full_like(m2, l2), m2))
        ], axis=1)

        result.append(pair_array)

    return np.vstack(result)  


def eval_test_vectorized_batch(degree, rng_coeff, points):

    phi0 = points[:, 0, 0]
    theta0 = points[:, 0, 1]
    phi1 = points[:, 1, 0]
    theta1 = points[:, 1, 1]
    N = len(points)
    aout = np.zeros(N, dtype=np.complex128)

    for l in range(degree):
        m_vals = np.arange(-l, l)  
        mu_vals = np.arange(-l, l)  
        M, MU = np.meshgrid(m_vals, mu_vals, indexing='ij')  

        Y1 = sph_harm_y(l, mu_vals, theta0[:, None], phi0[:, None])   
        Y2 = sph_harm_y(l,-mu_vals, theta1[:, None], phi1[:, None])  
        term_matrix = ((-1) ** ((l - M - MU)%2)) / (2 * l + 1) 

        terms = np.einsum('ij,nj,ni->n', term_matrix, Y2, Y1)  

        coeffs = rng_coeff[l, m_vals + l]  # shape: (2l,)
        aout += np.dot(terms[:, None], coeffs[None, :]).sum(axis=1) * np.exp(-l)

    return aout
    


def invar_basis_design_matrix_2D(points, degree):

    phi0 = points[:, 0, 0]
    theta0 = points[:, 0, 1]
    phi1 = points[:, 1, 0]
    theta1 = points[:, 1, 1]
    basis_columns = []

    for l in range(degree):
        m_vals = np.arange(-l, l)
        mu_vals = np.arange(-l, l)
        M, MU = np.meshgrid(m_vals, mu_vals, indexing='ij')  # shape (2l, 2l)

        Y_mu = sph_harm_y(l,  mu_vals, theta0[:, None],  phi0[:, None])     # shape (N, 2l)
        Y_neg_mu = sph_harm_y(l, -mu_vals, theta1[:, None], phi1[:, None]) # shape (N, 2l)

        weights = ((-1) ** ((l - M - MU)%2)) / (2 * l + 1)  # shape (2l, 2l)

        for i, m in enumerate(m_vals):
            # Use einsum to compute the bilinear sum over mu for all points
            B_lm = np.einsum('j,nj,nj->n', weights[i], Y_mu, Y_neg_mu)
            basis_columns.append(B_lm * np.exp(-l))  # apply scaling

    phi = np.stack(basis_columns, axis=1)  # shape (N, B)
    return phi
    

def fit_invar_least_squares_2D(points, values, degree):

    phi = invar_basis_design_matrix_2D(points, degree) 
    y = values
    coeffs = np.linalg.pinv(phi) @ y 
    return coeffs

def eval_invar_approx_2D(coeffs, degree, new_points):
    new_value = invar_basis_design_matrix_2D(new_points, degree)
    return new_value @ coeffs

def full_basis_design_matrix_2D(degree, rng_coeff, points):
    N = points.shape[0]
    basis = twod_coeff_all_vectorized(degree)  
    B = basis.shape[0]

    phi1 = points[:, 0, 0]
    theta1 = points[:, 0, 1]
    phi2 = points[:, 1, 0]
    theta2 = points[:, 1, 1]

    phi = np.zeros((N, B), dtype=np.complex128)

    for i, ((l1, m1), (l2, m2)) in enumerate(basis):
        Y1 = sph_harm_y(l1, m1, theta1, phi1)
        Y2 = sph_harm_y(l2, m2, theta2, phi2)
        phi[:, i] = Y1 * Y2

    return phi

def fit_full_least_squares_2D(points, values, degree):

    Φ = full_basis_design_matrix_2D(points, degree)  
    y = values
    coeffs = np.linalg.pinv(Φ) @ y 
    return coeffs

def eval_full_approx_2D(coeffs, degree, new_points):
    new_value = full_basis_design_matrix_2D(new_points, degree)
    return new_value @ coeffs


def eval_invar_sph_basis_3D(l1, l2, l3, m1, m2, m3, points):
    if m1 + m2 + m3 != 0:
        return np.zeros(points.shape[0], dtype=np.complex128)
    N = points.shape[0]
    result = np.zeros(N, dtype=np.complex128)

    phi1, theta1 = points[:, 0, 0], points[:, 0, 1]
    phi2, theta2 = points[:, 1, 0], points[:, 1, 1]
    phi3, theta3 = points[:, 2, 0], points[:, 2, 1]

    L_min = abs(l1 - l2)
    L_max = l1 + l2

    for L in range(L_min, L_max + 1):
        try:
            C_mm = float(clebsch_gordan(S(l1), S(m1), S(l2), S(m2), S(L), S(m1 + m2)))
            C_mm3 = float(clebsch_gordan(S(L), S(m1 + m2), S(L), S(m3), S(0), S(0)))
        except:
            continue

        for mu1 in range(-l1, l1 + 1):
            mu2 = -m3 - mu1
            if not (-l2 <= mu2 <= l2):
                continue

            try:
                C_mu = float(clebsch_gordan(S(l1), S(mu1), S(l2), S(mu2), S(L), S(mu1 + mu2)))
                C_mu3 = float(clebsch_gordan(S(L), S(mu1 + mu2), S(L), S(m3), S(0), S(0)))
            except:
                continue

            Y1 = sph_harm_y(l1, m1, theta1, phi1)
            Y2 = sph_harm_y(l2, m2, theta2, phi2)
            Y3 = sph_harm_y(l3, m3, theta3, phi3)

            coeff = C_mm * C_mu * C_mm3 * C_mu3
            result += coeff * Y1 * Y2 * Y3

    return result

def evaluate_sum_of_basis_3D(points, coeffs, degree):
    """
    Evaluate the function f(points) = sum c_{l1l2l3m1m2m3} * B_{l1l2l3}^{m1m2m3}(points)

    Parameters:
        points: ndarray of shape (N, 3, 2)
        coeffs: dict with keys (l1, l2, l3, m1, m2, m3) and real values
        degree: int, upper bound on l1 + l2 + l3

    Returns:
        values: ndarray of shape (N,) with complex values
    """
    N = points.shape[0]
    result = np.zeros(N, dtype=np.complex128)

    for (l1, l2, l3, m1, m2, m3), c_val in coeffs.items():
        if l1 + l2 + l3 > degree:
            continue
        if m1 + m2 + m3 != 0:
            continue
        basis_val = eval_invar_sph_basis_3D(l1, l2, l3, m1, m2, m3, points)
        result += c_val * basis_val

    return result



def generate_invar_basis_keys_3D(degree):
    """
    Generate constrained basis keys with m1 + m2 + m3 = 0 and l1+l2+l3 <= degree.
    
    Returns:
        keys: list of (l1, l2, l3, m1, m2, m3)
    """
    keys = []
    for l1 in range(degree + 1):
        for l2 in range(degree + 1 - l1):
            for l3 in range(degree + 1 - l1 - l2):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        m3 = -(m1 + m2)
                        if -l3 <= m3 <= l3:
                            keys.append((l1, l2, l3, m1, m2, m3))
    return keys

def eval_invar_design_matrix_3D(points, degree):
    """
    Construct the design matrix for the invariant spherical harmonic basis.
    
    Parameters:
        points: ndarray of shape (N, 3, 2)
        degree: int
    
    Returns:
        Φ: ndarray of shape (N, B)
        keys: list of (l1, l2, l3, m1, m2, m3)
    """
    keys = generate_invar_basis_keys_3D(degree)
    N = points.shape[0]
    B = len(keys)
    Φ = np.zeros((N, B), dtype=np.complex128)

    for j, (l1, l2, l3, m1, m2, m3) in enumerate(keys):
        Φ[:, j] = eval_invar_sph_basis_3D(l1, l2, l3, m1, m2, m3, points)

    return Φ, keys

def fit_invar_least_squares_3D(points, values, degree):

    
    Φ, keys = eval_invar_design_matrix_3D(points, degree)  

    coeff_vector, *_ = np.linalg.lstsq(Φ, values, rcond=None)  

    coeffs = {tuple(key): coeff_vector[i] for i, key in enumerate(keys)}
    return coeffs    



def evaluate_invar_least_squares_3D(points_new, coeffs, degree):

    Φ_new, keys =  eval_invar_design_matrix_3D(points_new, degree)

    coeff_vector = np.array([coeffs.get(tuple(k), 0.0) for k in keys], dtype=np.complex128)

    return Φ_new @ coeff_vector  




def generate_basis_keys_vectorized_3D(degree):
    """
    Generate (l1, l2, l3, m1, m2, m3) with l1 + l2 + l3 <= degree

    Returns:
        keys: ndarray of shape (B, 6)
    """
    l_triplets = []
    for l1 in range(degree + 1):
        for l2 in range(degree + 1 - l1):
            for l3 in range(degree + 1 - l1 - l2):
                l_triplets.append((l1, l2, l3))

    key_list = []
    for l1, l2, l3 in l_triplets:
        m1_vals = np.arange(-l1, l1 + 1)
        m2_vals = np.arange(-l2, l2 + 1)
        m3_vals = np.arange(-l3, l3 + 1)

        M1, M2, M3 = np.meshgrid(m1_vals, m2_vals, m3_vals, indexing='ij')
        M1 = M1.flatten()
        M2 = M2.flatten()
        M3 = M3.flatten()

        n = len(M1)

        keys = np.zeros((n, 6), dtype=int)
        keys[:, 0] = l1
        keys[:, 1] = l2
        keys[:, 2] = l3
        keys[:, 3] = M1
        keys[:, 4] = M2
        keys[:, 5] = M3

        key_list.append(keys)

    return np.vstack(key_list)

def gen_coeffs_vectorized_3D(degree):
    """
    Generate coefficients c_{l1 l2 l3 m1 m2 m3} 

    Returns:
        coeffs: dict {(l1, l2, l3, m1, m2, m3): complex}
    """
    keys = generate_basis_keys_vectorized_3D(degree)
    coeffs = {}

    for key in keys:
        l1, l2, l3, m1, m2, m3 = key
        decay = np.exp(-(l1 + l2 + l3))
        real = np.random.uniform(-1, 1)
        coeffs[tuple(key)] = (real) * decay

    return coeffs

def eval_design_matrix_vectorized_3D(points, degree):
    """
    Evaluate design matrix using constrained spherical harmonic triple basis.

    Parameters:
        points: ndarray (N, 3, 2)
        degree: int

    Returns:
        Φ: ndarray (N, B)
        keys: ndarray (B, 6)
    """
    N = points.shape[0]
    keys = generate_basis_keys_vectorized_3D(degree)

    # Precompute spherical harmonics for each point and each (l, m)
    Y = [{}, {}, {}]
    for i in range(3):
        phi = points[:, i, 0]
        theta = points[:, i, 1]
        for l in range(degree + 1):
            for m in range(-l, l + 1):
                Y[i][(l, m)] = sph_harm_y(l, m, theta, phi)

    # Evaluate all basis functions
    B = len(keys)
    Φ = np.empty((N, B), dtype=np.complex128)
    for j, (l1, l2, l3, m1, m2, m3) in enumerate(keys):
        Φ[:, j] = Y[0][(l1, m1)] * Y[1][(l2, m2)] * Y[2][(l3, m3)]

    return Φ, keys


def fit_least_squares_3D(points, values, degree):

    
    Φ, keys = eval_design_matrix_vectorized_3D(points, degree)  

    coeff_vector, *_ = np.linalg.lstsq(Φ, values, rcond=None)  

    coeffs = {tuple(key): coeff_vector[i] for i, key in enumerate(keys)}
    return coeffs    


def evaluate_full_least_squares_3D(points_new, coeffs, degree):

    Φ_new, keys = eval_design_matrix_vectorized_3D(points_new, degree)

    coeff_vector = np.array([coeffs.get(tuple(k), 0.0) for k in keys], dtype=np.complex128)

    return Φ_new @ coeff_vector  



# 2D standard LS in rotation invariant basis

if __name__ != '__main__':
    rng_coeff = np.random.uniform(-1,1, (50, 101))
    fig = plt.figure()
    degrees = [2,5,7,9,11,15]
    target_degree = 50
    test_points = sample_spheres_uniform(20, 2)
    target =  eval_test_vectorized_batch(target_degree, rng_coeff, test_points)
    Optimal_error = []
    for n in degrees:
        max_differences = []
        # for N in tqdm(range(50, 500, 50)):
        #     points = sample_spheres_uniform(N, 2)
        #     sample = eval_test_vectorized_batch(target_degree, rng_coeff, points)
        #     DM = invar_basis_design_matrix_2D(points, n)
        #     coeffs = fit_invar_least_squares_2D(points, sample, n)
        #     LSval = eval_invar_approx_2D(coeffs, n, test_points)
        #     max_differences.append(np.linalg.norm(LSval-target))
        Trunc = eval_test_vectorized_batch(n, rng_coeff, test_points)
        
        Optimal_error.append(np.linalg.norm(Trunc-target, 2))
        #plt.plot([N for N in range(50,500,50)], max_differences, label=f'Degree {n} approximation')
    plt.yscale('log')
    plt.title('L2 Truncation error')
    plt.xlabel("Truncation degree", )
    plt.ylabel("L2 difference")
    plt.legend()
    plt.plot(degrees, Optimal_error, label='||f-\Pi f||_2')
    plt.grid(True)
    plt.legend()


if __name__ == '__main__':
    rng_coeff = np.random.uniform(-1,1, (50, 101))
    fig = plt.figure()
    degrees = [2,5,7, 10]
    target_degree = 20
    test_points = sample_spheres_uniform(20, 3)
    target =  evaluate_sum_of_basis_3D(test_points, gen_coeffs_vectorized_3D(target_degree), target_degree)
    Optimal_error = []
    for n in degrees:
        max_differences = []
        # for N in tqdm(range(50, 500, 50)):
        #     points = sample_spheres_uniform(N, 2)
        #     sample = eval_test_vectorized_batch(target_degree, rng_coeff, points)
        #     DM = invar_basis_design_matrix_2D(points, n)
        #     coeffs = fit_invar_least_squares_2D(points, sample, n)
        #     LSval = eval_invar_approx_2D(coeffs, n, test_points)
        #     max_differences.append(np.linalg.norm(LSval-target))
        Trunc = evaluate_sum_of_basis_3D(test_points, gen_coeffs_vectorized_3D(n), n)
        
        Optimal_error.append(np.linalg.norm(Trunc-target, 2))
        #plt.plot([N for N in range(50,500,50)], max_differences, label=f'Degree {n} approximation')
    plt.yscale('log')
    plt.title('L2 Truncation error for 3 particles')
    plt.xlabel("Truncation degree", )
    plt.ylabel("L2 difference")
    plt.legend()
    plt.plot(degrees, Optimal_error, label='||f-\Pi f||_2')
    plt.grid(True)
    plt.legend()





    
    

    
    
    
    
    
    
    