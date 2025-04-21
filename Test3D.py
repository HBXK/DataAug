#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 18:15:31 2025

@author: hk
"""

import numpy as np
import numpy as jnp
import jax 
jax.config.update("jax_enable_x64", True)

import s2fft
from scipy.special import sph_harm_y
import matplotlib.pyplot as plt


def sample_spheres_uniform(N):
    phi = np.random.uniform(0, 2 * np.pi, size=(N,))
    
    cos_theta = np.random.uniform(-1, 1, size=(N,))
    theta = np.arccos(cos_theta)

    points = np.stack((phi, theta), axis=-1)
    return points



def equiangular_sampling(L):

    M = 2 * L
    theta_1d = (np.arange(M)) * np.pi / M     # shape (M,)
    # longitudes
    phi_1d   = np.arange(M) * (2 * np.pi / M)       # shape (M,)

    # make 2D grids
    phi, theta = np.meshgrid(phi_1d, theta_1d, indexing='ij')
    return phi, theta


def bandlimited_interpolation_s2fft(L, points):

    phi, theta = points
    f = 1/(1 + jnp.sin(phi)**2 * jnp.cos(theta)**2)   


    flm   = s2fft.forward_jax(f, L)
    f_rec = s2fft.inverse_jax(flm, L) 

    return f, flm, f_rec


def evaluate_trig(coeffs, pts):
    L, M = coeffs.shape
    assert M == 2*L - 1, "coeffs must have shape (L, 2L-1)"

    φ = pts[:, 0]
    θ = pts[:, 1]
    N = φ.size

     # accumulate result
    f = np.zeros(N, dtype=np.complex128)

    for l in range(L+1):         
        for m in range(-l, l+1):
            a = coeffs[l,  m + (L-1)]
            if a == 0:
                continue

            Y = sph_harm_y(l, m, θ, φ)
            f   += a * Y

    return f

def evaluate_target(pts):
    return 1/(1 + jnp.sin(pts[::, 0])**2 * jnp.cos(pts[::,1])**2)   


mse = []
for i in [2**l for l in range(6)]:
    points = equiangular_sampling(i)
    f, flm, f_rec = bandlimited_interpolation_s2fft(i, points)
    test_points = sample_spheres_uniform(10)

    mse.append(np.linalg.norm(evaluate_target(test_points)-evaluate_trig(flm, test_points), ord=2))


plt.figure()
plt.plot([2**l for l in range(6)], mse, marker='o')
plt.yscale('log')    
plt.title('MSE vs L')    # add a title
plt.xlabel('L')           # x‑axis label
plt.ylabel('Mean Squared Error ')    # y‑axis label
plt.grid(True, which='both', ls='--')     # grid for both major & minor ticks
plt.show()







