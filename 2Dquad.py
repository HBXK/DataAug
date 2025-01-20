#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:25:38 2024

@author: hk
"""
import numpy as np
from scipy.linalg import expm, eigh
from scipy.stats import vonmises
from itertools import product
import matplotlib.pyplot as plt



def points_circle(N):
    return np.random.uniform(0, 2 * np.pi, (N, 3))

def points_von_mises(N, kappa):

    return vonmises.rvs(kappa, size=(N, 3)) * 2 * np.pi


def gram_matrix(N, points):

    gram_matrices = np.zeros((N, 3, 3))

    for i in range(N):
        angles = points[i]
        for j in range(3):
            for k in range(3):
                gram_matrices[i, j, k] = np.cos(angles[j] - angles[k])

    return gram_matrices


def matrix_exponential_diagonalization(G):

    eigenvalues, eigenvectors = eigh(G)  # Compute eigenvalues and eigenvectors
    exp_diag = np.diag(np.exp(eigenvalues))  # Exponential of the eigenvalues
    return exp_diag   # Reconstruct the exponential


def f(N, Gs):
    traces = np.zeros(N)
    for i in range(N):
        exp_G = matrix_exponential_diagonalization(Gs[i])
        traces[i] = np.trace(exp_G)
    return traces


def design_matrix_unit_circle(thetas, degree):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) == degree]

    # Compute the design matrix
    design_matrix = []
    for group in thetas:
        z = np.exp(1j * group)  # Compute z = e^(i * theta) for the group
        row = [np.prod(z**k) for k in valid_powers]  # Compute each term in the polynomial
        design_matrix.append(row)

    # Augment the data with rotations by 2*pi/k for k = 1, ..., degree
    augmented_matrix = []
    for k in range(1, degree + 1):
        rotation = 2 * np.pi*k / (degree+1)
        for group in thetas:
            rotated_group = group + rotation  # Apply the rotation
            z = np.exp(1j * rotated_group)  # Compute z = e^(i * (theta + rotation))
            row = [np.prod(z**k) for k in valid_powers]  # Compute each term in the polynomial
            augmented_matrix.append(row)

    # Combine original and augmented matrices
    design_matrix = np.vstack([design_matrix, augmented_matrix])

    return np.array(design_matrix)



def design_matrix_orig(thetas, degree):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    #modified for invariant representation
    valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) == degree and sum(ki for ki in k)==0]

    # Compute the design matrix
    design_matrix = []
    for group in thetas:
        z = np.exp(1j * group)  # Compute z = e^(i * theta) for the group
        row = [np.prod(z**k) for k in valid_powers]  # Compute each term in the polynomial
        design_matrix.append(row)


    return np.array(design_matrix)




def design_matrix_MC(thetas, degree, nb):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) == degree]

    # Compute the design matrix
    design_matrix = []
    for group in thetas:
        z = np.exp(1j * group)  # Compute z = e^(i * theta) for the group
        row = [np.prod(z**k) for k in valid_powers]  # Compute each term in the polynomial
        design_matrix.append(row)

    # Augment the data with rotations by 2*pi/k for k = 1, ..., degree
    augmented_matrix = []
    for k in range(nb):
        rotation = rotations[k]
        for group in thetas:
            rotated_group = group + rotation  # Apply the rotation
            z = np.exp(1j * rotated_group)  # Compute z = e^(i * (theta + rotation))
            row = [np.prod(z**i) for i in valid_powers]  # Compute each term in the polynomial
            augmented_matrix.append(row)

    # Combine original and augmented matrices
    design_matrix = np.vstack([design_matrix, augmented_matrix])

    return np.array(design_matrix)


def augment_target_values(y, degree):

    y = np.asarray(y)
    # Replicate y (degree + 1) times: 1 original + degree augmentations
    augmented_y = np.tile(y, degree + 1)
    return augmented_y


def least_square(X, y):

    X_conj_transpose = np.conjugate(X.T)  
    w = np.linalg.solve(X_conj_transpose @ X, X_conj_transpose @ y)
    return w


def evaluate_polynomial(theta1, theta2, theta3, coefficients, degree):

    # Compute z = e^(i * theta) for the input angles
    z = np.exp(1j * np.array([theta1, theta2, theta3]))

    # Generate all valid powers of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) == degree]

    # Evaluate the polynomial
    value = sum(coeff * np.prod(z**k) for coeff, k in zip(coefficients, valid_powers))

    return value

def eval_poly_invar(theta1, theta2, theta3, coefficients, degree):

    # Compute z = e^(i * theta) for the input angles
    z = np.exp(1j * np.array([theta1, theta2, theta3]))

    # Generate all valid powers of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) == degree and sum(ki for ki in k)==0]

    # Evaluate the polynomial
    value = sum(coeff * np.prod(z**k) for coeff, k in zip(coefficients, valid_powers))

    return value




def psym(theta1, theta2, theta3, coefficients, degree):

    sym_value = 0
    for k in range(1, degree + 1):
        rotation = 2 * np.pi*k / (degree+1)

        rotated_theta1 = theta1 + rotation
        rotated_theta2 = theta2 + rotation
        rotated_theta3 = theta3 + rotation

        rotated_value = evaluate_polynomial(rotated_theta1, rotated_theta2, rotated_theta3, coefficients, degree)

        sym_value += rotated_value

    sym_value /= degree

    return sym_value

def psym_MC(theta1, theta2, theta3, coefficients, degree, nb):

    sym_value = 0
    for k in range(nb):
        rotation = rotations[k]

        rotated_theta1 = theta1 + rotation
        rotated_theta2 = theta2 + rotation
        rotated_theta3 = theta3 + rotation

        rotated_value = evaluate_polynomial(rotated_theta1, rotated_theta2, rotated_theta3, coefficients, degree)

        sym_value += rotated_value

    sym_value /= degree

    return sym_value


def compute_max_difference(N, degree, nb):

    # Generate points and compute the Gram matrix
    thetas = points_circle(N)
    gram_matrices = gram_matrix(thetas)

    # Compute target values
    y = f(gram_matrices)

    # Create the design matrix and augment the target values
    #X = design_matrix_unit_circle(thetas, degree)
    X = design_matrix_MC(thetas, degree, nb)

    augmented_y = augment_target_values(y, nb)

    # Compute least squares coefficients
    coefficients = least_square(X, augmented_y)

    # Randomly sample 30 more points
    test_points = points_circle(30)

    # Compute the max difference between psym and evaluate_polynomial
    max_difference = 0
    for theta1, theta2, theta3 in test_points:
        psym_value = psym(theta1, theta2, theta3, coefficients, degree)
        poly_value = evaluate_polynomial(theta1, theta2, theta3, coefficients, degree)
        difference = abs(psym_value - poly_value)
        max_difference = max(max_difference, difference)

    return max_difference



if __name__ == "__main__":
    max_differences = []
    degree = 4
    test_points = points_circle(50)
    for N in range(200,500,50):
        thetas = points_circle(N)
        gram_matrices = gram_matrix(N, thetas)
        X = design_matrix_orig(thetas, degree)
        y = f(N, gram_matrices)
        coeff = least_square(X, y)
        max_difference = 0
        poly_values =[] 
        for theta1, theta2, theta3 in test_points:
            poly_values.append(eval_poly_invar(theta1, theta2, theta3, coeff, degree))
        y_test = f(50, gram_matrix(50, test_points))
        max_differences.append(np.linalg.norm(poly_values-y_test))

    plt.figure()
    plt.yscale("log")
    plt.plot([i for i in range(200,500,50)], max_differences, marker="o")
    plt.title("Invariant representation error")
    plt.xlabel("Data points")
    plt.ylabel("||Prediction||")
    plt.grid()
    plt.show()



if __name__ != "__main__":
    # max_differences = []
    # degrees = range(1, 11)

    # for degree in degrees:
    #     max_diff = compute_max_difference(N, degree)
    #     max_differences.append(max_diff)
    max_differences = []
    degree = 4
    nbs = range(10,20)
    for nb in nbs:
        rotations = np.random.uniform(0,2*np.pi, (nb,))
        max_diff = compute_max_difference(N, degree,nb)
        max_differences.append(max_diff)
    # Plot the results
    plt.figure()
    plt.yscale("log")
    plt.plot(nbs, max_differences, marker="o")
    plt.title("Max Difference vs number of rotations")
    plt.xlabel("Number of rotations")
    plt.ylabel("Max difference")
    plt.grid()
    plt.show()









        