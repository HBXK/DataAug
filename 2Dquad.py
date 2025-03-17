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
from tqdm import tqdm


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



def F(thetas, degree, valid_powers, alpha = 2):
    valid_powers = np.array(valid_powers)
    # ck = np.zeros((len(valid_powers),))
    # i=0
    # for k in valid_powers:
    #     ck[i] = rng[i]*np.exp(-alpha*sum(abs(ki) for ki in k))
    #     i+=1
    # z = np.exp(1j*thetas)
    # return sum(ck[i]*np.cos(np.prod(z**valid_powers[i])) for i in range(len(valid_powers)))
    ck = rng[:len(valid_powers)] * np.exp(-alpha * np.linalg.norm(valid_powers, axis=1))
    phase = valid_powers @ thetas  # shape: (|K|,) because (|K|,3) @ (3,) => (|K|,)
    complex_exps = np.exp(1j * phase)  # shape (|K|,)
    cos_terms = np.cos(complex_exps)
    return np.sum(ck * cos_terms)


def design_matrix_unit_circle(thetas, degree, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    z = np.exp(1j * thetas)
    valid_powers = np.array(valid_powers)
    # Compute each term in the polynomial using broadcasting
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)


    # Augment the data with rotations by 2*pi/k for k = 1, ..., degree
    # augmented_matrix = []
    # for k in range(1, degree + 1):
    #     rotation = 2 * np.pi*k / (degree+1)
    #     for group in thetas:
    #         rotated_group = group + rotation  # Apply the rotation
    #         z = np.exp(1j * rotated_group)  # Compute z = e^(i * (theta + rotation))
    #         row = [np.prod(z**k) for k in valid_powers]  # Compute each term in the polynomial
    #         augmented_matrix.append(row)

    # # Combine original and augmented matrices
    # design_matrix = np.vstack([design_matrix, augmented_matrix])
    
    k_values = np.arange(1, degree + 1)[:, None, None]
    rotations = 2 * np.pi * k_values / (degree + 1)
    rotated_thetas = thetas[None, :, :] + rotations
    
    z_rotated = np.exp(1j * rotated_thetas)
    augmented_matrix = np.prod(z_rotated[:, :, None, :] ** valid_powers[None, None, :, :], axis=3)
    
    return np.vstack([design_matrix] + [augmented_matrix[i] for i in range(degree)])
    



def design_matrix_orig(thetas, degree, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    #modified for invariant representation
    valid_powers = np.array(valid_powers)
    # Compute the design matrix
    z = np.exp(1j * thetas)

    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)


    return np.array(design_matrix)

def design_matrix_full(thetas, degree, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")
    
    valid_powers = np.array(valid_powers)
    
    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    # valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]

    # Compute z = e^(i * theta) for all groups at once
    z = np.exp(1j * thetas)
    
    # Compute each term in the polynomial using broadcasting
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)

    # Compute the design matrix
    # design_matrix = []
    # for group in thetas:
    #     z = np.exp(1j * group)  # Compute z = e^(i * theta) for the group
    #     row = [np.prod(z**k) for k in valid_powers]  # Compute each term in the polynomial
    #     design_matrix.append(row)


    return np.array(design_matrix)



def design_matrix_MC(thetas, degree, nb, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = np.array(valid_powers)
    z = np.exp(1j * thetas)

    # Compute the design matrix
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)


    rotations = np.random.uniform(0,2*np.pi, (nb, ))[:, None, None]
    rotated_thetas = thetas[None, :, :] + rotations
   
    z_rotated = np.exp(1j * rotated_thetas)
    augmented_matrix = np.prod(z_rotated[:, :, None, :] ** valid_powers[None, None, :, :], axis=3)
   
    return np.vstack([design_matrix] + [augmented_matrix[i] for i in range(nb)])
   


def augment_target_values(y, degree):

    y = np.asarray(y)
    # Replicate y (degree + 1) times: 1 original + degree augmentations
    augmented_y = np.tile(y, degree + 1)
    return augmented_y


def least_square(X, y):

    X_conj_transpose = np.conjugate(X.T)  
    w = np.linalg.solve(X_conj_transpose @ X, X_conj_transpose @ y)
    return w

def least_squares_qr(X, y):

    # Perform full QR decomposition of X
    Q, R_full = np.linalg.qr(X, mode='complete')

    # Extract the upper triangular part R (n x n)
    n = X.shape[0]
    R = R_full[::, :n]  # Take only the first n rows of R_full

    # Compute Q^T * y
    QTy = np.dot(Q.T, y)
    
    # Solve R * beta = QTy_relevant
    beta = np.linalg.solve(R, QTy)
    
    return beta




def evaluate_polynomial(theta1, theta2, theta3, coefficients, degree, valid_powers):

    # Compute z = e^(i * theta) for the input angles
    z = np.exp(1j * np.array([theta1, theta2, theta3]))

    # Generate all valid powers of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    # valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]

    # Evaluate the polynomial
    valid_powers = np.array(valid_powers)
    
    value = np.sum(coefficients * np.prod(z**valid_powers, axis=1))

    return value

def eval_poly_invar(theta1, theta2, theta3, coefficients, degree, valid_powers):
    # Compute z = e^(i * theta) for the input angles
    z = np.exp(1j * np.array([theta1, theta2, theta3]))

    # Generate all valid powers of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = np.array(valid_powers)

    # Evaluate the polynomial
    value = np.sum(coefficients * np.prod(z**valid_powers, axis=1))

    return value




def psym(theta1,theta2,theta3, coefficients, degree, valid_powers):
    sym_value = 0
    for k in range(1, degree + 1):
        rotation = 2 * np.pi*k / (degree+1)

        rotated_theta1 = theta1 + rotation
        rotated_theta2 = theta2 + rotation
        rotated_theta3 = theta3 + rotation

        rotated_value = evaluate_polynomial(rotated_theta1, rotated_theta2, rotated_theta3, coefficients, degree, valid_powers)

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
    gram_matrices = gram_matrix(N, thetas)

    # Compute target values
    y = f(N,gram_matrices)

    # Create the design matrix and augment the target values
    X = design_matrix_unit_circle(thetas, degree)
    #X = design_matrix_MC(thetas, degree, nb)

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





rng = np.random.uniform(-1,1, size=(50000,))




    
if __name__!= "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Invariant Approximation")
    plt.xlabel("Data points")
    plt.ylabel("||Prediction-test||")
    degrees = [3,5,7,8,10,11]
    degreeF = 20
    test_points = points_circle(15)
    invar_approx= []
    for degree in degrees:
        max_differences = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree and sum(ki for ki in k)==0]
        valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]

        for N in tqdm([k for k in range(10,500,20)]):
            thetas = points_circle(N)
            gram_matrices = gram_matrix(N, thetas)
            X = design_matrix_orig(thetas, degree, valid_powers)
            #y = f(N, gram_matrices)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff = least_square(X, y)
            poly_values =[] 
            for theta1, theta2, theta3 in test_points:
                poly_values.append(eval_poly_invar(theta1, theta2, theta3, coeff, degree, valid_powers))
            #y_test = f(15, gram_matrix(15, test_points))
            
            y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
            max_differences.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))
            # if N==290:
            #     invar_approx.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))

        plt.plot([i for i in range(10,500,20)], max_differences, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()

    
if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Standard Approximation")
    plt.xlabel("Data points")
    plt.ylabel("||Prediction-test||")
    degrees = [3,5,7,8,10,11]
    degreeF = 20
    test_points = points_von_mises(15, 3)
    standard_approx = []
    for degree in degrees:
        max_differences = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
        for N in tqdm([k for k in range(500,6000,500)]):
            thetas = points_von_mises(N, 3)
            gram_matrices = gram_matrix(N, thetas)
            X = design_matrix_full(thetas, degree, valid_powers)
            #y = f(N, gram_matrices)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff = least_square(X, y)
            poly_values =[] 
            for theta1, theta2, theta3 in test_points:
                poly_values.append(evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers))
            #y_test = f(15, gram_matrix(15, test_points))
            
            y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
            max_differences.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))
            # if N==3900:
            #     standard_approx.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))

        plt.plot([i for i in [k for k in range(500, 6000,500)]], max_differences, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Augmented Approximation")
    plt.xlabel("Data points")
    plt.ylabel("||Prediction-test||")
    aug_approx = []
    degrees = [3,5,7,8,10,11]
    degreeF = 20
    test_points = points_von_mises(15, 3)
    for degree in degrees:
        max_differences = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
        for N in tqdm([k for k in range(10,500,20)]):
            thetas =  points_von_mises(N, 3)
            gram_matrices = gram_matrix(N, thetas)
            X = design_matrix_unit_circle(thetas, degree, valid_powers)
            #y = f(N, gram_matrices)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            y = augment_target_values(y, degree)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff = least_square(X, y)
            poly_values =[] 
            for theta1, theta2, theta3 in test_points:
                poly_values.append(evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers))
            #y_test = f(15, gram_matrix(15, test_points))
            
            y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
            max_differences.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))
            # if N==3900:
            #     standard_approx.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))

        plt.plot([i for i in range(00,500,20)], max_differences, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Augmented Approximation Psym")
    plt.xlabel("Data points")
    plt.ylabel("||Prediction-test||")
    degrees = [3,5,7,8,10,11]
    degreeF = 20
    test_points = points_von_mises(15, 3)
    for degree in degrees:
        aug_approx = []
        max_differences = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
        for N in tqdm([k for k in range(10,500,20)]):
            thetas =  points_von_mises(N, 3)
            gram_matrices = gram_matrix(N, thetas)
            X = design_matrix_unit_circle(thetas, degree, valid_powers)
            #y = f(N, gram_matrices)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            y = augment_target_values(y, degree)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff = least_square(X, y)
            max_difference = 0
            for theta1, theta2, theta3 in test_points:
                psym_value = psym(theta1, theta2, theta3, coeff, degree, valid_powers)
                poly_value = evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers)
                difference = abs(psym_value - poly_value)
                max_difference = max(max_difference, difference)     
            aug_approx.append(max_difference)
                
                # if N==3900:
            #     standard_approx.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))

        plt.plot([i for i in range(10,500,20)], aug_approx, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Standard Approximation PSym")
    plt.xlabel("Data points")
    plt.ylabel("||Prediction-test||")
    degrees = [3,5,7,8,10,11]
    degreeF = 20
    test_points = points_von_mises(15, 3)
    for degree in degrees:
        max_differences = []
        standard_approx = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
        for N in tqdm([k for k in range(500,10000,500)]):
            thetas = points_von_mises(N, 3)
            gram_matrices = gram_matrix(N, thetas)
            X = design_matrix_full(thetas, degree, valid_powers)
            #y = f(N, gram_matrices)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff = least_square(X, y)
            max_difference = 0
            for theta1, theta2, theta3 in test_points:
                psym_value = psym(theta1, theta2, theta3, coeff, degree, valid_powers)
                poly_value = evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers)
                difference = abs(psym_value - poly_value)
                max_difference = max(max_difference, difference)     
            standard_approx.append(max_difference)
            # if N==3900:
            #     standard_approx.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))

        plt.plot([i for i in [k for k in range(500, 10000,500)]], standard_approx, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()





if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Augmented MC-Approximation")
    plt.xlabel("Nb of rotations")
    plt.ylabel("||Prediction-test||")
    aug_approx = []
    degrees = [3,5,7,8,10,11]
    degreeF = 20
    test_points = points_von_mises(15, 3)
    for degree in degrees:
        max_differences = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
        thetas = points_von_mises(100, 3)
        gram_matrices = gram_matrix(100, thetas)
        for nb in tqdm([k for k in range(50, 500, 50)]):
            X = design_matrix_MC(thetas, degree, nb, valid_powers)
            #y = f(N, gram_matrices)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            y = augment_target_values(y, nb)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff = least_square(X, y)
            poly_values =[] 
            for theta1, theta2, theta3 in test_points:
                poly_values.append(evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers))
            #y_test = f(15, gram_matrix(15, test_points))
            
            y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
            max_differences.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))
            # if degree==21:
            #     standard_approx.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))

        plt.plot([i for i in range(50,500, 50)], max_differences, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Augmented MC-Approximation")
    plt.xlabel("Nb of rotations")
    plt.ylabel("||Prediction-test||")
    aug_approx = []
    degrees = [3,5,7,8,10,11]
    degreeF = 20
    test_points = points_von_mises(15, 3)
    for degree in degrees:
        max_differences = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
        thetas = points_von_mises(100, 3)
        MC_approx  =[]
        gram_matrices = gram_matrix(100, thetas)
        for nb in tqdm([k for k in range(50, 1000, 100)]):
            X = design_matrix_MC(thetas, degree, nb, valid_powers)
            #y = f(N, gram_matrices)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            y = augment_target_values(y, nb)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff = least_square(X, y)
            poly_values =[] 
            for theta1, theta2, theta3 in test_points:
                poly_values.append(evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers))
            #y_test = f(15, gram_matrix(15, test_points))
            
            max_difference = 0
            for theta1, theta2, theta3 in test_points:
                psym_value = psym(theta1, theta2, theta3, coeff, degree, valid_powers)
                poly_value = evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers)
                difference = abs(psym_value - poly_value)
                max_difference = max(max_difference, difference)     
            MC_approx.append(max_difference)
        plt.plot([i for i in range(50,1000, 100)], MC_approx, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()








if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Augmented Approximation")
    plt.xlabel("Data points")
    plt.ylabel("||Prediction-test||")
    aug_approx = []
    degrees = [2,5,8,11]
    degreeF = 20
    test_points = points_von_mises(15, 3)
    for degree in degrees:
        max_differences = []
        for N in tqdm(range(20,350,20)):
            thetas = points_von_mises(N, 3)
            gram_matrices = gram_matrix(N, thetas)
            X = design_matrix_unit_circle(thetas, degree)
            #y = f(N, gram_matrices)
            y = []
            for theta in thetas:
                y.append(F(theta, degreeF))
            y = np.array(y)
            y = augment_target_values(y, degree)
            coeff = least_square(X, y)
            poly_values =[] 
            for theta1, theta2, theta3 in test_points:
                poly_values.append(evaluate_polynomial(theta1, theta2, theta3, coeff, degree))
            #y_test = f(15, gram_matrix(15, test_points))
            y_test = []
            for theta in test_points:
                y_test.append(F(theta, degreeF))
            y_test = np.array(y_test)
            max_differences.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))
            if N==340:
                aug_approx.append(1/np.sqrt(15)*np.linalg.norm(poly_values-y_test))

        plt.plot([i for i in range(20,350,20)], max_differences, marker="o", label=f'{degree}')
    plt.legend()
    plt.grid()
    plt.show()

# plt.figure()
# plt.yscale("log")
# plt.title("Comparing different approximations")
# plt.xlabel("degree")
# plt.ylabel("Prediction error")
# plt.plot( [3,5,7,8,10,11], invar_approx, marker = "o", label="Invariant")
# plt.plot( [3,5,7,8,10,11], standard_approx, marker = "o", label="Standard (Has 10x more data)")
# plt.plot( [3,5,7,8,10,11], aug_approx, marker = "o", label="Augmented")
# plt.legend()
# plt.grid()
# plt.show()




if __name__ != "__main__":
    max_differences = []
    degrees = range(1, 12)
    N=100
    for degree in degrees:
        rotations = np.random.uniform(0,2*np.pi, degree+1)
        max_diff = compute_max_difference(N, degree, degree)
        max_differences.append(max_diff)
    # max_differences = []
    # degree = 4
    # nbs = range(10,20)
    # for nb in nbs:
    #     rotations = np.random.uniform(0,2*np.pi, (nb,))
    #     max_diff = compute_max_difference(N, degree,nb)
    #     max_differences.append(max_diff)
    # Plot the results
    plt.figure()
    plt.yscale("log")
    plt.plot(degrees, max_differences, marker="o")
    plt.title("Max Difference vs number of rotations")
    plt.xlabel("Number of rotations")
    plt.ylabel("Max difference")
    plt.grid()
    plt.show()









        