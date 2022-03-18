import numpy as np
from .correlations import pearson_coeff
from .normalizations import sum_normalization, minmax_normalization
import copy
import sys
import itertools


# equal weighting
def equal_weighting(X, types):
    """
    Calculate criteria weights using objective Equal weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray

    Returns
    -------
        ndarray
            vector of criteria weights
    """
    N = np.shape(X)[1]
    return np.ones(N) / N


# Entropy weighting
def entropy_weighting(X, types):
    """
    Calculate criteria weights using objective Entropy weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray

    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # normalize the decision matrix with sum_normalization method from normalizations as for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    # Transform negative values in decision matrix X to positive values
    pij = np.abs(pij)
    m, n = np.shape(pij)
    H = np.zeros((m, n))

    # Calculate entropy
    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))

    # Calculate degree of diversification
    d = 1 - h

    # Set w as the degree of importance of each criterion
    w = d / (np.sum(d))
    return w


# Standard Deviation weighting
def std_weighting(X, types):
    """
    Calculate criteria weights using objective Standard deviation weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # stdv = np.std(X, axis = 0)
    # Calculate the standard deviation of each criterion in decision matrix
    stdv = np.sqrt((np.sum(np.square(X - np.mean(X, axis = 0)), axis = 0)) / (X.shape[0]))
    # Calculate criteria weights by dividing the standard deviations by their sum
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(X, types):
    """
    Calculate criteria weights using objective CRITIC weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # Normalize the decision matrix using Minimum-Maximum normalization minmax_normalization from normalizations as for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    # Calculate the standard deviation
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    # Calculate correlation coefficients of all pairs of columns of normalized decision matrix
    correlations = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        correlations[i, j] = pearson_coeff(x_norm[:, i], x_norm[:, j])

    # Calculate the difference between 1 and calculated correlations
    difference = 1 - correlations
    # Multiply the difference by the standard deviation
    C = std * np.sum(difference, axis = 0)
    # Calculate the weights by dividing vector with C by their sum
    w = C / np.sum(C)
    return w


# Gini coefficient-based weighting
def gini_weighting(X, types):
    """
    Calculate criteria weights using objective Gini coefficient-based weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    m, n = np.shape(X)
    G = np.zeros(n)
    # Calculate the Gini coefficient for decision matrix X
    # iteration over criteria j = 1, 2, ..., n
    for j in range(0, n):
        # iteration over alternatives i = 1, 2, ..., m
        Yi = np.zeros(m)
        if np.mean(X[:, j]) != 0:
            for i, k in itertools.product(range(m), range(m)):
                Yi[i] += np.abs(X[i, j] - X[k, j]) / (2 * m**2 * (np.sum(X[:, j]) / m))
        else:
            for i, k in itertools.product(range(m), range(m)):
                Yi[i] += np.abs(X[i, j] - X[k, j]) / (m**2 - m)

        G[j] = np.sum(Yi)
    # calculate and return the criteria weights by dividing the vector of Gini coefficients by their sum
    return G / np.sum(G)


# MEREC weighting
def merec_weighting(matrix, types):
    """
    Calculate criteria weights using objective MEREC weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    X = copy.deepcopy(matrix)
    m, n = X.shape
    # Transform negative values in decision matrix X to positive values
    X = np.abs(X)
    # Normalize the decision matrix X with linear normalization method
    norm_matrix = np.zeros(X.shape)
    norm_matrix[:, types == 1] = np.min(X[:, types == 1], axis = 0) / X[:, types == 1]
    norm_matrix[:, types == -1] = X[:, types == -1] / np.max(X[:, types == -1], axis = 0)
    
    # Calculate the overall performance of the values in normalized matrix using a logarithmic measure with equal criteria weights
    S = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_matrix)), axis = 1)))

    # Calculate the performance of the alternatives by removing each criterion using the logarithmic measure
    Sp = np.zeros(X.shape)

    for j in range(n):
        norm_mat = np.delete(norm_matrix, j, axis = 1)
        Sp[:, j] = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_mat)), axis = 1)))

    # Calculate the summation of absolute deviations
    E = np.sum(np.abs(Sp - S.reshape(-1, 1)), axis = 0)

    # calculate the final weights of the criteria
    w = E / np.sum(E)
    return w


# Statistical Variance weighting
def stat_var_weighting(X, types):
    """
    Calculate criteria weights using objective Statistical variance weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """

    # Normalize the decision matrix X with minmax_normalization method from normalizations
    criteria_type = np.ones(np.shape(X)[1])
    xn = minmax_normalization(X, criteria_type)
    # Calculate the statistical variance for each criterion
    v = np.mean(np.square(xn - np.mean(xn, axis = 0)), axis = 0)
    # Calculate the final weights of the criteria
    w = v / np.sum(v)
    return w


# CILOS weighting
def cilos_weighting(X, types):
    """
    Calculate criteria weights using objective CILOS weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    xr = copy.deepcopy(X)
    # Convert negative criteria to positive criteria
    xr[:, types == -1] = np.min(X[:, types == -1], axis = 0) / X[:, types == -1]
    # Normalize the decision matrix xr using the sum normalization method
    xn = xr / np.sum(xr, axis = 0)
    
    # Calculate the square matrix
    A = xn[np.argmax(xn, axis = 0), :]
    
    # Calculate relative impact loss matrix
    pij = np.zeros((X.shape[1], X.shape[1]))
    for j, i in itertools.product(range(X.shape[1]), range(X.shape[1])):
        pij[i, j] = (A[j, j] - A[i, j]) / A[j, j]

    # Determine the weight system matrix
    F = np.diag(-np.sum(pij - np.diag(np.diag(pij)), axis = 0)) + pij
    # Calculate the criterion impact loss weight
    # The criteria weights q are determined from the formulated homogeneous linear system of equations
    # AA is the vector near 0
    AA = np.zeros(F.shape[0])
    # To determine the value of A we assume that the first element of A is close to 0 while others are zeros
    AA[0] = sys.float_info.epsilon
    # Solve the system equation
    q = np.linalg.inv(F).dot(AA)
    # Calculate and return the final weights of the criteria
    return q / np.sum(q)


# IDOCRIW weighting
def idocriw_weighting(X, types):
    """
    Calculate criteria weights using objective IDOCRIW weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # Calculate the Entropy weights
    q = entropy_weighting(X, types)
    # Calculate the CILOS weights
    w = cilos_weighting(X, types)
    # Aggregate the weight value of the attributes considering Entropy and CILOS weights
    weights = (q * w) / np.sum(q * w)
    return weights


# Angle weighting
def angle_weighting(X, types):
    """
    Calculate criteria weights using objective Angle weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    m, n = X.shape
    # Normalize the decision matrix X using sum_normalization method from normalizations
    X = sum_normalization(X, types)
    # Calculate elements of additional column (the reference attribute) which are qual to 1 / m
    B = np.ones(m) * (1 / m)
    # Calculate the angle between attraibutes in decision matrix X and the reference attribute
    u = np.arccos(np.sum(X / m, axis = 0) / (np.sqrt(np.sum(X ** 2, axis = 0)) * np.sqrt(np.sum(B ** 2))))
    # Calculate the final angle weights for each criterion
    w = u / np.sum(u)
    return w


# Coeffcient of variation weighting
def coeff_var_weighting(X, types):
    """
    Calculate criteria weights using objective Coefficient of variation weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    m, n = X.shape
    # Normalize the decisionmatrix X with sum_normalization method from normalizations
    criteria_types = np.ones(n)
    B = sum_normalization(X, criteria_types)

    # Calculate the standard deviation of each column
    Bm = np.sum(B, axis = 0) / m
    std = np.sqrt(np.sum(((B - Bm)**2), axis = 0) / (m - 1))

    # Calculate the Coefficient of Variation for each criterion
    ej = std / Bm
    # Calculate the weights for each criterion
    w = ej / np.sum(ej)
    return w