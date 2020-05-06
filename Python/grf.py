import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import ConjugateGradient
from pymanopt import Problem

def cost(M_guess, X_train, X_test, f_train, f_test, kernel):

    U_train = X_train @ M_guess
    U_test = X_test @ M_guess

    G = kernel(U_train)
    b = np.linalg.solve(G, f_train)

    K_test = kernel(U_test, U_train)
    g_test = K_test @ b

    r = 0.5 * np.linalg.norm(f_test - g_test)**2
    return r

def dcost(M_guess, X_train, X_test, f_train, f_test, kernel):

    ell = kernel.get_params()['k1__k2__length_scale']
    U_train = X_train @ M_guess
    U_test = X_test @ M_guess
    N_test = X_test.shape[0]

    G = kernel(U_train)
    b = np.linalg.solve(G, f_train)
    K_test = kernel(U_test, U_train)
    g_test = K_test @ b

    inv_P = np.diag(1.0/ell**2)
    dr = np.zeros(M_guess.shape)
    for i in range(N_test):
        U_tilde = U_test[i] - U_train
        dgdu = inv_P @ U_tilde.T @ (K_test[i,:] * b)
        dy = np.outer(dgdu, X_test[i,:]).T
        assert(dy.shape == M_guess.shape)
        dr += (f_test[i] - g_test[i]) * (dy - M_guess @ dy.T @ M_guess)

    return dr

def pred(M_guess, X_train, X_test, f_train, kernel):
    U_train = X_train @ M_guess
    U_test = X_test @ M_guess

    G = kernel(U_train)
    b = np.linalg.solve(G, f_train)

    K_test = kernel(U_test, U_train)
    g_test = K_test @ b

    return g_test

def grf_fit(M0, X_train, X_test, f_train, f_test, tol = 1e-5, verbosity=0):
    last_r = 1.0
    err = 1.0
    M_guess = M0.copy()
    d, m = M0.shape

    while err > tol:
        U_train = X_train @ M_guess
        ker = 1.0 * RBF(length_scale=[1 for _ in range(m)]) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=10, alpha=.1)
        gpr.fit(U_train, f_train)

        kernel = gpr.kernel_

        my_cost = lambda M: cost(M, X_train, X_test, f_train, f_test, kernel)
        my_dcost = lambda M: dcost(M, X_train, X_test, f_train, f_test, kernel)

        manifold = Stiefel(d, m)
        problem = Problem(manifold=manifold, cost=my_cost, grad=my_dcost, verbosity=verbosity)
        solver = ConjugateGradient()
        M_new = solver.solve(problem, x=M_guess)
        M_guess = M_new.copy()

        r = cost(M_guess, X_train, X_test, f_train, f_test, kernel)

        err = np.abs(last_r - r) / r
        last_r = r

    return M_guess