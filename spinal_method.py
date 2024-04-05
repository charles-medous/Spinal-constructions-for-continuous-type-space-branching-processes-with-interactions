"""Define the function that generates a trajectory using the spinal method.

Usage:
======
    Use 'trajectory' function to compare running times and estimates of x_bar.
    The other functions are the succesives steps that build up the trajectory.
"""
import numpy as np


def spinal_division_times(parameters):
    """Generate the random division times t_div of a binary tree.

    For a binary tree stopped at time T and of constant division rate r_psi,
    computes the values that only depends on the population size n and on
    the division times t_div.

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features

    Returns
    -------
    numpy array
        The list of division times from 0 to time parameters.T
    int
        The number n of individuals in the population at time parameters.T
    float
        The integral on [0, parameters.T] of the population size n
    float
        The integral on [0, parameters.T] of the square population size n^2
    """
    n_thresh = (np.exp(parameters.r_psi) - 1) * (parameters.N_0) 
    # Division time simulation for expected small populations
    if n_thresh < 30:
        t_div = [0]
        n = parameters.N_0
        int_n = 0
        int_n_square = 0   
        while t_div[-1] < parameters.T:
            delta_t = np.random.exponential(1 / (parameters.r_psi * n))
            if t_div[-1] + delta_t < parameters.T:
                t_div.append(t_div[-1] + delta_t)
            else:
                break
            int_n += n * delta_t
            int_n_square += delta_t * n ** 2
            n += 1
        # Computation of values
        int_n += n * (parameters.T - t_div[-1])
        int_n_square += n ** 2 * (parameters.T - t_div[-1])
        # Adding the ending time in the list
        t_div.append(parameters.T)
        return(t_div, int(n), int_n, int_n_square)

    # Division time simulation for expected big populations
    else:
        n_max = int(np.exp(1.5 + parameters.r_psi) * (parameters.N_0 + 1))
        t_div = np.zeros(n_max + 1 - parameters.N_0)
        t_div[1:] = np.cumsum(
            np.random.exponential(1 / (parameters.r_psi *
                                       np.arange(parameters.N_0, n_max))))
        idx_last = np.argmax(t_div > parameters.T)
        if idx_last == 0 or idx_last == n_max:
            idx_last = n_max + 1 - parameters.N_0
            t_div = np.append(t_div, parameters.T)
        t_div[idx_last] = parameters.T
        n = parameters.N_0 + idx_last - 1
        sum_times = np.sum(t_div[: idx_last])
        int_n = n * parameters.T - sum_times
        int_n_square = (n ** 2 * parameters.T - (2 * parameters.N_0 - 1) *
                        sum_times - 2 * sum(np.cumsum(
                            t_div[idx_last - 1: 0: - 1])))
        return(t_div[: idx_last + 1], int(n), int_n, int_n_square)


def spinal_division_values(parameters, n):
    """Construct the squeleton of the binary tree of division timees t_div.

    Generates at each branching times the index idx_loss of the individual
    randomly chosen to jump and the distribution Lambda of the traits at birth.

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features
    int
        The size of the population at time T

    Returns
    -------
    numpy array
        The list of traits distribution from 0 to time parameters.T
    numpy array
        The list of branching indexes from 0 to time parameters.T
    """
    # Values of lambda at each division event
    lambd = (np.random.beta(parameters.alpha - 1, parameters.alpha - 1,
                            int(n - parameters.N_0)))
    # index of the individuals that divided at each division event
    idx_div = np.random.randint(np.arange(parameters.N_0, n))
    return(lambd, idx_div)


def spine_initial_choice(parameters):
    """Choose the initial spinal individual.

    The spine will always be the first individual in the population,
    that of index 0. Every branching events of the spine happen when
    idx_div == 0 and the new spines will be further chosen in another function.

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features

    Returns
    -------
    none.
        Modify 'in place' the list of initial traits of the individuals in
        order to put the spine in the first position.
    """
    v = 0
    idx_spine = 0
    u = np.random.uniform() * np.sum(parameters.z_0)
    while v < u:
        v += parameters.z_0[idx_spine]
        idx_spine += 1
    idx_spine += -1
    parameters.z_0[[0, idx_spine]] = parameters.z_0[[idx_spine, 0]]
    return()


def spinal_loss_events(parameters, t_div, n):
    """Generate the random loss events of the binary tree of div times t_div.

    Each event generates the branching time, the index of the branching
    individual and the new trait factor theta. The events for the
    individuals outside the spine and for the spine are separatly generated.

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features
    numpy array
        The list of division times from 0 to time parameters.T
    int
        The size of the population at time T

    Returns
    -------
    numpy array
        The list of loss times outside the spine
    numpy array
        The list of indexes of individuals outside the spine that branched
    numpy array
        The list of factors theta for losses outside the spine
    numpy array
        The list of loss times for the spine (always of index 0)
    numpy array
        The list of factors theta for the spine losses
    """
    # Loss events outside the spine
    t_loss = []
    idx_loss = []
    theta = []
    for i in range(int(n - parameters.N_0) + 1):
        counter_loss = 0
        t_l = [t_div[i]]
        while t_l[- 1] < t_div[i + 1]:
            temp = (np.random.exponential(1 /
                                          (parameters.d * parameters.K_loss *
                                           (parameters.N_0 + i) *
                                           (parameters.N_0 + i - 1))))
            if t_l[- 1] + temp < t_div[i + 1]:
                t_l.append(t_l[- 1] + temp)
                counter_loss += 1
            else:
                break
        if counter_loss > 0:
            idx_loss.append(np.random.randint(1, parameters.N_0 + i,
                                              size=counter_loss))
            theta.append(np.random.beta(parameters.a - 1,
                                        parameters.b, size=counter_loss))
        else:
            idx_loss.append([])
            theta.append([])
        t_loss.append(t_l)

    # Loss events for the spine
    t_loss_star = []
    theta_star = []
    for i in range(int(n - parameters.N_0) + 1):
        counter_loss = 0
        t_l = [t_div[i]]
        while t_l[- 1] < t_div[i + 1]:
            temp = np.random.exponential(1 /
                                         (parameters.d * (parameters.N_0 + i)))
            if t_l[- 1] + temp < t_div[i + 1]:
                t_l.append(t_l[- 1] + temp)
                counter_loss += 1
            else:
                break
        if counter_loss > 0:
            theta_star.append(np.random.beta(parameters.a, parameters.b,
                                             size=counter_loss))
        else:
            theta_star.append([])
        t_loss_star.append(t_l)
    return(t_loss, idx_loss, theta, t_loss_star, theta_star)


def spinal_individual_traits(parameters, t_div, n, lambd, idx_div, t_loss,
                             idx_loss, theta, t_loss_star, theta_star):
    """Generate the random loss events of the binary tree of div times t_div.

    Each event generates the branching time, the index of the branching
    individual and the new trait factor theta. The events for the individuals
    outside the spine and for the spine are separatly generated.

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features
    numpy array
        The list of division times from 0 to time parameters.T
    int
        The size of the population at time T

    Returns
    -------
    numpy array
        The list of loss times outside the spine
    numpy array
        The list of indexes of individuals outside the spine that branched
    numpy array
        The list of factors theta for losses outside the spine
    numpy array
        The list of loss times for the spine (always of index 0)
    numpy array
        The list of factors theta for the spine losses
    """
    # Initialization
    z_t = np.zeros(n)  # array with the individual types
    z_t[0: parameters.N_0] = parameters.z_0  # intialization
    int_b = 0  # integrate term of total mass

    # Until the last division event
    for i in range(int(n - parameters.N_0)):
        # Integrale of biomass computation
        int_b += (sum(z_t) / parameters.mu *
                  (np.exp(parameters.mu * (t_div[i + 1] - t_div[i])) - 1))
        # Non-spinal contribution to the integral
        for j in range(len(idx_loss[i][:])):
            int_b -= ((1 - theta[i][j]) / parameters.mu * z_t[idx_loss[i][j]] *
                      (np.exp(parameters.mu * (t_div[i + 1] - t_div[i])) -
                       np.exp(parameters.mu * (t_loss[i][j + 1] - t_div[i]))))
        # Spinal contribution to the integral
        for j in range(len(theta_star[i][:])):
            int_b -= ((1 - theta_star[i][j]) / parameters.mu * z_t[0] *
                      (np.exp(parameters.mu * (t_div[i + 1] - t_div[i])) -
                       np.exp(parameters.mu * (t_loss_star[i][j + 1] -
                                               t_div[i]))))

        # Outside spine loss events
        for j in range(len(idx_loss[i][:])):
            z_t[idx_loss[i][j]] *= theta[i][j]
        # Spine loss events
        for thet in theta_star[i][:]:
            z_t[0] *= thet
        # Division events
        (z_t[idx_div[i]], z_t[parameters.N_0 + i]) = \
            (lambd[i] * z_t[idx_div[i]], (1 - lambd[i]) * z_t[idx_div[i]])
        if idx_div[i] == 0:  # if spinal branching
            u = np.random.uniform()
            if u > lambd[i]:  # if the new spine is the second one
                z_t[[0, parameters.N_0 + i]] = z_t[[parameters.N_0 + i, 0]]
        # Deterministic growth between jumps
        for j in range(len(z_t)):
            z_t[j] *= np.exp(parameters.mu*(t_div[i + 1] - t_div[i]))

    # From the last division event to the simulation time parameters.T
    # Integrale of biomass computation
    int_b += (sum(z_t) / parameters.mu *
              (np.exp(parameters.mu * (t_div[int(n - parameters.N_0) + 1] -
                                       t_div[int(n - parameters.N_0)])) - 1))
    for j in range(len(idx_loss[int(n - parameters.N_0)][:])):  # Outside the spine
        int_b -= ((1 - theta[int(n - parameters.N_0)][j]) / parameters.mu *
                  z_t[idx_loss[int(n - parameters.N_0)][j]] *
                  (np.exp(parameters.mu * (t_div[int(n - parameters.N_0) + 1] -
                                           t_div[int(n - parameters.N_0)])) -
                   np.exp(parameters.mu * (t_loss[int(n - parameters.N_0)][j + 1] -
                                           t_div[int(n - parameters.N_0)]))))
    for j in range(len(theta_star[int(n - parameters.N_0)][:])):  # For the spine
        int_b -= ((1 - theta_star[int(n - parameters.N_0)][j]) / parameters.mu *
                  z_t[0] *
                  (np.exp(parameters.mu * (t_div[int(n - parameters.N_0) + 1] -
                                          t_div[int(n - parameters.N_0)])) -
                   np.exp(parameters.mu *
                          (t_loss_star[int(n - parameters.N_0)][j + 1] -
                           t_div[int(n - parameters.N_0)]))))
    # Outside spine loss events
    for j in range(len(idx_loss[int(n - parameters.N_0)][:])):
        z_t[idx_loss[int(n - parameters.N_0)][j]] *= theta[int(n - parameters.N_0)][j]
    # Spine loss events
    for thet in theta_star[int(n - parameters.N_0)][:]:
        z_t[0] *= thet
    # Deterministic growth between jumps
    for j in range(len(z_t)):
        z_t[j] *= np.exp(parameters.mu * (t_div[int(n - parameters.N_0) + 1] -
                                          t_div[int(n - parameters.N_0)]))
    return(z_t, int_b)


def trajectory(parameters):
    """Generate a single trajectory of the spinal process up to time T.

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features

    Returns
    -------
    float
        The value of x_bar estimated with this trajectory.
    """
    # Computing binary tree structure S
    (t_div, n, int_n, int_n_square) = spinal_division_times(parameters)
    (lambd, idx_div) = spinal_division_values(parameters, n)

    # Initialisation of the spine
    spine_initial_choice(parameters)

    # Computing the spinal piece-wise deterministic Markov process indexed on S
    (t_loss, idx_loss, theta, t_loss_star, theta_star) = \
        spinal_loss_events(parameters, t_div, n)
    (z_t, int_b) = spinal_individual_traits(parameters, t_div, n,
                                            lambd, idx_div, t_loss,
                                            idx_loss, theta, t_loss_star,
                                            theta_star)

    # Computing the exponential terms
    func_n = (int(n - parameters.N_0) * np.log(parameters.r * parameters.K_div /
                                            parameters.r_psi) +
              parameters.C_1 * int_n + parameters.C_2 * int_n_square)
    log_pi = np.sum(np.log(z_t)) - np.sum(np.log(parameters.z_0))
    log_x_bar = (np.log(np.sum(parameters.z_0) / n) + parameters.mu *
                 parameters.T + func_n + log_pi - parameters.r * int_b)
    return(np.exp(log_x_bar))
