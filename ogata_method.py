"""Define the function that generates a trajectory using the ogata method.

Usage:
======
    Use 'trajectory' function to compare running times and estimates of x_bar
"""
import numpy as np


def trajectory(parameters):
    """Generate a single trajectory of the original process up to time T.

    This trajectory is computed using the Ogata's acceptance-rejection method.

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features

    Returns
    -------
    float
        The value of x_bar estimated with this trajectory.
    """
    t_simu = 0
    x_t = []
    n_pop = len(parameters.z_0)
    for i in range(n_pop):
        x_t.append(parameters.z_0[i])
    n_reject = 0
    n_div = 0
    n_loss = 0
    while t_simu < parameters.T:
        # Next jump time computation
        tau_max = (parameters.r * sum(x_t) * np.exp(parameters.mu
                                                    * (parameters.T - t_simu))
                   + parameters.d * n_pop ** 2)
        t_jump = np.random.exponential(1/tau_max)
        if t_jump + t_simu > parameters.T:
            break

        # Temporary population up to time t_simu+t_jump
        z_temp = []
        for i in range(n_pop):
            z_temp.append(x_t[i] * np.exp(parameters.mu * t_jump))

        # Choosing the type of event
        u = np.random.uniform()*tau_max
        if u <= parameters.r * sum(z_temp):  # Branching event is a division
            idx = 0
            while parameters.r * np.cumsum(z_temp)[idx] < u:
                idx += 1
            lambd = np.random.beta(parameters.alpha, parameters.alpha)
            z_temp.append(z_temp[idx] * (1-lambd))
            z_temp[idx] *= lambd
            # Update population values
            for i in range(n_pop):
                x_t[i] = z_temp[i]
            x_t.append(z_temp[-1])
            n_pop += 1
            t_simu += t_jump
            n_div += 1
        # The branching event is a loss
        elif (parameters.r * sum(z_temp) < u and
              u <= parameters.r * sum(z_temp) + parameters.d * n_pop ** 2):
            idx = np.random.randint(0, n_pop)
            theta = np.random.beta(parameters.a, parameters.b)
            z_temp[idx] *= theta
            # Update population values
            for i in range(n_pop):
                x_t[i] = z_temp[i]
            t_simu += t_jump
            n_loss += 1

        else:  # The branching event is rejected
            n_reject += 1
    for i in range(n_pop):
        x_t[i] *= np.exp(parameters.mu*(parameters.T-t_simu))
    idx_unif = np.random.randint(0, n_pop)
    return(x_t[idx_unif])
