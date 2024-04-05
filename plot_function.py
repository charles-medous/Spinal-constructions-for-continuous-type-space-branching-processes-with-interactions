"""Define the function that plot the efficiency of the 2 spinal methods.

Usage:
======
    Use 'plot_running_times' function to compare running times of the 2 spinal
    methods with Ogata method. Choose a number of Monte-Carlo iterations and
    change the base parameters values if necessary. The discretization step to
    explore the set of parameters can also be tuned if necessary.
"""
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib
from itertools import repeat

import spinal_method
import ogata_method

matplotlib.rcParams['mathtext.fontset'] = 'stix'


def plot_running_times(parameters, M, N_pts = 15,
                       base = [(0.2 , 0.2 , 0.2 , 2 , 1),
                               (0.8 , 0.8 , 0.8 , 2 , 3),
                               (1.2 , 1.2 , 1.2 , 2 , 5)]):
    """Plot efficiency of the spinal methods for different parameter values.

    For a chosen number M of Monte-Carlo iterations, compute the estimation of
    x_bar using the 2 spinal methods and the Ogata one for different parameter
    values. Plot the efficiency of the spinal methods compared to the Ogata one
    as the ratio of running times for generating M trajectories .

    Parameters
    ----------
    class object
        The class 'parameters' that contains all the model features
    int
        The number of Monte-Carlo iterations
    int (optional)
        The number of sampled points for each parameter in the parameters space
    list (optional)
        The base values for the parameters

    Returns
    -------
    None.
    """
    param = [np.linspace(2, 11, 10),np.linspace(0.05, 10, N_pts),
             np.linspace(0.05, 3.5, N_pts), np.linspace(0.05, 3.5, N_pts),
             np.linspace(0.05, 3.5, N_pts)]
    n_columns = len(base[0])
    n_rows = len(base)
    str_param = [r'$N_0$', r'$x_0$', r'$\mu$', r'$r$', r'$d$']
    fig,axes = plt.subplots(n_rows,n_columns)
    for i in range (n_rows):
        for j in range (n_columns):
            (parameters.mu, parameters.r, parameters.d, parameters.N_0,
             parameters.init_mass) = base[i]
            n_pts = len(param[j])
            t_spine_1 = np.ones(n_pts)
            t_spine_2 = np.ones(n_pts)
            t_ogata = np.ones(n_pts)
            x_axis = param[j]
            for n in tqdm(range(0, n_pts), mininterval=1, ncols=100,
                          desc="Plot simulation times. Progress " +
                          str(1+n_columns * i + j) + "/" +
                          str(n_rows*n_columns) + ": "):
                if j == 0: 
                    parameters.N_0 = x_axis[n]
                elif j == 1:
                    parameters.init_mass = x_axis[n]
                elif j == 2:
                    parameters.mu = x_axis[n]
                elif j == 3:
                    parameters.r = x_axis[n]
                elif j == 4:
                    parameters.d = x_axis[n]
                # Spinal method 1
                t_0 = time.time()
                parameters.psi = 1
                for _ in repeat(None, M):
                    spinal_method.trajectory(parameters)
                t_spine_1[n]=time.time()-t_0
                # Spinal method 2
                t_0 = time.time()
                parameters.psi = 2
                for _ in repeat(None, M):
                    spinal_method.trajectory(parameters)
                t_spine_2[n]=time.time()-t_0
                # Ogata's method
                t_0 = time.time()
                for _ in repeat(None, M):
                    ogata_method.trajectory(parameters)
                t_ogata[n]=time.time()-t_0
            axes[i][j].axhline(y=1, color='grey')  
            axes[i][j].set_box_aspect(1)
            axes[i][j].plot(x_axis,t_ogata/t_spine_1, label = r'$\psi_1$'+'-spine method' )
            axes[i][j].plot(x_axis,t_ogata/t_spine_2, label = r'$\psi_2$'+'-spine method')
            axes[i][j].tick_params(axis='both', labelsize = 15)  
            axes[0][j].set_title('Efficiency with '+str_param[j], fontsize = 20)
            axes[n_rows-1][j].set_xlabel(str_param[j], fontsize = 20)
            axes[i][0].set_ylabel(r'$\frac{T_O}{T_S}$     ',fontsize = 30, rotation=0)
    for i in range(n_rows-1):
        for j in range(n_columns):
            axes[i][j].set_xticks([])
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[n_rows-1][2].legend(handles, labels,loc='upper center', 
                 bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=2,fontsize = 15)       
    return()
