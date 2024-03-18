"""Estimate x_bar with the 2 spinal methods and the ogata method.

Usage:
======
    Choose (M, idx) the Monte_Carlo number of iterations, and the set of
    parameters on which to compare the 3 methods.

    M: an integer from 1000 to 1.000.000 (prompt input)
    idx: an integer between 0 and 13 (prompt input)

    Print the estimated values for x_bar for the 3 methods and the relative
    percent difference (RPD). The RPD is defined as the difference divided by
    the average of the two values.
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

import spinal_method
import ogata_method

matplotlib.rcParams['mathtext.fontset'] = 'stix'


class Parameters:
    """Class containing all the features of the model.

    The parameters given as class attributes should not be changed to keep the
    setting of the introduced model.
    The parameters mu,r,d,N_0,init_mass are user defined and the functions
    print_constant and plot_distribution display the chosen model features.
    """

    T = 1  # Time length of the simulated trajectory
    alpha = 20  # Parameter of the division repartition, >= 2
    b = 2  # loss parameter #1, >= 1
    a = 10  # loss parameter #2, > param # 1

    def __init__(self, mu, r, d, N_0, init_mass, psi):
        """Initialize the parameters mu,r,d,N_0,init_mass and psi.

        Computes the initial population z_0 and some
        constant values. The __setattr__ method is overwriten thus the
        attributes are directly changed in the class dictionary using __dict__.
        """
        # mass growth rate
        self.__dict__['mu'] = mu
        # division rate for unit size
        self.__dict__['r'] = r
        # loss rate for a single individual
        self.__dict__['d'] = d
        # Initial size of the population >= 2
        self.__dict__['N_0'] = N_0
        # Mean mass of the individuals in the initial population
        self.__dict__['init_mass'] = init_mass
        # Chosen psi function (1 or 2)
        self.__dict__['psi'] = psi
        # Initial traits
        self.__dict__['z_0'] = init_mass * np.ones(int(N_0))
        # r_psi = 1 if psi == 1, = r if psi == 2
        self.__dict__['r_psi'] = (psi - 1) * r + 2 - psi
        # mean of 1/Lambda(1-Lambda)
        self.__dict__['K_div'] = 4 + 2 / (self.alpha - 1)
        # mean of 1/Theta
        self.__dict__['K_loss'] = (self.a + self.b - 1) / (self.a - 1)
        self.__dict__['C_1'] = self.r_psi - mu - d * (self.K_loss - 1)
        self.__dict__['C_2'] = d * (self.K_loss - 1)

    def __setattr__(self, name, value):
        """Overwrite special method to set a value to an attribute.

        The attributes that depend on the changes of the setted attribute are
        changed directly using __dict__ method. Attributes other than mu, r, d,
        N_0, init_mass and psi cannot be changed.
        """
        if name in self.__dict__:
            if name == "mu":
                self.__dict__[name] = value
                self.__dict__['C_1'] = (self.r_psi-value-self.d *
                                        (self.K_loss - 1))
            elif name == "r":
                self.__dict__[name] = value
                self.__dict['r_psi'] = (self.psi-1) * value + 2 - self.psi
                self.__dict__['C_1'] = (self.r_psi - self.mu - self.d *
                                        (self.K_loss - 1))
            elif name == "d":
                self.__dict__[name] = value
                self.__dict__['C_1'] = (self.r_psi - self.mu - value *
                                        (self.K_loss - 1))
                self.__dict__['C_2'] = value * (self.K_loss - 1)
            elif name == "N_0":
                self.__dict__[name] = value
                self.__dict__['z_0'] = self.init_mass * np.ones(int(value))
            elif name == "init_mass":
                self.__dict__[name] = value
                self.__dict__['z_0'] = value * np.ones(int(self.N_0))
            elif name == "psi":
                self.__dict__[name] = value
                self.__dict__['r_psi'] = (value - 1) * self.r + 2 - value
                self.__dict__['C_1'] = (self.r_psi - self.mu - self.d *
                                        (self.K_loss - 1))
            else:
                print('error: you cannot change this value')
        else:
            pass

    def print_constant(self):
        """Print all the model features in a matplotlib figure.

        Print the parameters chosen by the users as well as the computed
        normalisation constant K_div and K_loss and the constant C_1 and C_2
        defined in the presented model. Display values and units in teX font.
        """
        line_0 = (r'\mu = % .2f \ g.s^{-1}, \ r= % .2f \ s^{-1},\ d = % .2f \
                  \ s^{-1},\ N_0 = % d,\ x_0 = % .2f \ g' % (self.mu, self.r,
                  self.d, self.N_0, self.init_mass))
        line_1 = r'K_{div} := \mathbb{E}\left[\frac{1}{\Lambda(1 - \Lambda)} \
            \right] = % .2f' % self.K_div
        line_2 = r'K_{loss} := \mathbb{E}\left[\frac{1}{\Theta} \right] = \
            % .2f' % self.K_loss
        line_3 = r'C_1 := r_{\psi} - \mu - d(K_{loss} - 1) = % .2f \ s^{-1}' \
            % self.C_1
        line_4 = r'C_2 := d(K_{loss} - 1) = % .2f \ s^{-1}' % self.C_2

        plt.figure(figsize=(18, 6))
        ax = plt.axes([0, 0, 0.25, 0.25])  # left,bottom,width,height
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.text(0.1, 3.5, '$ % s$' % line_0, size=50)
        plt.text(0.1, 2.7, '$ % s$' % line_1, size=50)
        plt.text(0.1, 1.7, '$ % s$' % line_2, size=50)
        plt.text(0.1, 0.9, '$ % s$' % line_3, size=50)
        plt.text(0.1, 0.1, '$ % s$' % line_4, size=50)

    def plot_distributions(self):
        """Plot the probability density functions of Theta and Lambda."""
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(np.random.beta(self.a, self.b, size=500000), bins=70,
                 density=True)
        ax1.set_title('Probability density of ' + r'$\Theta$')
        ax2.set_title('Probability density of ' + r'$\Lambda$')
        ax2.hist(np.random.beta(self.alpha, self.alpha, size=500000), bins=70,
                 density=True)
        f.show()


if __name__ == "__main__":
    Param_set = [(0.3, 0.3, 0.3, 2, 0.5), (0.3, 0.3, 0.3, 5, 0.5),
                 (0.3, 0.3, 0.3, 8, 0.5), (0.3, 0.05, 0.3, 2, 0.5),
                 (0.3, 1, 0.3, 2, 0.5), (0.3, 1.5, 0.3, 2, 0.5),
                 (0.3, 0.3, 0.3, 2, 1), (0.3, 0.3, 0.3, 2, 1.5),
                 (0.05, 0.3, 0.3, 2, 0.5), (1, 0.3, 0.3, 2, 0.5),
                 (1.5, 0.3, 0.3, 2, 0.5), (0.3, 0.3, 0.05, 2, 0.5),
                 (0.3, 0.3, 1, 2, 0.5), (0.3, 0.3, 1.8, 2, 0.5)]
    print('Choose idx of the parameters (mu, r, d, N_0, init_mass) in the ' +
          'folowing list :\n' +
          '[(0.3, 0.3, 0.3, 2, 0.5), (0.3, 0.3, 0.3, 5, 0.5),\n'+
                       '(0.3, 0.3, 0.3, 10, 0.5), (0.3, 0.05, 0.3, 2, 0.5),\n'+
                       '(0.3, 1, 0.3, 2, 0.5), (0.3, 2, 0.3, 2, 0.5),\n'+
                       '(0.3, 0.3, 0.3, 2, 1), (0.3, 0.3, 0.3, 2, 2),\n'+
                       '(0.05, 0.3, 0.3, 2, 0.5), (1, 0.3, 0.3, 2, 0.5),\n'+
                       '(2, 0.3, 0.3, 2, 0.5), (0.3, 0.3, 0.05, 2, 0.5),\n'+
                       '(0.3, 0.3, 1, 2, 0.5), (0.3, 0.3, 2, 2, 0.5)]\n')
    idx = -1
    while idx < 0 or idx > 13:
        idx = int(input("Enter the chosen idx between 0 and 13: idx = "))
    M = 0
    run = 'n'
    while run == 'n' or run == 'no' or run == 'N' or run == 'NO':
        M = int(input("Enter the number of Monte-Carlo iterations: M = "))
        if idx in [0, 6, 7, 8, 9, 10]:
            running_time = M * 3.7 / 10000
        elif idx in [4, 12]:
            running_time = M * 4.9 / 10000
        elif idx in [3, 11]:
            running_time = M * 3.1 / 10000
        elif idx == 1:
            running_time = M * 8.7 / 10000
        elif idx in [5, 13]:
            running_time = M * 6 / 10000
        elif idx == 2:
            running_time = M * 18 / 10000

        if  M < 100000:
            print("running time: around %.d secondes" % (round(running_time)))
        else:
            print("running time: around %.1f min" % (running_time/60))
        run = input("do you confirm? (y/n) ")
  
    (mu, r, d, N_0, init_mass) = Param_set[idx]
    parameters = Parameters(mu, r, d, N_0, init_mass, 1)
    x_bar_spine = 0
    x_bar_spine_1 = 0
    x_bar = 0

    for i in tqdm(range(0, M), mininterval=1, ncols=100,
                  desc="x_bar estimation. Progress: "):
        # Spinal method 1
        parameters.psi = 1
        x_bar_spine += spinal_method.trajectory(parameters) / M
        # Spinal method 2
        parameters.psi = 2
        x_bar_spine_1 += spinal_method.trajectory(parameters) / M
        # Ogata's method
        x_bar += ogata_method.trajectory(parameters) / M

    print('\n \n x_bar = %.3f with spinal method 1' % (x_bar_spine))
    print('\n x_bar = %.3f with spinal method 2' % (x_bar_spine_1))
    print('\n x_bar= %.3f with Ogata method' % (x_bar))
    rpd = [2 * abs(x_bar_spine-x_bar_spine_1) / (x_bar_spine + x_bar_spine_1),
           2 * abs(x_bar_spine-x_bar) / (x_bar_spine + x_bar),
           2 * abs(x_bar-x_bar_spine_1) / (x_bar + x_bar_spine_1)]
    print('\n Relative Percent Difference: RPD_{S1,S2} = ' +
          "{:.1%}".format(rpd[0]) + ', RPD_{S1,O} = '  +
          "{:.1%}".format(rpd[1]) + ', RPD_{S2,O} = ' +
          "{:.1%}".format(rpd[2]))
