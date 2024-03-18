"""Plot the efficiency of the 2 spinal methods, relatively to the Ogata method.

Usage:
======
    Choose M, the Monte_Carlo number of iterations, and the function
    plot_running_times from module plot_function will plot the running times of
    the two spinal methods, normalized by the running time of the Ogata method.

    M: an integer from 100 to 1.000.000 (prompt input)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import plot_function

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

    def __init__(self, mu=0.5, r=0.5, d=0.5, N_0=2, init_mass=0.5, psi=1):
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
        self.__dict__['N_0'] = int(N_0)
        # Mean mass of the individuals in the initial population
        self.__dict__['init_mass'] = init_mass
        # Chosen psi function (1 or 2)
        self.__dict__['psi'] = psi
        # Initial traits
        self.__dict__['z_0'] = init_mass * np.ones(int(N_0))
        # r_psi = 1 if psi == 1, = r if psi == 2
        self.__dict__['r_psi'] = (psi - 1) * r + 2 - psi
        # Mean of 1/Lambda(1-Lambda)
        self.__dict__['K_div'] = 4 + 2 / (self.alpha - 1)
        # Mean of 1/Theta
        self.__dict__['K_loss'] = (self.a + self.b - 1) / (self.a - 1)
        # Parameters in the integral term
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
                self.__dict__['r_psi'] = (self.psi-1) * value + 2 - self.psi
                self.__dict__['C_1'] = (self.r_psi - self.mu - self.d *
                                        (self.K_loss - 1))
            elif name == "d":
                self.__dict__[name] = value
                self.__dict__['C_1'] = (self.r_psi - self.mu - value *
                                        (self.K_loss - 1))
                self.__dict__['C_2'] = value * (self.K_loss - 1)
            elif name == "N_0":
                self.__dict__[name] = int(value)
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
        line_0 = (r'\mu\ =\ % .2f\ g.s^{-1},\ ' % (self.mu) +
            r'r\ =\ % .2f\ s^{-1},\ d\ =\ % .2f \ s^{-1}' % (self.r, self.d) +
            r',\ N_0\ =\ % d,\ x_0\ =\ % .2f \ g' % (self.N_0, self.init_mass))
        line_1 = (r'K_{div}\ :=\ ' +
                  r'\mathbb{E}\left[\frac{1}{\Lambda(1 - \Lambda)} \right]\ ' +
                  ' =\ % .2f' % self.K_div)
        line_2 = (r'K_{loss}\ :=\ \mathbb{E}\left[\frac{1}{\Theta} \right]\ ' +
                  r'= \ % .2f' % self.K_loss)
        line_3 = (r'C_1\ :=\ r_{\psi} - \mu - d(K_{loss} - 1)\ ' +
                  r'=\ % .2f \ s^{-1}' % self.C_1)
        line_4 = r'C_2\ :=\ d(K_{loss} - 1)\ =\ % .2f \ s^{-1}' % self.C_2

        plt.figure(figsize=(19.5, 6))
        ax = plt.axes([0, 0, 0.25, 0.25])  # left,bottom,width,height
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.text(0.1, 3.5, '$ % s$' % line_0, size=50)
        plt.text(0.1, 2.7, '$ % s$' % line_1, size=50)
        plt.text(0.1, 1.7, '$ % s$' % line_2, size=50)
        plt.text(0.1, 0.9, '$ % s$' % line_3, size=50)
        plt.text(0.1, 0.15, '$ % s$' % line_4, size=50)

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
    M = 0
    run = 'n'
    while run == 'n' or run == 'no' or run == 'N' or run == 'NO':
        M = int(input("Enter the number of Monte-Carlo iterations: M = "))
        if M <= 50:
            running_time = 'less than 1 min'
            print("running time is:", running_time)
        elif M > 50 and M < 6000:
            running_time = M / 50
            print("running time is around: %d min" % (running_time))
        else:
            running_time = M / 6000
            print("running time is around: %d h" % (running_time))
        run = input("do you confirm? (y/n) ")

    parameters = Parameters()
    plot_function.plot_running_times(parameters, M)
