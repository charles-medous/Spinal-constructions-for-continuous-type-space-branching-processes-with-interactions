# Simulation of a Yule model with interactions

This repository gives the simulation algorithm for the 
Yule model with interactions introduced in the article
'Spinal construction for continuous-type space branching
processes with interactions', available on arxiv:
https://arxiv.org/pdf/2309.15449.pdf


The algorithmic 'spinal' method proposed in the article is
tested against the classical Ogata's method to generate a
large number of stochastic trajectories of the population. 
The `python` implementation uses the classical libraries
`numpy`, `matplotlib`, `tqdm` and `itertools` to be able 
to run on every setup. However an optimized version using 
parallel programming and more fancy packages like 
`multiprocessing` and `oml` will be soon proposed.

# Installation

Download/Clone the repository, and acces it.

```sh
git clone https://github.com/charles-medous/Spinal-constructions-for-continuous-type-space-branching-processes-with-interactions.git
cd Spinal-constructions-for-continuous-type-space-branching-processes-with-interactions

```

Please ensure that the required packages are locally 
installed on your python version.

On windows, type `cmd` in the search bar and hit `Enter`
to open the command line. On linux, open your terminal or
shell. If you installed python via Anacomba, directly 
use the conda shell. Then type

```sh
# Create a virtual environnement
python3 -m venv venv

# Activate the virtual environnement
source venv/bin/activate

# Install the mandatory modules
pip install -r requirements.txt
```

The file `main.py` compares the efficiency of the spinal 
methods with the Ogata method, based on their running
time:
```sh
python3 main.py
```
The file `x_bar_estimation.py` displays the estimated 
value of the mean size of an individual picked uniformly
at random in the population at the simulation time T.
This value is estimated with a Monte-Carlo routine for 
the 3 different trajectorial methods:
```sh
python3 x_bar_estimation.py
```
