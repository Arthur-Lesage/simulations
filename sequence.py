from monte_carlo_particle_simulation import simulate
from monte_carlo_animate_data import animate
from pair_correlation import get_pair_correlation_of_last_frames
from Test_particle_insertion import TPI
from datetime import datetime
import numpy as np 

import numpy as np

cut_off = 5
def lennard_jones(r, sigma=1.0, cutoff=cut_off):
    r_cutoff = cutoff * sigma
    return np.where(r <= r_cutoff, 4.0 * ((sigma / r)**12 - (sigma / r)**6), 0.0)

def hard_disk(r, sigma=1.0):
    return np.where(r <= sigma, np.inf, 0.0)

def attractive_square_well(r, sigma=1.0, lam=1.5, epsilon=-1.0):
    conditions = [
        r <= sigma,
        (r > sigma) & (r <= lam * sigma),
        r > lam * sigma
    ]
    choices = [np.inf, epsilon, 0.0]
    return np.select(conditions, choices)

def repulsive_square_well(r, sigma=1.0, lam=1.5, epsilon=1.0):
    conditions = [
        r <= sigma,
        (r > sigma) & (r <= lam * sigma),
        r > lam * sigma
    ]
    choices = [np.inf, epsilon, 0.0]
    return np.select(conditions, choices)

def attractive_yukawa(r, sigma=1.0, kappa=1.5, epsilon=-1.0, cutoff=cut_off):
    r_cutoff = cutoff * sigma
    v_yukawa = (epsilon * sigma / r) * np.exp(-kappa * (r / sigma - 1.0))
    
    conditions = [
        r <= sigma,
        (r > sigma) & (r <= r_cutoff),
        r > r_cutoff
    ]
    choices = [np.inf, v_yukawa, 0.0]
    return np.select(conditions, choices)

def repulsive_yukawa(r, sigma=1.0, kappa=1.5, epsilon=1.0, cutoff=cut_off):
    r_cutoff = cutoff * sigma
    v_yukawa = (epsilon * sigma / r) * np.exp(-kappa * (r / sigma - 1.0))
    
    conditions = [
        r <= sigma,
        (r > sigma) & (r <= r_cutoff),
        r > r_cutoff
    ]
    choices = [np.inf, v_yukawa, 0.0]
    return np.select(conditions, choices)

potentials = [repulsive_square_well, attractive_yukawa, repulsive_yukawa]

start_time = datetime.now()
print(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

box_size = 50
particle_diameter = 1
amount_of_particles = 1000
for pot in potentials:
    filename = f'{pot.__name__}'
    print(f'Caculating {filename}:')
    #simulate(filename=f'{filename}',
    #        potential = pot,
    #        amount_of_particles=amount_of_particles,
    #        box_size=box_size,
    #        particle_diameter=particle_diameter,
    #        steps=10**7,
    #        equilibration_steps=10**6,
    #        monte_carlo_steps_per_frame=10**4
    #        )
    #animate(f'{filename}.npz')
    #get_pair_correlation_of_last_frames(f'{filename}.npz',
    #                                    cut_off=cut_off,
    #                                    bin_size=0.02,
    #                                    amount_of_frames=900)
    TPI(f'{filename}.npz', 
            cut_off=cut_off,
            amount_of_iterations=250, 
            amount_of_frames=900)

end_time = datetime.now()
print(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {end_time - start_time}")


