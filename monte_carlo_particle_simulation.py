import numpy as np
import random as rnd

from matplotlib import pyplot as plt

def compute_energy_difference(particle_number, new_position, positions, box_size, potential):
    old_position = positions[:, particle_number]
    old_relative_positions = positions.T-old_position
    old_relative_positions -= box_size * np.round(old_relative_positions / box_size)
    old_distances = np.linalg.norm(old_relative_positions, axis=1)
    old_distances[particle_number] = np.inf
    old_energy = np.sum(potential(old_distances))

    new_relative_positions = positions.T - new_position
    new_relative_positions -= box_size * np.round(new_relative_positions / box_size)
    new_distances = np.linalg.norm(new_relative_positions, axis=1)
    new_distances[particle_number] = np.inf
    new_energy = np.sum(potential(new_distances))

    return new_energy - old_energy


def simulate(amount_of_particles = 100, interaction_strength = 1, cut_off = 5, kT = 1, box_size = 50, particle_diameter = 1, starting_position = 'random',  steps=10**7, equilibration_steps=10**6, monte_carlo_steps_per_frame=None, filename = None):

    #rename and rescale variables
    N = amount_of_particles
    d = particle_diameter
    max_displacement = 0.25 * d
    if monte_carlo_steps_per_frame is None:
        monte_carlo_steps_per_frame = N
    if filename is None:
        filename = f'{interaction_strength}_kT {N}_particles'
    volume_fraction = (N * d**2) / box_size**2

    #define potential
    def potential(r, sigma=d, epsilon=interaction_strength * kT):
        if epsilon == 0:
            return 0
        
        s_r = sigma / r
        return np.where(
            r > cut_off,
            0,
            4 * epsilon * (s_r ** 12 - s_r ** 6)
        )

    

    #set particle starting positions
    if starting_position == 'array':
        cols = int(box_size // d)
        i = np.arange(N)

        positions = np.array([
            (i % cols) * d + d / 2,
            (i // cols) * d + d / 2])

    elif starting_position == 'random':
        x_pos = [np.random.uniform(0,box_size) for _ in range(N)]
        y_pos = [np.random.uniform(0, box_size) for _ in range(N)]

    else:
        x_pos = [np.random.uniform(0, box_size) for _ in range(N)]
        y_pos = [np.random.uniform(0, box_size) for _ in range(N)]

    positions = np.array([x_pos, y_pos])

    history=[]

    accepted = 0
    for s in range(steps):
        particle_number = rnd.randint(0,N-1)
        dx = np.random.uniform(-max_displacement, max_displacement)
        dy = np.random.uniform(-max_displacement, max_displacement)

        new_position = positions[:, particle_number] + np.array([dx, dy])
        new_position -= box_size * np.round(new_position / box_size)

        delta_E = compute_energy_difference(particle_number, new_position, positions, box_size, potential)

        if delta_E <= 0 or np.exp(-delta_E/kT) > np.random.uniform(0,1):
            positions[:, particle_number] = new_position
            accepted += 1

        if s % N==0: 
            print(f"\rStep {s} of {steps} ({s/steps:.2%})", end="")

            if s < steps // 5:         #check acceptance every N steps and adjust for the first 20% of the simulation
                acceptance = accepted / N
                if s < equilibration_steps:
                    if acceptance > 0.5:
                        max_displacement *= 1.1
                    elif acceptance < 0.25:
                        max_displacement *= 0.9
                accepted = 0

        if s >= equilibration_steps and s % monte_carlo_steps_per_frame == 0: #make snapshots of the simulation after equilibration
            history.append(positions.copy())
        

    #save simulation
    np.savez(f'position data/{filename}.npz',
            pos=history,           ##mandatory information (for animation)
            BOX_SIZE=box_size,
            PARTICLE_RADIUS=d/2,

            AMOUNT_OF_PARTICLES=N, ##optional information
            INTERACTION_STRENGTH=f'{interaction_strength} kT',
            VOLUME_FRACTION=volume_fraction,
            kT=f'{kT}J',
            STARTING_POSITION = starting_position,
            AMOUNT_OF_STEPS = steps,
            STEPS_PER_FRAME = monte_carlo_steps_per_frame,
            EQUILIBRATION_STEPS = equilibration_steps,
            CUT_OFF = cut_off
            )
    print('\nDone!')