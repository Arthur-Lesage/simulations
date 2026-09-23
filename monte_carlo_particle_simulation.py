import numpy as np
import random as rnd

from matplotlib import pyplot as plt

UM = 1e-6  # micrometer

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


def simulate(amount_of_particles = 100, interaction_strength = 1, frames=100, monte_carlo_steps_per_frame=None, temperature = 300, box_size = 30, particle_radius = 1, starting_position = 'random', filename = None):
    global UM

    #rename and rescale variables
    kT = temperature * 1.38e-23
    box_size *= UM  # size of box side
    N = amount_of_particles
    a = particle_radius * UM
    max_displacement = 0.25 * a
    if monte_carlo_steps_per_frame is None:
        monte_carlo_steps_per_frame = N
    if filename is None:
        filename = f'{interaction_strength}_kT {N}_particles'
    volume_fraction = (N * np.pi * a ** 2) / box_size ** 2

    #define potential
    def potential(r, sigma=2 * a, epsilon=interaction_strength * kT):
        if epsilon == 0:
            return 0
        
        s_r = sigma / r
        return 4 * epsilon * (s_r ** 12 - s_r ** 6) #lenard-jones

    #set particle starting positions
    if starting_position == 'array':
        cols = int(box_size // (2*a))
        x_pos=[]
        y_pos=[]
        for i in range(N):
            x = (i % cols) * 2 * a + a
            x_pos.append(x)
            y = (i // cols) * 2 * a + a
            y_pos.append(y)

    elif starting_position == 'random':
        x_pos = [np.random.uniform(0,box_size) for _ in range(N)]
        y_pos = [np.random.uniform(0, box_size) for _ in range(N)]

    else:
        x_pos = [np.random.uniform(0, box_size) for _ in range(N)]
        y_pos = [np.random.uniform(0, box_size) for _ in range(N)]

    positions = np.array([x_pos, y_pos])

    history=[positions.copy()]
    for s in range(frames):
        print(f"\rSimulating {interaction_strength} kT, {N} particles: {100*s/(frames-1):.2f}%", end="")

        accepted = 0
        for i in range(monte_carlo_steps_per_frame):
            particle_number = rnd.randint(0,N-1)
            dx = np.random.uniform(-max_displacement, max_displacement)
            dy = np.random.uniform(-max_displacement, max_displacement)

            new_position = positions[:, particle_number] + np.array([dx, dy])
            new_position -= box_size * np.round(new_position / box_size)

            delta_E = compute_energy_difference(particle_number, new_position, positions, box_size, potential)

            if delta_E <= 0 or np.exp(-delta_E/kT) > np.random.uniform(0,1):
                positions[:, particle_number] = new_position
                accepted += 1
        acceptance = accepted / monte_carlo_steps_per_frame

        if s < frames //5:
            if acceptance > 0.6:
                max_displacement *= 1.1
            elif acceptance < 0.4:
                max_displacement *= 0.9

        history.append(positions.copy())

        #save simulation
        np.savez(f'position data/{filename}.npz',
                 pos=history,           ##mandatory information (for animation)
                 BOX_SIZE=box_size,
                 PARTICLE_RADIUS=a,

                 AMOUNT_OF_PARTICLES=N, ##optional information
                 INTERACTION_STRENGTH=f'{interaction_strength} kT',
                 VOLUME_FRACTION=volume_fraction,
                 kT=f'{kT}J',
                 MICROMETER = UM,
                 STARTING_POSITION = starting_position,
                 AMOUNT_OF_FRAMES = frames,
                 STEPS_PER_FRAME = monte_carlo_steps_per_frame
                 )
    print('\nDone!')