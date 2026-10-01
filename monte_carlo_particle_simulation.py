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


def simulate(filename, potential, amount_of_particles = 100, kT = 1, box_size = 50, particle_diameter = 1, starting_position = 'random',  steps=10**7, equilibration_steps=10**6, monte_carlo_steps_per_frame=None):

    #rename and rescale variables
    N = amount_of_particles
    d = particle_diameter
    max_displacement = 0.25 * d
    if monte_carlo_steps_per_frame is None:
        monte_carlo_steps_per_frame = N
    volume_fraction = (N * d**2) / box_size**2    

    #set particle starting positions
    if starting_position == 'array':
        cols = int(box_size // d)
        i = np.arange(N)

        positions = np.array([
            (i % cols) * d + d / 2,
            (i // cols) * d + d / 2])

    elif starting_position == 'random':
        # Max placement attempts per particle to avoid infinite loops at high density
        max_attempts = 1000
        positions_list = []

        for i in range(N):
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                # Generate a candidate coordinate (keeping particle fully inside box)
                # If particles can cross boundaries (PBC), use (0, box_size) instead
                candidate = np.random.uniform(d / 2.0, box_size - d / 2.0, size=2)
                
                if len(positions_list) == 0:
                    positions_list.append(candidate)
                    placed = True
                else:
                    existing_pos = np.array(positions_list)
                    dx = existing_pos[:, 0] - candidate[0]
                    dy = existing_pos[:, 1] - candidate[1]

                    # --- Uncomment if using Periodic Boundary Conditions (PBC) ---
                    # dx = dx - box_size * np.round(dx / box_size)
                    # dy = dy - box_size * np.round(dy / box_size)

                    distances = np.sqrt(dx**2 + dy**2)

                    # Check if candidate overlaps with any existing particle
                    if np.all(distances >= d):
                        positions_list.append(candidate)
                        placed = True
                
                attempts += 1

            if not placed:
                raise RuntimeError(
                    f"Failed to place particle {i+1}/{N} without overlap after {max_attempts} attempts. "
                    f"The target density (packing fraction) is too high for random placement."
                )

        # Output shape: (2, N) matching your original positions = np.array([x_pos, y_pos])
        positions = np.array(positions_list).T

    else:
        # Default / fallback placement (e.g., random without overlap constraint, or grid)
        x_pos = np.random.uniform(0, box_size, size=N)
        y_pos = np.random.uniform(0, box_size, size=N)
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
            print(f"\rSimulating step {s} of {steps} ({s/steps:.2%})", end="")

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
            VOLUME_FRACTION=volume_fraction,
            kT=kT,
            STARTING_POSITION = starting_position,
            AMOUNT_OF_STEPS = steps,
            STEPS_PER_FRAME = monte_carlo_steps_per_frame,
            EQUILIBRATION_STEPS = equilibration_steps,
            )
    print('\rSimulation Done!                                      ')