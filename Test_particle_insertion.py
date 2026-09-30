import numpy as np
from monte_carlo_particle_simulation import compute_energy_difference
from matplotlib import pyplot as plt

###INPUT###
filename = '1_kT 1000_particles.npz'
amount_of_iterations = 250
amount_of_frames = 100
multiplier = 0.7


#LOAD DATA
g_data = np.load('pair correlation data/1_kT 1000_particles.npz', allow_pickle=True)
g_HD = g_data['g']
bin_size = g_data['BIN_SIZE']
r = g_data['r']

data = np.load(f"position data/{filename}", allow_pickle=True)
pos = data["pos"]
box_size = data["BOX_SIZE"]
N = data["AMOUNT_OF_PARTICLES"]
cut_off = data["CUT_OFF"]
d = data["PARTICLE_RADIUS"] * 2
kT = data["kT"]

#CONSTANTS
max_energy = -kT*np.log(10e-20)    

N_TP = int(np.round(np.sqrt(10*N))**2) #Amount of test particles, amount closest to 10N that fills a square

#make a grid of test particles
cols = int(np.sqrt(N_TP))
spacing = box_size / cols
i = np.arange(N_TP)
TP_pos = np.array([
    (i % cols) * spacing,
    (i // cols) * spacing])

#calculate comparision matrixes for all frames
all_comparision_matrixes = []
for i in range(amount_of_frames):
    print(f'\rComparision matrix {round(i/amount_of_frames*100, 1)}% calculated', end = '')
    comparision_matrix= []
    for TP in TP_pos.T:
        relative_positions = pos[-i].T - TP
        relative_positions-= box_size * np.round(relative_positions / box_size)
        distances = np.linalg.norm(relative_positions, axis=-1)
        distances = np.where(distances < cut_off, distances, np.nan)
        histogram = np.histogram(distances, bins=np.arange(0, cut_off + bin_size, bin_size))[0]
        comparision_matrix.append(histogram)    
    all_comparision_matrixes.append(np.array(comparision_matrix))
print(f'\r Comparision matrix calculation done')

g_HD = np.where(g_HD < 10e-20, 10e-20, g_HD) #remove near 0 values

u = -kT * np.log(g_HD)  #initial potential
u = np.nan_to_num(u, posinf=max_energy, neginf=-max_energy) #remove nan values and set upper and lowerlimit on energy

hist = [u]
for i in range(amount_of_iterations):
    iteration_progress = round(i / amount_of_iterations * 100, 1) #for the progress bar
    bulk_average = []
    local_average = []
    for j in range(amount_of_frames):
        frame_progress = round(j / amount_of_frames * 100, 1) #for the progress bar
        print(f'\rIterations:{iteration_progress}% frames:{frame_progress}%', end='') #progress bar
        comparision_matrix = all_comparision_matrixes[j]
        TP_additional_energy = u @ comparision_matrix.T #gives row matrix of the added energy of every TP
        exponential = np.exp(-TP_additional_energy/kT)  #gives probability of TP being there (missing constant prefactor)
        bulk_average.append(np.mean(exponential))       #takes the average of all particles
        local_average.append(np.divide(exponential @ comparision_matrix, comparision_matrix.sum(axis=0))) #list where every element is the average taken over all particles that are at distance r_i from another particle
    bulk_average = np.mean(bulk_average)            #bulk average averaged over all frames
    local_average = np.mean(local_average, axis=0)  #local average averaged over all frames

    g_TPI = np.divide(local_average, bulk_average)
    ratio = np.divide(g_HD, g_TPI)
    ratio = np.where(ratio < 10e-20, 10e-20, ratio)
    log_ratio = np.log(ratio)
    u = u -kT * multiplier * log_ratio
    u = np.nan_to_num(u, posinf=max_energy, neginf=-max_energy)
    hist.append(u)

np.savez(f'TPI result data/{filename}',
        FINAL_POTENTIAL=hist[-1],
        POTENTIAL_HISTORY=hist,
        r = r,
        BIN_SIZE = bin_size
        )

