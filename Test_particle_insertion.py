import numpy as np
from monte_carlo_particle_simulation import compute_energy_difference
from matplotlib import pyplot as plt


filename = '3_kT 1000_particles.npz'
bin_size = 0.02

g_data = np.load('pair correlation data/3_kT 1000_particles.npz', allow_pickle=True)

data = np.load(f"position data/{filename}", allow_pickle=True)

pos = data["pos"][-1]
box_size = data["BOX_SIZE"]
N = data["AMOUNT_OF_PARTICLES"]
cut_off = data["CUT_OFF"]
d = data["PARTICLE_RADIUS"] * 2
kT = data["kT"]
kT = 1 ##&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&REMOVE THIS LINEEEEEEEE&&&&&&&&&&&&&&&&&


N_TP = int(np.round(np.sqrt(10*N))**2)

cols = int(np.sqrt(N_TP))
spacing = box_size / cols
i = np.arange(N_TP)
TP_pos = np.array([
    (i % cols) * spacing,
    (i // cols) * spacing])

comparision_matrix= []
for TP in TP_pos.T:
    distances = np.linalg.norm(pos.T - TP, axis=1)
    distances = np.where(distances < cut_off, distances, np.nan)
    histogram = np.histogram(distances, bins=np.arange(0, cut_off + bin_size, bin_size))[0]
    comparision_matrix.append(histogram)

g_HD = g_data['g']

u_0 = -kT * np.log(g_HD)