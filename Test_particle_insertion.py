import numpy as np
from monte_carlo_particle_simulation import compute_energy_difference


filename = '3_kT 1000_particles.npz'
bin_size = 0.02


data = np.load(f"position data/{filename}", allow_pickle=True)

pos = data["pos"][-1]
box_size = data["BOX_SIZE"]
N = data["AMOUNT_OF_PARTICLES"]
cut_off = data["CUT_OFF"]
d = data["PARTICLE_RADIUS"] * 2



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

