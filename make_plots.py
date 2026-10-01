from matplotlib import pyplot as plt
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

potential = lennard_jones
filename = 'lennard_jones.npz'

TPI_data = np.load(f"TPI result data/{filename}", allow_pickle=True)
hist = TPI_data['POTENTIAL_HISTORY']
r = TPI_data['r']

colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(hist)))

fig, ax = plt.subplots(figsize=(8, 5))

for i, h in enumerate(hist):
    ax.plot(r, h, color=colors[i], alpha=0.7)

ax.set_xlabel('r')
ax.set_ylabel('Potential')

sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=len(hist) - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('First iteration to last iteration')

plt.plot(r, hist[0], color = 'green', alpha = 0.7, label = 'Initial guess')
plt.plot(r, hist[-1], color = 'red', alpha = 0.7, label = 'Final potential found by TPI')
plt.plot(r, potential(r), color= 'blue', alpha = 0.7, label = 'Actual potential')


plt.title('TPI iterations vs real potential')
plt.ylim(-1.2,1.2)
plt.xlim(0.9,3)

plt.legend()
plt.show()