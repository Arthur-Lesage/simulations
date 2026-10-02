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

for potential in [lennard_jones, hard_disk, attractive_square_well, repulsive_square_well, attractive_yukawa, repulsive_yukawa]:
    filename = potential.__name__

    TPI_data = np.load(f"TPI result data/{filename}.npz", allow_pickle=True)
    hist = TPI_data['POTENTIAL_HISTORY']
    r = TPI_data['r']

    plt.plot(r, hist[-1], linewidth = 4, color = 'red', alpha = 1, label = 'Final potential found by TPI')
    plt.plot(r, hist[0], color = 'green', alpha = 0.7, label = 'Initial guess')
    plt.plot(r, potential(r), color= 'blue', alpha = 0.7, label = 'Simulated potential')

    plt.title(f'TPI vs simulated {filename.replace('_', ' ')} potential')
    plt.xlabel('r')
    plt.ylabel('Potential')
    plt.ylim(-1.2,1.2)
    plt.xlim(0.9,3)

    plt.legend()
    #plt.show()
    plt.savefig(f'TPI result plots/{filename.replace('_', ' ')}', bbox_inches = 'tight')
    plt.close()