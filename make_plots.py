from matplotlib import pyplot as plt
import numpy as np

filename = '1_kT 1000_particles.npz'

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

def potential(r):
        return 4  * (1/r ** 12 - 1/r ** 6)

plt.plot(r, hist[0], color = 'green', alpha = 0.7, label = 'Initial guess')
plt.plot(r, hist[-1], color = 'red', alpha = 0.7, label = 'Final potential found by TPI')
plt.plot(r, potential(r), color= 'blue', alpha = 0.7, label = 'Actual potential')


plt.title('TPI iterations vs real potential')
plt.ylim(-1.2,1)
plt.xlim(0.9,3)

plt.legend()
plt.show()