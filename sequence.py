from monte_carlo_particle_simulation import simulate
from monte_carlo_animate_data import animate
from pair_correlation import get_pair_correlation_of_last_frames
from datetime import datetime

start_time = datetime.now()
print(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

box_size = 100
amount_of_particles = 1000
for i in [0, 0.5, 3, 10, 50]:
    interaction_strength = i

    simulate(amount_of_particles=amount_of_particles,
             interaction_strength=interaction_strength,
             frames=1800, box_size=box_size,
             monte_carlo_steps_per_frame=10000,
             filename=f'{interaction_strength}_kT {amount_of_particles}_particles')
    animate(f'{interaction_strength}_kT {amount_of_particles}_particles.npz')
    #get_pair_correlation_of_last_frames(f'{interaction_strength}_kT {amount_of_particles}_particles.npz', amount_of_frames=200)

end_time = datetime.now()
print(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {end_time - start_time}")