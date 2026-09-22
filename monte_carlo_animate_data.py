import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

##THIS WAS MOSTLY WRITTEN USING AI

def animate_positions(pos, particle_radius, metadata, box_size, interval=200):

    n_frames = pos.shape[0]
    n_particles = pos.shape[2]

    fig = plt.figure(figsize=(10, 5.8))

    ax = fig.add_axes([0.06, 0.08, 0.70, 0.88])
    info_ax = fig.add_axes([0.68, 0.03, 0.20, 0.88])

    info_ax.axis("off")

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect("equal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    sweep_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=11
    )

    # ------------------------------------------------------
    # Parameter list
    # ------------------------------------------------------

    parameter_string = ""

    for key, value in metadata.items():

        if isinstance(value, float):
            parameter_string += f"{key:<28} {value:.4g}\n"
        else:
            parameter_string += f"{key:<28} {value}\n"

    info_ax.text(
        0,
        1,
        parameter_string,
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black")
    )

    # ------------------------------------------------------
    # Create particles
    # ------------------------------------------------------

    particles = []
    ghosts = []

    for _ in range(n_particles):

        circle = plt.Circle((0, 0), particle_radius, color="blue")
        ax.add_patch(circle)
        particles.append(circle)

        ghost_set = []

        for _ in range(8):
            ghost = plt.Circle((0, 0), particle_radius, color="blue")
            ghost.set_visible(False)
            ax.add_patch(ghost)
            ghost_set.append(ghost)

        ghosts.append(ghost_set)

    # ------------------------------------------------------
    # Animation update
    # ------------------------------------------------------

    def update(frame):

        margin = 2 * particle_radius

        for i in range(n_particles):

            x = pos[frame, 0, i]
            y = pos[frame, 1, i]

            particles[i].center = (x, y)

            ghost_positions = []

            if x < margin:
                ghost_positions.append((x + box_size, y))
            if x > box_size - margin:
                ghost_positions.append((x - box_size, y))

            if y < margin:
                ghost_positions.append((x, y + box_size))
            if y > box_size - margin:
                ghost_positions.append((x, y - box_size))

            if x < margin and y < margin:
                ghost_positions.append((x + box_size, y + box_size))

            if x < margin and y > box_size - margin:
                ghost_positions.append((x + box_size, y - box_size))

            if x > box_size - margin and y < margin:
                ghost_positions.append((x - box_size, y + box_size))

            if x > box_size - margin and y > box_size - margin:
                ghost_positions.append((x - box_size, y - box_size))

            for j in range(8):

                if j < len(ghost_positions):
                    ghosts[i][j].center = ghost_positions[j]
                    ghosts[i][j].set_visible(True)
                else:
                    ghosts[i][j].set_visible(False)

        sweep_text.set_text(f"Frame: {frame}")

        return (
            particles
            + [g for sub in ghosts for g in sub]
            + [sweep_text]
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=True
    )

    return ani

def save_animation(ani, total_frames, filename, fps=30):

    def progress(frame, total):
        percent = 100 * frame / total
        print(f"\rSaving animation {filename[:-4]}: {percent:.1f}%", end="")

    ani.save(
        f'animations/{filename}',
        fps=fps,
        dpi=200,
        savefig_kwargs={"pad_inches": 0},
        progress_callback=lambda i, n: progress(i, total_frames)
    )

    print("\nDone!")

def animate(filename, fps=30):


    data = np.load(f"position data/{filename}", allow_pickle=True)

    pos = data["pos"]  # shape = (frames, 2, N)

    # Read every saved variable automatically
    metadata = {}

    for key in data.files:
        if key == "pos":
            continue

        value = data[key]

        if np.ndim(value) == 0:
            value = value.item()

        metadata[key] = value

    BOX_SIZE = metadata["BOX_SIZE"]
    a = metadata["PARTICLE_RADIUS"]
    N = metadata["AMOUNT_OF_PARTICLES"]

    ani = animate_positions(
        pos,
        particle_radius=a,
        metadata=metadata,
        box_size=BOX_SIZE,
        interval=200
    )

    save_animation(
        ani,
        len(pos),
        f"{filename[:-4]}.mp4",
        fps=fps
    )