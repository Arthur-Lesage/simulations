import numpy as np
from matplotlib import pyplot as plt
UM = 1e-6  # micrometer
def make_plot(g, r, data, averaged_frames, filename):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0.06, 0.08, 0.70, 0.88])
    info_ax = fig.add_axes([0.78, 0.03, 0.20, 0.88])
    info_ax.axis("off")

    ax.set_title('Pair Correlation', fontsize=20)
    ax.set_xlim(0,max(r))
    ax.set_ylim(0,max(g))
    ax.set_xlabel("r [m]")
    ax.set_ylabel("g(r)")

    # Read every saved variable automatically
    metadata = {}

    for key in data.files:
        if key == "pos":
            continue

        value = data[key]

        if np.ndim(value) == 0:
            value = value.item()

        metadata[key] = value

    parameter_string = ""

    for key, value in metadata.items():

        if isinstance(value, float):
            parameter_string += f"{key:<28} {value:.4g}\n"
        else:
            parameter_string += f"{key:<28} {value}\n"
    parameter_string += f"{'AVERAGED FRAMES':<28} {averaged_frames:.4g}"

    info_ax.text(
        0,
        1,
        parameter_string,
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black")
    )

    ax.plot(r, g)

    fig.savefig(f'pair correlation functions/{filename}')

    print('Done!')


def get_pair_correlation_of_last_frames(filename, bin_size = 10**-1, amount_of_frames = 1):

    print(f"\rGetting pair correlation function of {filename[:-4]}")

    bin_size *= UM
    data = np.load(f"position data/{filename}", allow_pickle=True)
    box_size = data["BOX_SIZE"]
    N = data["AMOUNT_OF_PARTICLES"]

    r_max = box_size / 2
    bins = np.arange(0, r_max + bin_size, bin_size)

    pos = data["pos"]
    hist_total = np.zeros(len(bins) - 1)
    for frame in pos[-amount_of_frames:]:
        frame = frame.T

        # get distances between particles:
        diff = frame[:, None, :] - frame[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=-1)
        d = dist[np.triu_indices_from(dist, k=1)]

        # make histogram
        hist, edges = np.histogram(d, bins=bins)
        hist_total += hist
    hist_total /= amount_of_frames
    dr = edges[1] - edges[0]
    r = edges[:-1] + dr / 2
    g = hist_total * box_size ** 2 / (np.pi * r * dr * N ** 2)

    make_plot(g, r, data, amount_of_frames, f"{filename[:-4]}.png")
