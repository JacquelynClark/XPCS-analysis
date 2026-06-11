import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def load_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)

    ids = data[:, 0].astype(int)

    t = data[:, 1]
    times_ns = t * 200  # convert to nanoseconds

    return ids, times_ns

def create_time_bins(times_ns, bin_width_ns=9e9):
    start_time = times_ns.min()
    end_time = times_ns.max()

    bin_edges = np.arange(start_time, end_time + bin_width_ns, bin_width_ns)
    bin_indices = np.digitize(times_ns, bin_edges)

    bins = []
    for i in range(1, len(bin_edges)):
        indices = np.where(bin_indices == i)[0]
        if len(indices) > 0:
            bins.append(indices)

    return bins, bin_edges

def report_bin_durations(times_ns, bins, max_print=10):
    print("\nBin duration report:")
    for i, indices in enumerate(bins[:max_print]):
        t_min = times_ns[indices].min()
        t_max = times_ns[indices].max()
        duration_sec = (t_max - t_min) * 1e-9
        print(f"Bin {i+1}: Duration = {duration_sec:.6f} seconds")

def decode_ids(ids):
    x = ids % 512
    y = ids // 512
    return x, y

def plot_overall(ids, output_path, title):
    x = ids % 512
    y = ids // 512

    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=[512, 512], range=[[0, 512], [0, 512]])

    cbar = plt.colorbar()
    cbar.set_label("Counts per pixel")

    plt.xlabel("X")
    plt.ylabel("Y", rotation=0, labelpad=15)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_movie(ids, times_ns, bins, output_path, fps=5):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Initialize empty image
    img = ax.imshow(
        np.zeros((512, 512)),
        origin='lower',
        vmin=0,
        vmax=1000,   # contrast fixing
        extent=[0, 512, 0, 512]
    )

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Counts per pixel")

    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation=0, labelpad=15)

    def update(frame):
        indices = bins[frame]
        bin_ids = ids[indices]

        x = bin_ids % 512
        y = bin_ids // 512

        # Compute histogram manually
        hist, _, _ = np.histogram2d(
            x, y,
            bins=[512, 512],
            range=[[0, 512], [0, 512]]
        )

        img.set_data(hist.T)  # transpose for correct orientation

        # Time labeling
        t_min = times_ns[indices].min() * 1e-9
        t_max = times_ns[indices].max() * 1e-9

        ax.set_title(f"{t_min:.2f}s - {t_max:.2f}s")

        return [img]

    anim = FuncAnimation(fig, update, frames=len(bins), blit=True)

    writer = FFMpegWriter(fps=fps)
    anim.save(output_path, writer=writer)

    plt.close()

def process_files(file_list, output_dir="binned_plots"):
    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_list:
        print(f"\nProcessing file: {file_path}")

        ids, times_ns = load_data(file_path)

        bins, bin_edges = create_time_bins(times_ns)

        report_bin_durations(times_ns, bins)

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Overall plot 
        overall_output = os.path.join(
            output_dir,
            f"{base_name}_overall.png"
        )

        plot_overall(
            ids,
            overall_output,
            title=f"{base_name} - Overall"
        )

        # Create movie
        movie_output = os.path.join(
            output_dir,
            f"{base_name}_movie.mp4"
        )

        create_movie(ids, times_ns, bins, movie_output, fps=5)

if __name__ == "__main__":
    files = [
        r"C:\Users\roadr\Downloads\LBNL Projects\FCGT\data\FCGT_data_12-15-24\TimePix3\250K_1800s.t3pa"
        
    ]

    process_files(files)