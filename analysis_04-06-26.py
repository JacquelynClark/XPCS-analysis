"""
April 10th, 2026
 Jacquelyn Clark 
 Takes in files, bins the data in time into approximately one second intervals, plots a 2D image, of x vs y 0 to 512 with a color bar legend.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import PyMca5.tests

def load_data(file_path):
    """Load a two-column .t3pa file, skipping the header line.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        ids (np.ndarray): 1D array of encoded pixel IDs.
        times_ns (np.ndarray): 1D array of timestamps in nanoseconds.
    """
    data = np.loadtxt(file_path, skiprows=1)

    ids = data[:, 0].astype(int)

    t = data[:, 1]          # raw time
    times_ns = t * 200      # convert to nanoseconds

    return ids, times_ns

def create_time_bins(times_ns, bin_width_ns=9e9):
    """Create time bins of approximately 1 second.

    Parameters:
        times_ns (np.ndarray): Array of timestamps in nanoseconds.
        bin_width_ns (float): Desired bin width in nanoseconds (default = 1 second).

    Returns:
        bins (list): List of arrays of indices belonging to each bin.
        bin_edges (np.ndarray): Edges used for binning.
    """
    start_time = times_ns.min()
    end_time = times_ns.max()

    # Create bin edges
    bin_edges = np.arange(start_time, end_time + bin_width_ns, bin_width_ns)

    # Digitize times into bins
    bin_indices = np.digitize(times_ns, bin_edges)

    bins = []
    for i in range(1, len(bin_edges)):
        indices = np.where(bin_indices == i)[0]
        if len(indices) > 0:
            bins.append(indices)

    return bins, bin_edges

def report_bin_durations(times_ns, bins, max_print=10):
    """Print the actual duration of the first few bins.

    Parameters:
        times_ns (np.ndarray): Array of timestamps.
        bins (list): List of index arrays for each bin.
        max_print (int): Number of bins to print.
    """
    print("\nBin duration report:")
    for i, indices in enumerate(bins[:max_print]):
        t_min = times_ns[indices].min()
        t_max = times_ns[indices].max()
        duration_sec = (t_max - t_min) * 1e-9
        #print("Time range (ns):", t_min, "to", t_max)
        print(f"Bin {i+1}: Duration = {duration_sec:.6f} seconds")

def decode_ids(ids):
    """Decode compressed IDs into (x, y) coordinates.

    The encoding is: ID = y*512 + x

    Parameters:
        ids (np.ndarray): Encoded IDs.

    Returns:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.
    """
    x = ids % 512
    y = ids // 512
    return x, y

def plot_bin(x, y, output_path, title):
    """Generate and save a 2D histogram plot for a bin.

    Parameters:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.
        output_path (str): File path to save the plot.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(6, 5))

    # 2D histogram
    hist = plt.hist2d(x, y, bins=[512, 512], range=[[0, 512], [0, 512]])

    # Colorbar shows density
    cbar = plt.colorbar()
    cbar.set_label("Counts per pixel")

    plt.xlabel("X")
    plt.ylabel("Y", rotation=0, labelpad=15)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_overall(ids, output_path, title):
    """
    Plot all events in the file on a single 2D histogram.

    Parameters:
        ids (np.ndarray): Encoded pixel IDs.
        output_path (str): Output image path.
        title (str): Plot title.
    """
    # Decode all IDs into x and y
    x = ids % 512
    y = ids // 512

    plt.figure(figsize=(6, 5))

    # 2D histogram
    plt.hist2d(x, y, bins=[512, 512], range=[[0, 512], [0, 512]])

    cbar = plt.colorbar()
    cbar.set_label("Counts per pixel")

    plt.xlabel("X")
    plt.ylabel("Y", rotation=0, labelpad=15)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_files(file_list, output_dir="binned_plots"):
    """Main processing function.

    Parameters:
        file_list (list): List of file paths to process.
        output_dir (str): Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_list:
        print(f"\nProcessing file: {file_path}")

        # Load data
        ids, times_ns = load_data(file_path)

        # Create bins
        bins, bin_edges = create_time_bins(times_ns)

        # Report bin durations
        report_bin_durations(times_ns, bins)

        # Process each bin
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Plot overall (all data, no binning)
        overall_output = os.path.join(
            output_dir,
            f"{base_name}_overall.png"
        )

        plot_overall(
            ids,
            overall_output,
            title=f"{base_name} - Overall"
        )

        for i, indices in enumerate(bins):
            bin_ids = ids[indices]

            # Decode to x, y
            x, y = decode_ids(bin_ids)

            # Output filename
            output_filename = f"{base_name}_bin_{i+1:02d}.png"
            output_path = os.path.join(output_dir, output_filename)

            # Plot
            plot_bin(
                x,
                y,
                output_path,
                title=f"{base_name} - Bin {i+1}"
            )

if __name__ == "__main__":
    files = [
        r"C:\Users\roadr\Downloads\LBNL Projects\FCGT\data\FCGT_data_12-15-24\TimePix3\300K_1800s_2.t3pa"
        
        # Add more files here manually
        ]

    process_files(files)

