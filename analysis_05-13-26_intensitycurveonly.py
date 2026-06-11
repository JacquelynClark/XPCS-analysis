"""
April 28th, 2026
Jacquelyn Clark
TimePix3 Intensity Analysis
---------------------------

This script ONLY analyzes intensity and creates several
temperature vs intensity plots.

Intensity definitions included:
1. Total counts
2. Log10(total counts)
3. Counts normalized to max
4. Counts per second

Plots are automatically saved to new folder 'intensity_analysis'.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================

DATA_DIR = r"C:\Users\roadr\Downloads\LBNL Projects\FCGT\data\FCGT_data_12-15-24\TimePix3"
PLOT_DIR = r"C:\Users\roadr\Downloads\LBNL Projects\FCGT"
OUTPUT_DIR = os.path.join(PLOT_DIR, "intensity_analysis")

TIME_CONVERSION_NS = 200  # time*200 = ns

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def extract_temperature(filename):
    """
    Extract temperature from filename.

    Example:
        sample_300K_run1.t3pa -> 300
    """

    match = re.search(r'(\d+)K', filename)

    if match:
        return int(match.group(1))

    return None


def load_t3pa_file(filepath):
    """
    Load first two columns from TimePix3 file.
    """

    packed = []
    time_raw = []

    with open(filepath, "r") as f:

        # Skip header row
        next(f)

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            try:
                packed.append(int(parts[0]))
                time_raw.append(int(parts[1]))

            except ValueError:
                continue

    return np.array(packed), np.array(time_raw)


# ============================================================
# FIND FILES
# ============================================================

t3pa_files = sorted(
    [
        f for f in Path(DATA_DIR).iterdir()
        if f.is_file() and f.suffix == ".t3pa"
    ]
)

print(f"\nFound {len(t3pa_files)} .t3pa files")

# ============================================================
# STORAGE
# ============================================================

temperatures = []

total_counts = []

counts_per_second = []

# ============================================================
# MAIN LOOP
# ============================================================

for file_path in t3pa_files:

    print(f"\nProcessing {file_path.name}")

    temperature = extract_temperature(file_path.name)

    if temperature is None:

        print("No temperature found in filename")
        continue

    packed, time_raw = load_t3pa_file(file_path)

    if len(packed) == 0:

        print("No valid data")
        continue

    # --------------------------------------------------------
    # TOTAL COUNTS
    # --------------------------------------------------------

    counts = len(packed)

    # --------------------------------------------------------
    # TOTAL MEASUREMENT TIME
    # --------------------------------------------------------

    time_ns = time_raw * TIME_CONVERSION_NS

    total_time_seconds = (
        np.max(time_ns) - np.min(time_ns)
    ) * 1e-9

    if total_time_seconds <= 0:
        total_time_seconds = 1

    cps = counts / total_time_seconds

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    temperatures.append(temperature)

    total_counts.append(counts)

    counts_per_second.append(cps)

# ============================================================
# CONVERT TO ARRAYS
# ============================================================

temperatures = np.array(temperatures)

total_counts = np.array(total_counts)

counts_per_second = np.array(counts_per_second)

# Sort by temperature
sort_idx = np.argsort(temperatures)

temperatures = temperatures[sort_idx]

total_counts = total_counts[sort_idx]

counts_per_second = counts_per_second[sort_idx]

# ============================================================
# DERIVED INTENSITIES
# ============================================================

log_counts = np.log10(total_counts)

normalized_counts = total_counts / np.max(total_counts)

# ============================================================
# PLOT FUNCTION
# ============================================================

def save_plot(x, y, ylabel, title, filename):

    plt.figure(figsize=(8, 6))

    plt.plot(
        x,
        y,
        marker='o'
    )

    plt.xlabel("Temperature (K)")

    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(True)

    save_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved {filename}")


# ============================================================
# SAVE PLOTS
# ============================================================

save_plot(
    temperatures,
    total_counts,
    "Total Counts",
    "Temperature vs Total Counts",
    "temperature_vs_total_counts.png"
)

save_plot(
    temperatures,
    log_counts,
    "log10(Total Counts)",
    "Temperature vs log10(Intensity)",
    "temperature_vs_log_counts.png"
)

save_plot(
    temperatures,
    normalized_counts,
    "Normalized Intensity",
    "Temperature vs Normalized Intensity",
    "temperature_vs_normalized_counts.png"
)

save_plot(
    temperatures,
    counts_per_second,
    "Counts Per Second",
    "Temperature vs Counts Per Second",
    "temperature_vs_counts_per_second.png"
)

# ============================================================
# SAVE DATA FILE
# ============================================================

output_txt = os.path.join(
    OUTPUT_DIR,
    "intensity_results.txt"
)

with open(output_txt, "w") as f:

    f.write(
        "Temperature(K)\t"
        "TotalCounts\t"
        "Log10Counts\t"
        "NormalizedCounts\t"
        "CountsPerSecond\n"
    )

    for i in range(len(temperatures)):

        f.write(
            f"{temperatures[i]}\t"
            f"{total_counts[i]}\t"
            f"{log_counts[i]}\t"
            f"{normalized_counts[i]}\t"
            f"{counts_per_second[i]}\n"
        )

print("\nSaved intensity_results.txt")

print("\nIntensity analysis complete")