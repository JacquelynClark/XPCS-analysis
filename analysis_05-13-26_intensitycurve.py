"""
May 13th, 2026
 Jacquelyn Clark 
TimePix3 Analysis Script
Robust Version for .t3pa Files
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

DETECTOR_SIZE = 512

TIME_BIN_SECONDS = 30

OUTPUT_DIR = os.path.join(r"C:\Users\roadr\Downloads\LBNL Projects\FCGT", "processed_plots")

# ============================================================
# OUTPUT FOLDERS
# ============================================================

overall_dir = os.path.join(OUTPUT_DIR, "overall_images")
timebin_dir = os.path.join(OUTPUT_DIR, "time_binned_images")
tempplot_dir = os.path.join(OUTPUT_DIR, "temperature_plots")

os.makedirs(overall_dir, exist_ok=True)
os.makedirs(timebin_dir, exist_ok=True)
os.makedirs(tempplot_dir, exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def decode_xy(packed):

    x = packed % DETECTOR_SIZE
    y = packed // DETECTOR_SIZE

    return x, y


def extract_temperature(filename):

    match = re.search(r'(\d+)K', filename)

    if match:
        return int(match.group(1))

    return None


def create_image(x, y):

    image, _, _ = np.histogram2d(
        y,
        x,
        bins=[DETECTOR_SIZE, DETECTOR_SIZE],
        range=[[0, DETECTOR_SIZE], [0, DETECTOR_SIZE]]
    )

    return image


def load_t3pa_file(filepath):
    """
    Robust loader for malformed .t3pa files.

    Ignores the header and only reads the first
    two tab-separated columns from each line.
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

            # Require at least 2 columns
            if len(parts) < 2:
                continue

            try:
                packed.append(int(parts[0]))
                time_raw.append(int(parts[1]))

            except ValueError:
                continue

    return np.array(packed), np.array(time_raw)


# ============================================================
# FIND VALID FILES
# ============================================================

t3pa_files = sorted(
    [
        f for f in Path(DATA_DIR).iterdir()
        if f.is_file() and f.suffix == ".t3pa"
    ]
)

print(f"\nFound {len(t3pa_files)} valid .t3pa files")

for f in t3pa_files:
    print(f.name)

# ============================================================
# MAIN PROCESSING
# ============================================================

temperature_values = []
intensity_values = []

for file_path in t3pa_files:

    print(f"\nProcessing {file_path.name}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    packed, time_raw = load_t3pa_file(file_path)

    if len(packed) == 0:
        print("No valid data found")
        continue

    print(f"Loaded {len(packed)} events")

    # --------------------------------------------------------
    # DECODE PIXELS
    # --------------------------------------------------------

    x, y = decode_xy(packed)

    # --------------------------------------------------------
    # TIME CONVERSION
    # --------------------------------------------------------

    time_ns = time_raw * 200

    time_seconds = time_ns * 1e-9

    # --------------------------------------------------------
    # TEMPERATURE + INTENSITY
    # --------------------------------------------------------

    temperature = extract_temperature(file_path.name)

    total_intensity = len(x)

    if temperature is not None:

        temperature_values.append(temperature)
        intensity_values.append(total_intensity)

    # --------------------------------------------------------
    # OVERALL IMAGE
    # --------------------------------------------------------

    overall_image = create_image(x, y)

    plt.figure(figsize=(8, 8))

    plt.imshow(
        overall_image,
        origin="lower",
        aspect="equal"
    )

    plt.colorbar(label="Counts")

    plt.title(f"{file_path.stem}\nOverall Detector Image")

    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")

    save_path = os.path.join(
        overall_dir,
        f"{file_path.stem}_overall.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    print("Saved overall image")

    # --------------------------------------------------------
    # TIME BINNED IMAGES
    # --------------------------------------------------------

    max_time = np.max(time_seconds)

    bins = np.arange(
        0,
        max_time + TIME_BIN_SECONDS,
        TIME_BIN_SECONDS
    )

    for i in range(len(bins) - 1):

        t0 = bins[i]
        t1 = bins[i + 1]

        mask = (time_seconds >= t0) & (time_seconds < t1)

        if np.sum(mask) == 0:
            continue

        image = create_image(
            x[mask],
            y[mask]
        )

        plt.figure(figsize=(8, 8))

        plt.imshow(
            image,
            origin="lower",
            aspect="equal"
        )

        plt.colorbar(label="Counts")

        plt.title(
            f"{file_path.stem}\n"
            f"{t0:.0f}s to {t1:.0f}s"
        )

        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")

        save_name = (
            f"{file_path.stem}_"
            f"{int(t0)}s_{int(t1)}s.png"
        )

        save_path = os.path.join(
            timebin_dir,
            save_name
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

    print("Saved time-binned images")

# ============================================================
# TEMPERATURE VS INTENSITY
# ============================================================

if len(temperature_values) > 0:

    temperature_values = np.array(temperature_values)
    intensity_values = np.array(intensity_values)

    sort_idx = np.argsort(temperature_values)

    temperature_values = temperature_values[sort_idx]
    intensity_values = intensity_values[sort_idx]

    plt.figure(figsize=(8, 6))

    plt.plot(
        temperature_values,
        intensity_values,
        marker="o"
    )

    plt.xlabel("Temperature (K)")
    plt.ylabel("Total Intensity")

    plt.title("Temperature vs Intensity")

    plt.grid(True)

    save_path = os.path.join(
        tempplot_dir,
        "temperature_vs_intensity.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    print("\nSaved temperature plot")

print("\nProcessing complete")