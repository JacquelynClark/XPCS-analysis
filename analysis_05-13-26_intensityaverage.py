"""
May 13th, 2026
Jacquelyn Clark
Fast TimePix3 Peak Intensity Analysis
------------------------------------

This script:

1. Loads .t3pa files quickly
2. Builds detector images using numpy only
3. Finds:
    A) Maximum pixel intensity
    B) Maximum - average intensity
4. Extracts temperature from filename
5. Saves:
    - Temperature vs Maximum Intensity
    - Temperature vs (Maximum - Average)

Quicker version hopefully (doesn't plot the pixels).
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# USER SETTINGS

DATA_DIR = r"C:\Users\roadr\Downloads\LBNL Projects\FCGT\data\FCGT_data_12-15-24\TimePix3"
PLOT_DIR = r"C:\Users\roadr\Downloads\LBNL Projects\FCGT"

OUTPUT_DIR = os.path.join(PLOT_DIR, "fast_peak_analysis")

DETECTOR_SIZE = 512

# Small averaging region size around brightest point
ROI_SIZE = 5

# CREATE OUTPUT FOLDER

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FUNCTIONS

def extract_temperature(filename):

    match = re.search(r'(\d+)K', filename)

    if match:
        return int(match.group(1))

    return None


def load_t3pa_fast(filepath):
    """
    Faster loader.
    Only grabs first two columns.
    """

    packed = []
    time_raw = []

    with open(filepath, "r") as f:

        # Skip header
        next(f)

        for line in f:

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            try:
                packed.append(int(parts[0]))
                time_raw.append(int(parts[1]))

            except:
                continue

    return np.array(packed, dtype=np.int32)


def build_image_fast(packed):
    """
    Very fast detector image creation.

    packed = y * 512 + x
    """

    x = packed % DETECTOR_SIZE
    y = packed // DETECTOR_SIZE

    image = np.zeros(
        (DETECTOR_SIZE, DETECTOR_SIZE),
        dtype=np.int32
    )

    np.add.at(image, (y, x), 1)

    return image


def roi_average(image, x0, y0, size=5):

    half = size // 2

    xmin = max(0, x0 - half)
    xmax = min(DETECTOR_SIZE, x0 + half + 1)

    ymin = max(0, y0 - half)
    ymax = min(DETECTOR_SIZE, y0 + half + 1)

    roi = image[ymin:ymax, xmin:xmax]

    return np.mean(roi)


# FIND FILES

t3pa_files = sorted(
    [
        f for f in Path(DATA_DIR).iterdir()
        if f.is_file() and f.suffix == ".t3pa"
    ]
)

print(f"\nFound {len(t3pa_files)} .t3pa files")

# STORAGE

temperatures = []
max_values = []
max_minus_avg = []

# MAIN LOOP

for file_path in t3pa_files:

    print(f"\nProcessing {file_path.name}")

    temperature = extract_temperature(file_path.name)

    if temperature is None:

        print("No temperature found")
        continue

    # LOAD DATA
    
    packed = load_t3pa_fast(file_path)

    if len(packed) == 0:

        print("No valid data")
        continue

    # BUILD IMAGE
    
    image = build_image_fast(packed)

    # FIND MAXIMUM
    
    max_intensity = np.max(image)

    max_position = np.unravel_index(
        np.argmax(image),
        image.shape
    )

    y_peak, x_peak = max_position

    # LOCAL AVERAGE AROUND PEAK
    
    local_average = roi_average(
        image,
        x_peak,
        y_peak,
        size=ROI_SIZE
    )

    # MAX - AVERAGE
    
    peak_minus_average = max_intensity - local_average

    # STORE
    
    temperatures.append(temperature)

    max_values.append(max_intensity)

    max_minus_avg.append(peak_minus_average)

    print(f"Max intensity: {max_intensity}")

# SORT

temperatures = np.array(temperatures)

max_values = np.array(max_values)

max_minus_avg = np.array(max_minus_avg)

sort_idx = np.argsort(temperatures)

temperatures = temperatures[sort_idx]

max_values = max_values[sort_idx]

max_minus_avg = max_minus_avg[sort_idx]

# PLOT FUNCTION

def save_plot(y, ylabel, title, filename):

    plt.figure(figsize=(8, 6))

    plt.plot(
        temperatures,
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

# SAVE PLOTS

save_plot(
    max_values,
    "Maximum Pixel Intensity",
    "Temperature vs Maximum Intensity",
    "temperature_vs_maximum_intensity.png"
)

save_plot(
    max_minus_avg,
    "Maximum - Local Average",
    "Temperature vs Peak Contrast",
    "temperature_vs_peak_minus_average.png"
)

# SAVE NUMERICAL RESULTS

results_path = os.path.join(
    OUTPUT_DIR,
    "peak_results.txt"
)

with open(results_path, "w") as f:

    f.write(
        "Temperature(K)\t"
        "MaxIntensity\t"
        "MaxMinusAverage\n"
    )

    for i in range(len(temperatures)):

        f.write(
            f"{temperatures[i]}\t"
            f"{max_values[i]}\t"
            f"{max_minus_avg[i]}\n"
        )

print("\nSaved numerical results")

print("\nPeak intensity analysis complete")