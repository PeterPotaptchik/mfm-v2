import matplotlib.pyplot as plt
import re
import os
import numpy as np
from collections import defaultdict

# --- Core Data Parsing Function ---
def parse_data(file_path):
    """
    Parses a fid_scores.txt file to extract time, steps, and FID score.

    Args:
        file_path (str): The full path to the fid_scores.txt file.

    Returns:
        dict: A nested dictionary structured as {time: {steps: fid_score}}.
              Returns an empty dictionary if the file is not found.
    """
    data = defaultdict(dict)
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Skipping.")
        return {}

    # Regex to capture t, steps, and the FID score.
    # It captures floating point numbers for t and the score, and an integer for steps.
    pattern = re.compile(r"t_([\d.]+)_const_(\d+)_True: ([\d.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                t_val = float(match.group(1))
                steps_val = int(match.group(2))
                fid_score = float(match.group(3))
                data[t_val][steps_val] = fid_score
    
    # Convert defaultdict to a regular dict for cleaner output, though not strictly necessary
    return dict(data)

# --- Main Plotting Function ---
def plot_fid_grid(timestamps, legend_names, base_path="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/"):
    """
    Generates and displays a single row of plots for FID scores.

    Each subplot in the row corresponds to a specific time t. Each line within
    a subplot corresponds to a different experiment run (from the provided timestamps).

    Args:
        timestamps (list[str]): A list of the timestamp strings,
                                e.g., ["2025-08-08/17-48-55", ...].
        legend_names (list[str]): A list of names for the legend, corresponding
                                  to each timestamp.
        base_path (str): The constant base path to the output directories.
    """
    if not timestamps or not legend_names or len(timestamps) != len(legend_names):
        raise ValueError("Timestamps and legend_names must be non-empty and of the same length.")

    all_experiments_data = {}
    all_times = set()
    all_steps = set()

    # --- Data Loading and Aggregation ---
    for timestamp, name in zip(timestamps, legend_names):
        full_path = os.path.join(base_path, timestamp, "fid_scores.txt")
        experiment_data = parse_data(full_path)

        if not experiment_data:
            continue

        all_experiments_data[name] = experiment_data
        
        # Aggregate all unique time and step values to build the plot axes
        for t, steps_data in experiment_data.items():
            all_times.add(t)
            for step in steps_data.keys():
                all_steps.add(step)
    
    if not all_experiments_data:
        print("No data was loaded or generated. Cannot create a plot.")
        return

    sorted_times = sorted(list(all_times))
    sorted_steps = sorted(list(all_steps))

    # --- Plotting Setup ---
    # Create a single row of plots, one for each time point.
    ncols = len(sorted_times)
    nrows = 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, 5), sharey=False, squeeze=False)
    fig.suptitle('FID Score vs. Number of Steps at Different Times', fontsize=20, y=1.02)

    # Use a colormap to automatically get distinct colors for each experiment
    colors = plt.cm.viridis(np.linspace(0, 1, len(legend_names)))
    
    # --- Plot Generation ---
    for i, t_val in enumerate(sorted_times):
        ax = axes[0, i] # Index into the single row of axes

        for j, name in enumerate(legend_names):
            if name not in all_experiments_data:
                continue

            experiment_data = all_experiments_data[name]
            if t_val in experiment_data:
                # Get the FID scores for the current time t_val, ordered by the number of steps
                steps_data = experiment_data[t_val]
                y_values = [steps_data.get(step, np.nan) for step in sorted_steps] # Use NaN for missing data
                
                ax.plot(sorted_steps, y_values, marker='o', linestyle='-', label=name, color=colors[j])

        ax.set_title(f't = {t_val}', fontsize=14)
        ax.set_xlabel('Number of Steps', fontsize=10)
        ax.set_xticks(sorted_steps) # Ensure all step values are shown as ticks
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_ylabel('FID Score', fontsize=10) 
        
    # --- Aesthetics and Cleanup ---
    # Create a single, shared legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_names), fontsize=12, title="Experiments")

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle and legend
    # Construct the full path for the output file
    save_path = os.path.join(base_path, "fid_scores.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    print(f"Plot successfully saved to: {save_path}")
    plt.close(fig) # Optional: closes the figure to free up memory


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Define the unique parts of your file paths
    experiment_timestamps = [
        "2025-08-08/17-48-55",
        "2025-08-08/17-52-04",
        "2025-08-08/17-52-49",
        "2025-08-08/17-59-17",
        "2025-08-11/18-00-02"
    ]

    # 2. Define the corresponding names for the plot legend
    experiment_names = [
        "default",
        "t_cond_anneal_50k",
        "explicit_v00_train",
        "no_weight_on_t_cond",
        # "use_noise_parametrization"
    ]

    # 3. Call the plotter function
    # The base_path argument is optional if your script is in a location
    # where the default path is correct.
    plot_fid_grid(experiment_timestamps, experiment_names)
