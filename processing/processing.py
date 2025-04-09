import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Get directories from config.yaml
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

rois_cc200_dir = config["rois_cc200_dir"]
unified_array_file = config["unified_array_file"]
corr_matrices_file = config["corr_matrices_file"]


# Define Directory and File Pattern
file_pattern = "*.1D"
target_shape = (884, 146, 200)

# Get list of all .1D files
file_list = glob(os.path.join(rois_cc200_dir, file_pattern))
print(f"Found {len(file_list)} .1D files in {rois_cc200_dir}")

# Cell: Function to Load and Standardize a .1D File
def load_and_standardize_1d(filepath, target_timepoints=146):
    try:
        # Load the .1D file into a NumPy array
        data = np.loadtxt(filepath)
        
        # Verify 200 ROIs
        if data.shape[1] != 200:
            print(f"Warning: {os.path.basename(filepath)} has {data.shape[1]} columns, expected 200")
            return None
        
        # Standardize to 146 time points
        current_timepoints = data.shape[0]
        if current_timepoints == target_timepoints:
            return data
        elif current_timepoints > target_timepoints:
            # Truncate to first 146 time points
            return data[:target_timepoints, :]
        else:
            # Pad with zeros to reach 146 time points
            padding = np.zeros((target_timepoints - current_timepoints, 200))
            return np.vstack((data, padding))
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return None

# Cell: Create the Unified Array
# Initialize the array with zeros
unified_array = np.zeros(target_shape)

# Process each file and fill the array
for i, filepath in enumerate(file_list[:target_shape[0]]):  # Limit to 884 files
    standardized_data = load_and_standardize_1d(filepath, target_timepoints=target_shape[1])
    if standardized_data is not None:
        unified_array[i] = standardized_data
        print(f"Added {os.path.basename(filepath)} at index {i}: {standardized_data.shape}")
    else:
        print(f"Skipped {os.path.basename(filepath)} due to errors")

# Cell: Verify the Result
print(f"\nFinal array shape: {unified_array.shape}")
print(f"First participant, first 5 time points, first 5 ROIs:\n{unified_array[0, :5, :5]}")
print(f"Last participant, last 5 time points, last 5 ROIs:\n{unified_array[-1, -5:, -5:]}")

# Cell: Save the Array to Disk
np.save(unified_array_file, unified_array)
print(f"Saved array to {unified_array_file}")

# Cell: Load and Verify Saved Array
loaded_array = np.load(unified_array_file)
print(f"Loaded array shape: {loaded_array.shape}")
print(f"Arrays match: {np.array_equal(unified_array, loaded_array)}")

# Cell: Function to Compute Correlation Matrix for One Participant
# Compute Pearson correlation matrix using numpy's corrcoef
# corrcoef expects rows as variables (ROIs) and columns as observations (time points),
# so we transpose the input (200, 146)
def compute_correlation_matrix(time_series):
    """
    Compute the 200x200 correlation matrix from a 146x200 time series matrix.
    
    Parameters:
    - time_series: np.ndarray of shape (146, 200) - time points x ROIs
    
    Returns:
    - corr_matrix: np.ndarray of shape (200, 200) - correlation matrix
    """

    # Check for std == 0 columns (constant time series)
    stds = np.std(time_series, axis=0)
    valid_mask = stds > 1e-6

    # If participant has bad .1D file, print
    if np.isnan(time_series).any():
        print(f"⚠️ Participant {i} has NaNs in time series BEFORE correlation")

    # Fallback: if too many bad columns, skip
    if valid_mask.sum() < 10:
        print(f"⚠️ Participant {i}: Too many zero-std ROIs")

    # Masked time series
    clean_series = time_series[:, valid_mask]

    # Compute correlation matrix on cleaned ROIs
    partial_corr = np.corrcoef(clean_series.T)

    # Fill full 200x200 matrix with NaNs and place partial result in it
    corr_matrix = np.full((200, 200), np.nan)
    idx = np.where(valid_mask)[0]
    for a, ai in enumerate(idx):
        for b, bi in enumerate(idx):
            corr_matrix[ai, bi] = partial_corr[a, b]
    
    # Ensure the output is 200x200 (it should be by default)
    return corr_matrix

# Cell: Compute Correlation Matrices for All Participants
# Initialize array to store correlation matrices
num_participants = loaded_array.shape[0]  # 884
corr_matrices = np.zeros((num_participants, 200, 200))

# Process each participant's time series
for i in range(num_participants):
    corr_matrices[i] = compute_correlation_matrix(loaded_array[i])
    if (i + 1) % 100 == 0:  # Progress update every 100 participants
        print(f"Processed {i + 1}/{num_participants} participants")

print(f"Correlation matrices shape: {corr_matrices.shape}")

# Log how many matrices have NaNs
n_nan_matrices = np.sum([np.isnan(m).any() for m in corr_matrices])
print(f"⚠️ Found {n_nan_matrices} participants with NaNs in their correlation matrices.")

# Replace all NaNs with zeros or another fallback
corr_matrices_clean = np.nan_to_num(corr_matrices, nan=0.0, posinf=1.0, neginf=-1.0)

# Save the clean array
np.save(corr_matrices_file, corr_matrices_clean)

# Check if any NaNs
arr = np.load(corr_matrices_file)
print("Still has NaNs?", np.isnan(arr).any())  # Should be False
print("Shape:", arr.shape)