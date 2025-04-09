import os
import pandas as pd
import numpy as np
from glob import glob
import re
import yaml

# Get directories from config.yaml
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

rois_cc200_dir = config["rois_cc200_dir"]
phenotype_csv = config["phenotype_csv"]
labels_abide_file = config["labels_abide_file"]
subject_ids_abide_file = config["subject_ids_abide_file"]
unified_array_file = config["unified_array_file"]

# Load the phenotype CSV
pheno = pd.read_csv(phenotype_csv)

# Create a mapping: subject ID → label (1 = ASD, 0 = TDC)
pheno_labels = pheno.set_index("SUB_ID")["DX_GROUP"].map({1: 1, 2: 0})

# Sorted list of .1D files (same order used to generate your correlation matrix)
file_list = sorted(glob(os.path.join(rois_cc200_dir, "*.1D")))

# Extract subject IDs from the filenames and get labels
labels = []
subject_ids = []

for path in file_list:
    filename = os.path.basename(path)

    # Extract the 7-digit subject ID using regex
    match = re.search(r"(\d{7})", filename)
    if match:
        sub_id = int(match.group(1))
        label = pheno_labels.get(sub_id)

        if pd.isna(label):
            print(f"⚠️ Subject {sub_id} not found in phenotype file.")
        else:
            subject_ids.append(sub_id)
            labels.append(label)
    else:
        print(f"❌ Could not extract subject ID from {filename}")

labels = np.array(labels)
print(f"✅ Loaded {len(labels)} labels.")

# Save for future use
np.save(labels_abide_file, labels)
np.save(subject_ids_abide_file, np.array(subject_ids))

corr_mat = np.load(unified_array_file)
labels_file = np.load(labels_abide_file)
subject_ids = np.load(subject_ids_abide_file)
print(corr_mat.shape, labels_file.shape, subject_ids.shape)
# Should print: (884, 200, 200) (884,) (884,)
