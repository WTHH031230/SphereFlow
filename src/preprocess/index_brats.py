import os
import json
import glob
from tqdm import tqdm

# ================= Configuration =================
# Path where the processed .npy files are stored
# (the output of preprocess_brats.py)
DATA_ROOT = os.path.join(os.path.abspath(os.environ.get("DATA_ROOT", "./datasets")), "brats")

# Output path for the JSON file
OUTPUT_DIR = os.environ.get("SPLIT_OUT_DIR", "./data")
JSON_FILENAME = "brats2023_distribution.json"
# ===============================================

def generate_dataset_json():
    """
    Scans the processed data directory and generates a JSON file listing all samples.
    Structure:
    {
        "train": [ {"path": "...", "case": "...", "slice_idx": ...}, ... ],
        "val":   [ ... ],
        "test":  [ ... ]
    }
    """
    
    dataset_dict = {
        "train": [],
        "val": [],
        "test": []
    }
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root {DATA_ROOT} does not exist.")
        return

    print(f"Scanning data from: {DATA_ROOT}")

    for phase in dataset_dict.keys():
        phase_dir = os.path.join(DATA_ROOT, phase)
        
        if not os.path.exists(phase_dir):
            print(f"Warning: Directory for {phase} set not found at {phase_dir}")
            continue
            
        file_pattern = os.path.join(phase_dir, "*.npy")
        file_list = sorted(glob.glob(file_pattern))
        
        print(f"Processing {phase} set: found {len(file_list)} files.")
        
        for file_path in tqdm(file_list, desc=f"Indexing {phase}"):
            try:
                # Get filename: "BraTS-GLI-XXXXX-XXX_040.npy"
                filename = os.path.basename(file_path)
                
                name_no_ext = os.path.splitext(filename)[0]
                
                # Split by the last underscore to separate Case Name and Slice Index
                # Assumption: Filename format is "{case_name}_{slice_idx}.npy"
                last_underscore_index = name_no_ext.rfind('_')
                
                if last_underscore_index != -1:
                    case_name = name_no_ext[:last_underscore_index]
                    slice_idx = int(name_no_ext[last_underscore_index + 1:])
                else:
                    # Fallback if the name does not match {case_name}_{slice_idx}.npy
                    case_name = name_no_ext
                    slice_idx = -1
                
                sample_entry = {
                    "path": os.path.abspath(file_path),
                    "case": case_name,
                    "slice_idx": slice_idx
                }
                
                dataset_dict[phase].append(sample_entry)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    output_path = os.path.join(OUTPUT_DIR, JSON_FILENAME)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(dataset_dict, f, indent=4)
        
        print(f"\nSuccess! JSON generated at: {output_path}")
        print(f"Summary:")
        print(f"  Train samples: {len(dataset_dict['train'])}")
        print(f"  Val samples:   {len(dataset_dict['val'])}")
        print(f"  Test samples:  {len(dataset_dict['test'])}")
        
    except IOError as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    generate_dataset_json()