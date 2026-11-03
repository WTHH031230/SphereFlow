import os
import json
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from monai.transforms import ResizeWithPadOrCropd

# Configuration
# 2D SynthRAD2023 (brain) slices with sub-folders A_norm (MRI), B_norm (CT), C_norm (mask), values in [-1, 1]
DATA_DIR = os.environ.get("MRICT_RAW_DIR", "./raw_data/SynthRAD2023_brain/train2D")
SAVE_DIR = os.path.join(os.path.abspath(os.environ.get("DATA_ROOT", "./datasets")), "mri_ct")
JSON_SAVE_PATH = os.environ.get("MRICT_JSON", "./data/brain_mri_ct_distribution.json")
TARGET_SHAPE = (256, 256)
SEED = 42
NUM_WORKERS = 32

def process_file(file_name, output_dir_base):
    try:
        # Load modalities A(MRI) and B(CT), C(Mask)
        img_a_path = os.path.join(DATA_DIR, 'A_norm', file_name)
        img_b_path = os.path.join(DATA_DIR, 'B_norm', file_name)
        img_c_path = os.path.join(DATA_DIR, 'C_norm', file_name)
        
        if not (os.path.exists(img_a_path) and os.path.exists(img_b_path) and os.path.exists(img_c_path)):
            return None
            
        img_a = np.load(img_a_path)
        img_b = np.load(img_b_path)
        img_c = np.load(img_c_path)
        
        # Filter out NaN or Inf
        if not (np.isfinite(img_a).all() and np.isfinite(img_b).all() and np.isfinite(img_c).all()):
            # Returning None skips this file
            return None
        
        # 1. Map [-1, 1] to [0, 1]
        img_a = (img_a.astype(np.float32) + 1.0) / 2.0
        img_b = (img_b.astype(np.float32) + 1.0) / 2.0
        
        # Map label C from [-1, 1] to [0, 1] to prevent uint8 casting warning
        label = (img_c.astype(np.float32) + 1.0) / 2.0

        # 2. Stack A and B channels-first: [2, H, W]
        if img_a.ndim == 2:
            img_a = np.expand_dims(img_a, axis=0) # [1, H, W]
        else:
            img_a = img_a.reshape(1, img_a.shape[-2], img_a.shape[-1])
            
        if img_b.ndim == 2:
            img_b = np.expand_dims(img_b, axis=0)
        else:
            img_b = img_b.reshape(1, img_b.shape[-2], img_b.shape[-1])
            
        if label.ndim == 2:
            label = np.expand_dims(label, axis=0)
        else:
            label = label.reshape(1, label.shape[-2], label.shape[-1])

        # Create multi-modality 2-channel image
        # Modality A (MRI) -> index 0, Modality B (CT) -> index 1
        c, h, w = img_a.shape
        images = np.zeros((2, h, w), dtype=np.float32)
        
        images[0, :, :] = img_a[0, :, :]
        images[1, :, :] = img_b[0, :, :]

        data = {
            'image': images,       # Shape: [2, H, W]
            'label': label         # Shape: [1, H, W]
        }

        # 3. Apply ResizeWithPadOrCropd to pad to 256x256 (centered automatically)
        transform = ResizeWithPadOrCropd(
            keys=['image', 'label'], 
            spatial_size=TARGET_SHAPE, 
            mode='constant',
            value=0
        )
        
        data = transform(data)
        
        # Same layout as preprocess_brats.py
        save_dict = {
            'image': data['image'].astype(np.float32),  # (2, 256, 256)
            'label': data['label'].astype(np.uint8)     # (1, 256, 256)
        }
        # Squeeze the label to [H, W]; the datasets add the channel dim themselves
        save_dict['label'] = save_dict['label'].squeeze(0)  # Make it (256, 256)
        
        save_path = os.path.join(output_dir_base, file_name)
        np.save(save_path, save_dict)
        return {"path": save_path}
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found.")
        return
        
    a_dir = os.path.join(DATA_DIR, 'A_norm')
    all_files = sorted([f for f in os.listdir(a_dir) if f.endswith('.npy')])
    
    # Split Dataset (Train=0.7, Val=0.1, Test=0.2)
    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=SEED)
    val_files, test_files = train_test_split(temp_files, test_size=2/3, random_state=SEED)
    
    print(f"Total: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    
    phases = {'train': train_files, 'val': val_files, 'test': test_files}
    distribution = {'train': [], 'val': [], 'test': []}
    
    for phase, file_list in phases.items():
        print(f"\nProcessing {phase} set with {NUM_WORKERS} workers...")
        phase_dir = os.path.join(SAVE_DIR, phase)
        os.makedirs(phase_dir, exist_ok=True)
        
        func = partial(process_file, output_dir_base=phase_dir)
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(executor.map(func, file_list), total=len(file_list)))
            
        for res in results:
            if res is not None:
                distribution[phase].append(res)
                
    with open(JSON_SAVE_PATH, 'w') as f:
        json.dump(distribution, f, indent=4)
        
    print(f"\nJSON distribution saved to {JSON_SAVE_PATH}")
    print("All processing done!")

if __name__ == '__main__':
    main()
