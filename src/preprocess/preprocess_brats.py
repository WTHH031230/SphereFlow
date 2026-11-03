import os
import numpy as np
import SimpleITK as sitk
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from monai.transforms import ResizeWithPadOrCropd

# ================= Configuration =================
# Raw BraTS 2023 GLI training data (folder containing BraTS-GLI-xxxxx-xxx case dirs)
DATA_ROOT = os.environ.get("BRATS_RAW_DIR", "./raw_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
# Output root for the processed 2D slices: $DATA_ROOT/brats/{train,val,test}/*.npy
SAVE_ROOT = os.path.join(os.path.abspath(os.environ.get("DATA_ROOT", "./datasets")), "brats")

MODALITIES = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2w.nii.gz', '-t2f.nii.gz']
LABEL_SUFFIX = '-seg.nii.gz'

SLICE_NUM = 80          
TARGET_SHAPE = (256, 256) 
SEED = 42               
NUM_WORKERS = 32         
# ===============================================

def trans_brats_label(x):
    mask_WT = x.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 3] = 1

    mask_TC = x.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 3] = 1

    mask_ET = x.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 3] = 1
    
    mask = np.stack([mask_WT, mask_TC, mask_ET], axis=0)
    return mask

def process_case(case_path, save_dir, phase):
    # Use ResizeWithPadOrCropd to center and pad with constant
    resize_pad = ResizeWithPadOrCropd(
        keys=['image', 'label'], 
        spatial_size=(155, 256, 256), 
        mode='constant', 
        value=0
    )
    
    data = {
        'image': None,
        'label': None
    }
    try:
        case_name = os.path.basename(case_path)
        
        images = []
        for mod_suffix in MODALITIES:
            file_path = os.path.join(case_path, case_name + mod_suffix)
            if not os.path.exists(file_path):
                return 
                
            itk_img = sitk.ReadImage(file_path)
            img_arr = sitk.GetArrayFromImage(itk_img)
            images.append(img_arr)
        
        images = np.stack(images, axis=0) 
        
        # Manual Normalization: clip to [0, 4095] then divide by 4095
        images = np.clip(images, 0, 4095)
        images = images / 4095.0
        
        label_path = os.path.join(case_path, case_name + LABEL_SUFFIX)
        itk_label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(itk_label) 
        label = trans_brats_label(label)

        data['image'] = images.astype(np.float32)
        data['label'] = label.astype(np.int8)

        data = resize_pad(data)

        images = data['image']
        label = data['label']
        _, d, h, w = images.shape
        
        start_slice = (d - SLICE_NUM) // 2
        end_slice = start_slice + SLICE_NUM
        
        if start_slice < 0: start_slice = 0
        if end_slice > d: end_slice = d
        
        images_cropped = images[:, start_slice:end_slice, :, :] 
        label_cropped = label[:, start_slice:end_slice, :, :]      
        
        case_save_dir = os.path.join(save_dir, phase)
        os.makedirs(case_save_dir, exist_ok=True)
        
        actual_slices = images_cropped.shape[1] 
        for i in range(actual_slices):
            slice_img = images_cropped[:, i, :, :] 
            slice_seg = label_cropped[:, i, :, :]  

            save_dict = {
                'image': slice_img.astype(np.float32),
                'label': slice_seg.astype(np.uint8),
                'case_name': case_name,
                'slice_idx': start_slice + i
            }
            
            save_name = f"{case_name}_{start_slice + i:03d}.npy"
            np.save(os.path.join(case_save_dir, save_name), save_dict)
            
    except Exception as e:
        print(f"Error processing {case_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root {DATA_ROOT} does not exist.")
        return

    all_cases = sorted([os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT) 
                        if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    print(f"Total cases found: {len(all_cases)}")
    
    train_cases, temp_cases = train_test_split(all_cases, test_size=0.3, random_state=SEED)
    val_cases, test_cases = train_test_split(temp_cases, test_size=2/3, random_state=SEED)
    
    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")
    
    phases = {'train': train_cases, 'val': val_cases, 'test': test_cases}
    
    for phase, case_list in phases.items():
        print(f"\nProcessing {phase} set with {NUM_WORKERS} workers...")
        
        func = partial(process_case, save_dir=SAVE_ROOT, phase=phase)
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            list(tqdm(executor.map(func, case_list), total=len(case_list)))
            
    print("\nAll processing done!")

if __name__ == "__main__":
    main()