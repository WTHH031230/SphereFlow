# SphereFlow: Missing Modality Imputation via Geometric Transport on Hypersphere
by Huihan Wang\*, Juncheng Wang\*, Yuxiang Feng, Wenlong Hou, Guoqi Yu, Chao Xu, Yang Liu, Baigui Sun, and Shujun Wang†.

### Introduction

This repository is for our NeurIPS 2026 paper: "SphereFlow: Missing Modality Imputation via Geometric Transport on Hypersphere". This work imputes missing modalities by transporting frozen DINOv2 latents on the hypersphere, evaluated on BraTS 2023 (MRI) and SynthRAD2023 (MRI ↔ CT).

### Usage

1. Clone this repo and create the environment;
```
conda env create -f environment.yml
conda activate sphereflow
```
2. Prepare the data. Download the preprocessed data from [Hugging Face](https://huggingface.co/WTHH031230/SphereFlow);
```
hf download WTHH031230/SphereFlow --include "data/*" --local-dir .
for f in data/*.tar; do tar -xf "$f" -C /path/to/processed; done
```
or build it from the raw data, where `MRICT_RAW_DIR` holds the SynthRAD2023 (brain) 2D slices in `A_norm/` (MRI), `B_norm/` (CT) and `C_norm/` (mask), in [-1, 1];
```
BRATS_RAW_DIR=/path/to/brats DATA_ROOT=/path/to/processed bash scripts/preprocess/brats.sh
MRICT_RAW_DIR=/path/to/synthrad DATA_ROOT=/path/to/processed bash scripts/preprocess/mri_ct.sh
```
Then write the splits;
```
DATA_ROOT=/path/to/processed bash scripts/preprocess/make_splits.sh --check
```
3. Download the pretrained weights into `./checkpoints/` (layout in [checkpoints/README.md](checkpoints/README.md)). For training, also download the DINO discriminator;
```
hf download WTHH031230/SphereFlow --include "checkpoints/*" --local-dir .
hf download nyu-visionx/RAE-collections discs/dino_vit_small_patch8_224.pth --local-dir models
```
4. Training on BraTS;
```
bash scripts/train/brats/step1_single_decoder.sh
bash scripts/train/brats/step2_interp_decoder.sh
bash scripts/train/brats/step3_shared_encoder.sh
bash scripts/train/brats/finetune_decoder.sh  # optional
bash scripts/train/brats/step4_flow.sh
```
Training on MRI ↔ CT;
```
bash scripts/train/mri_ct/step1_single_decoder.sh
bash scripts/train/mri_ct/step2_interp_decoder.sh
bash scripts/train/mri_ct/step3_shared_encoder.sh
bash scripts/train/mri_ct/step4_flow.sh
```
5. Testing;
```
bash scripts/eval/brats.sh
bash scripts/eval/mri_ct.sh
```

### Baselines

We also release the code of the baselines (MMSyn, MMGAN, MMT, ModImp, M2DN, Zhang) and of the downstream 3D U-Net segmentation. For CodeBrain we followed the authors' [open-source code](https://github.com/ycwu1997/CodeBrain), adapted to our data and evaluation.

1. Create the environment;
```
conda env create -f baseline/environment.yml
conda activate sphereflow_baseline
```
2. Training and testing (`<method>`: MMSyn, MMGAN, MMT, ModImp, M2DN, Zhang);
```
bash baseline/scripts/<method>/train_brats.sh
bash baseline/scripts/<method>/eval_brats.sh
bash baseline/scripts/<method>/train_mri_ct.sh
bash baseline/scripts/<method>/eval_mri_ct.sh
```
3. Downstream segmentation;
```
bash downstream/scripts/train.sh
bash downstream/scripts/eval.sh
```

### Citation

If SphereFlow is useful for your research, please consider citing:

    @inproceedings{wang2026sphereflow,
        title     = {SphereFlow: Missing Modality Imputation via Geometric Transport on Hypersphere},
        author    = {Wang, Huihan and Wang, Juncheng and Feng, Yuxiang and Hou, Wenlong and Yu, Guoqi and Xu, Chao and Liu, Yang and Sun, Baigui and Wang, Shujun},
        booktitle = {Advances in Neural Information Processing Systems},
        year      = {2026}
    }

### Acknowledgments

Our code is adapted from [RAE](https://github.com/bytetriper/RAE). Thanks to the authors for their valuable work.

### Questions

If any questions, feel free to contact 'wang2003huihan@gmail.com'.
