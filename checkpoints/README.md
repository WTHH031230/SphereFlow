# Checkpoints

Weights are not stored in git. Download them from [Hugging Face](https://huggingface.co/WTHH031230/SphereFlow) and place them here, or point
`CKPT_DIR` to another directory with the same layout. The scripts use these
file names as defaults; every path can also be overridden with the environment variable named
in the corresponding script.

```text
checkpoints/
├── sphereflow/
│   ├── brats/
│   │   ├── rae.pt      # shared DINOv2-B encoder + 4 ViT-G decoders
│   │   └── flow.pt     # DiT^DH-XXL flow
│   └── mri_ct/
│       ├── rae.pt      # shared DINOv2-B encoder + 2 ViT-G decoders
│       └── flow.pt     # DiT^DH-XL flow
├── baselines/
│   ├── brats/{MMSyn,MMGAN,MMT,ModImp,M2DN,Zhang}.pt
│   └── mri_ct/{MMSyn,MMGAN,MMT,ModImp,M2DN,Zhang}.pt
└── unet3d/brats.pt     # 3D U-Net segmenter
```

Notes

* External weights that are downloaded separately into `models/` (not here):
  `models/discs/dino_vit_small_patch8_224.pth` (DINO discriminator used for decoder training,
  from the RAE release `nyu-visionx/RAE-collections`).
  DINOv2 (`facebook/dinov2-with-registers-base`) is fetched from the Hugging Face Hub.
