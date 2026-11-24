# TLP Balancer

Code for **"TLP Balancer: Predictive Thread Allocation for Multitenant Inference in Embedded GPUs"**, IEEE Embedded Systems Letters, vol. 17, no. 3, June 2025. DOI: [10.1109/LES.2024.3497587](https://doi.org/10.1109/LES.2024.3497587)

TLP Balancer predicts the number of warps to give each model in a merged and fused kernel. The prediction comes from a model of per-thread operation counts (Eq. (1) and (2) in the paper), instead of splitting warps evenly between the models.

## Requirements

- NVIDIA Jetson Orin (8 SMs, 48 warps per SM)
- CUDA (tested with 11.4)
- Python 3 with NumPy

## Repository layout

```
├── exp.sh                               # Build + run all experiments
├── experiments_seq.py                   # Sequential execution (baseline)
├── experiments_merge_and_fusion_seq.py  # Merged & fused execution (mode 0 / 1 / 3)
├── parsing.py                           # Average queuing delay / inference time of a result file
├── random_model.txt                     # Random order of 2000 requests (ResNet50 / ViT-Base / VGG16)
├── results/                             # Results used in the paper
└── src/
    ├── ResNet50_ViT.cu                  # Host code, ResNet50 + ViT-Base fused inference, VGG16 inference
    ├── warp_allocation.cu               # Warp allocation policies (TLP Balancer, prior work, Oracle)
    ├── layer_fused_matmul.cu            # Fused matmul kernel for two models, ViT attention kernel
    └── vgg_kernels.cu                   # VGG16 kernels
```

## VGG16 data

VGG16 runs on real weights. These are not in the repository because they are 1.7GB. Put them in `data_vgg16/` at the repository root:

```
data_vgg16/
├── input_cat.txt          # input image (224x224x3)
├── bias1.txt  ... bias16.txt
├── conv1.txt  ... conv13.txt
└── fc14.txt   fc15.txt   fc16.txt
```

Each file holds whitespace-separated floating-point values. ResNet50 and ViT-Base use randomly initialized tensors, because the experiments measure only execution time.

## Build and run

Run everything from the repository root, because the scripts use relative paths.

```bash
# Build the shared library used by the Python scripts
nvcc -Xcompiler -fPIC -shared -o tlp_balancer.so src/ResNet50_ViT.cu

# Run all experiments (1000 requests each)
./exp.sh
```

Single runs:

```bash
./experiments_seq.py <num_requests> <avg_inference_time>
./experiments_merge_and_fusion_seq.py <mode> <num_requests> <avg_inference_time>
```

| Policy | Command | Result files |
|---|---|---|
| Sequential (baseline) | `./experiments_seq.py 1000 1.043856534` | `results/sequential*.txt` |
| Merge and Fusion (prior work, even warp split) | `./experiments_merge_and_fusion_seq.py 0 1000 0.855183874` | `results/merge_fusion*.txt` |
| **TLP Balancer** | `./experiments_merge_and_fusion_seq.py 1 1000 0.649835207` | `results/tlp_balancer*.txt` |
| Oracle | `./experiments_merge_and_fusion_seq.py 3 1000 0.642967405` | `results/oracle*.txt` |

Sequential has its own script and takes no mode argument.

`<avg_inference_time>` sets the Poisson arrival rate (arrival interval = avg_inference_time × 100/95). The values above are the ones used in the paper and in `exp.sh`.

## Results

Each policy produces two files in `results/`:

- `<policy>.txt`: one line per request (`model  queuing_delay  inference_time`), followed by a summary.
- `<policy>_warp_allocation.txt`: one line per fused layer (`layer: #ResNet #ViT ResNet_warps ViT_warps kernel_time_ms`).

`oracle_old_table*.txt` comes from an earlier version of the Oracle warp table. `oracle*.txt` matches the table in `src/warp_allocation.cu`.

Result files are opened in append mode. Move old results away before running again.

```bash
./parsing.py results/tlp_balancer.txt   # average queuing delay / inference time
```

## Citation

```bibtex
@article{gil2025tlpbalancer,
  title   = {TLP Balancer: Predictive Thread Allocation for Multitenant Inference in Embedded GPUs},
  author  = {Gil, Minseong and Jeon, Jaebeom and Kim, Junsu and Choi, Sangun and Koo, Gunjae and Yoon, Myung Kuk and Oh, Yunho},
  journal = {IEEE Embedded Systems Letters},
  volume  = {17},
  number  = {3},
  pages   = {180--183},
  year    = {2025},
  doi     = {10.1109/LES.2024.3497587}
}
```
