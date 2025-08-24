<!-- <div align="center">
<p style="font-size: 20pt;">
Entropy-Driven Adaptive Clustering Attention for Continuous Hyperspectral Image Super-Resolution
</p> -->

# Entropy-Driven Adaptive Clustering Attention for Continuous Hyperspectral Image Super-Resolution

This framework is tested on Python 3.9.x / 3.12 and Pytorch 2.3.0 with CUDA 11.8.

## Installation Requirements

You can simply install all needed Python packages by running `pip install -r requirement.txt`.

## Prepare Your Model

You can put in your model and register with the model name that you prefer using `register_model`, then import your model in `model/__init__.py`. 

> We are trying using `hydra` dynamic importing. So, this api may be changed.

Specify your model's `train_step`, `val_step` and `patch_merge_step` (optional) if you are working on *Pansharpening* and *HMIF* tasks, or, if you are working *VIS-IR* and *medical image fusion* tasks, you should specify the `train_fusion_step` and `val_fusion_step` functions.

 You can find detailed example usage in the `model/`.

## Starting Training

Here are two scripts for training your model. 

1. scripts implemented by torch-run.
2. scripts implemented by Huggingface Accelerate. 

### Fusion Tasks

For fusion tasks, you should switch to `scripts/accelerate_run.sh` and modify it to suit your way.

### Inference

After you trained your model, you need to infer model on test set.

- For fusion tasks, you need to run `scripts/torch_sharpening_run.sh` (after you modify it to suit your model).


## Test Metrics

We provide the Matlab package to test the metrics on those four tasks. Please check them in `Pansharpening_Hyper_SR_Matlab_Test_Package` and `VIS_IR_Matlab_Test_Package`.

### Basic Usage

- For sharpening tasks, you simply test the metrics in Matlab

``` matlab
cd Pansharpening_Hyper_SR_Matlab_Test_Package

%% when testing the reduced-resolution metrics on HMIF tasks
% Args:
% path: the saved fused image `.mat` file, find it in `visualized_img/`
% ratio: upscale ratio, e.g., 4
% full_res: we keep it to 1, not changed
% const: max value of the dataset
analysis_ref_batched_images(path, ratio, full_res, const)

- For fusion tasks, you can run `runDir` in `VIS_IR_Matlab_Test_Package/` in Matlab.

``` matlab
cd VIS_IR_Matlab_Test_Package;

% Args:
% vi_dir: visible images dir
% ir_dir: infrared images dir
% fusion_dir: fused images from your model
% method_name: the name of the method
% test_mode_easy (0 or 1): some metrics may use much time to test;
%                          1 for east, 0 for overall metrics to test.
% test_ext: vi and ir image extension, default to be png.
% fused_ext: fused image extension, default to be png.
runDir(vi_dir, ir_dir, fusion_dir, method_name, test_mode_easy, varargin)

% and also a multiprocessing test matlab script is involved.
mp_run(fusion_dir, dataset_name, method_name, rgb_test, test_mode_easy, varargin)
```
If you find it is troublesome to open an matlab to test, we prepare a python script to help you test with matlab process in background.

```python
python py_run_matlab_VIS_IR_test.py -f <your/fused/path> -m <your_method_name> -d <dataset_name>
```

# About Dataset

The used datasets includes:
- HMIF: indoor [CAVE and Harvard datasets](https://github.com/shangqideng/PSRT?tab=readme-ov-file#data), remote sensing [Paiva, Houston, Washington datasets](https://github.com/liangjiandeng/HyperPanCollection);



