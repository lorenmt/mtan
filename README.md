# Multi-Task Attention Network (MTAN)
This repository contains the source code of Multi-Task Attention Network (MTAN) and baselines from the paper, [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704), introduced by [Shikun Liu](http://shikun.io/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).

## Experiments
### Image-to-Image Predictions
Under folder `im2im_pred`, we have provided our proposed model alongside with all the baselines presented in the paper. All the models were written in `pytorch`, so please first make sure you have  `pytorch 1.0` framework or above installed in your machine.

Download our post-processed `NYUv2` dataset [here](https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=0) which we used in the paper. The original `NYUv2` dataset can be found in [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) with pre-computed ground-truth normals are from [here](https://cs.nyu.edu/~deigen/dnl/).

All the models (files) are described in the following table:

| File Names        | Type       |  Flags  |  Comments |
| ------------- |-------------| -----|-----|
| `model_segnet_single.py`     | Single  | task, dataroot | standard single task learning |
| `model_segnet_stan.py`     | Single  | task, dataroot | attention network applied on one task |
| `model_segnet_split.py`     | Multi  | weight, dataroot, temp, type | standard multi-task learning baseline which splits at the last layer (also known as hard-parameter sharing) |
| `model_segnet_cross.py`     | Multi  | weight, dataroot, temp | our implementation on the [Cross Stitch Network](https://arxiv.org/abs/1604.03539) |
| `model_segnet_mtan.py`     | Multi  | weight, dataroot, temp | our approach |

For each flag, it represents

| Flag Names        | Usage  |  Comments |
| ------------- |-------------| -----|
| `task`     | pick task to train: semantic (semantic segmentation, depth-wise cross-entropy loss), depth (depth estimation, l1 norm loss) or normal (normal prediction, cos-similarity loss)  | only availiable in single-task learning |
| `dataroot`   | directoy root for NYUv2 dataset  | just put under the folder `im2im_pred` to avoid any concerns  |
| `weight`   | weighting options for multi-task learning: equal, DWA (our proposal), uncert (our implementation on [Weight Uncertainty Method](https://arxiv.org/abs/1705.07115))  |  only available in multi-task learning |
| `temp`   | hyper-parameter temperature in DWA weighting option  | to determine the softness of task weighting |
| `type`   | different versions of multi-task baseline split: standard, deep, wide  | only available in the split |
