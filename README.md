# Multi-Task Attention Network (MTAN)
This repository contains the source code of Multi-Task Attention Network (MTAN) and baselines from the paper, [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704), introduced by [Shikun Liu](http://shikun.io/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).

## Experiments
### Image-to-Image Predictions (One-to-Many)
Under folder `im2im_pred`, we have provided our proposed network alongside with all the baselines presented in the paper. All the models were written in `pytorch`. So please first make sure you have  `pytorch 1.0` framework or above installed in your machine.

Download our pre-processed `NYUv2` dataset [here](https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=0) which we used in the paper. The original `NYUv2` dataset can be found in [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) with pre-computed ground-truth normals are from [here](https://cs.nyu.edu/~deigen/dnl/).

All the models (files) are described in the following table:

| File Names        | Type       |  Flags  |  Comments |
| ------------- |-------------| -----|-----|
| `model_segnet_single.py`     | Single  | task, dataroot | standard single task learning |
| `model_segnet_stan.py`     | Single  | task, dataroot | attention network applied on one task |
| `model_segnet_split.py`     | Multi  | weight, dataroot, temp, type | standard multi-task learning baseline in which the shared network splits at the last layer (also known as hard-parameter sharing) |
| `model_segnet_cross.py`     | Multi  | weight, dataroot, temp | our implementation of the [Cross Stitch Network](https://arxiv.org/abs/1604.03539) |
| `model_segnet_dense.py`     | Multi  | weight, dataroot, temp | standard multi-task learning baseline in which each task has its own paramter space (also known as soft-paramter sharing)  |
| `model_segnet_mtan.py`     | Multi  | weight, dataroot, temp | our approach |

For each flag, it represents

| Flag Names        | Usage  |  Comments |
| ------------- |-------------| -----|
| `task`     | pick task to train: semantic (semantic segmentation, depth-wise cross-entropy loss), depth (depth estimation, l1 norm loss) or normal (normal prediction, cos-similarity loss)  | only availiable in single-task learning |
| `dataroot`   | directory root for NYUv2 dataset  | just put under the folder `im2im_pred` to avoid any concerns  |
| `weight`   | weighting options for multi-task learning: equal (summation of all task losses), DWA (our proposal), uncert (our implementation of the [Weight Uncertainty Method](https://arxiv.org/abs/1705.07115))  |  only available in multi-task learning |
| `temp`   | hyper-parameter temperature in DWA weighting option  | to determine the softness of task weighting |
| `type`   | different versions of multi-task baseline split: standard, deep, wide  | only available in the baseline split |

To run any model, `cd im2im_pred/` and run `python MODEL_NAME.py --FLAG_NAME 'FLAG_OPTION'`.

### Visual Decathlon Challenge (Many-to-Many)
We have also provided source code for [Visual Decathlon Challenge](http://www.robots.ox.ac.uk/~vgg/decathlon/) which we build MTAN based on [Wide Residual Network](https://arxiv.org/abs/1605.07146) with the implementation [here](https://github.com/meliketoy/wide-resnet.pytorch).

To run the code, first download the dataset and devkit at the official Visual Decathlon Challenge website [here](http://www.robots.ox.ac.uk/~vgg/decathlon/#download). Then, put `decathlon_mean_std.pickle` into the folder of the downloaded dataset `decathlon-1.0-data`.

Finally, run `python model_wrn_mtan.py` for training `python model_wrn_eval --dataset 'imagenet' and 'notimagenet'` for evaluation and `python coco_results.py` for COCO format.

### Other Notices
1. The provided code is highly optimised. If you find any unusual behaviour, please post an issue or directly contact my email below.
2.  Training the provided code will present a slightly better performance than the reported result in the paper. If you want to compare any model in the paper, please run the model directly with your own training strategies (learning rate, optimiser, etc), whilst keeping all models with the consistent training strategies to ensure fairness.
3.  From my personal experience, building a better architecture is always more helpful than finding a better task weighting in multi-task learning.

## Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```
@article{liu2018mtan,
  title={End-to-end multi-task learning with attention},
  author={Liu, Shikun and Johns, Edward and Davison, Andrew J},
  journal={arXiv preprint arXiv:1803.10704},
  year={2018}
}
```

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.
