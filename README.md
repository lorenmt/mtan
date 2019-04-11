# Multi-Task Attention Network (MTAN)
This repository contains the source code of Multi-Task Attention Network (MTAN) and baselines from the paper, [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704), introduced by [Shikun Liu](http://shikun.io/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).

## Experiments
### Image-to-Image Predictions (One-to-Many)
Under the folder `im2im_pred`, we have provided our proposed network alongside with all the baselines on `NYUv2` dataset presented in the paper. All the models were written in `pytorch`. So please first make sure you have  `pytorch 1.0` framework or above installed in your machine.

Download our pre-processed `NYUv2` dataset [here](https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=0) which we used in the paper. The original `NYUv2` dataset can be found in [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) with pre-computed ground-truth normals from [here](https://cs.nyu.edu/~deigen/dnl/). 

All the models (files) are built with SegNet and described in the following table:

| File Name        | Type       |  Flags  |  Comments |
| ------------- |-------------| -----|-----|
| `model_segnet_single.py`     | Single  | task, dataroot | standard single task learning |
| `model_segnet_stan.py`     | Single  | task, dataroot | our approach whilst applied on one task |
| `model_segnet_split.py`     | Multi  | weight, dataroot, temp, type | multi-task learning baseline in which the shared network splits at the last layer (also known as hard-parameter sharing) |
| `model_segnet_dense.py`     | Multi  | weight, dataroot, temp | multi-task learning baseline in which each task has its own paramter space (also known as soft-paramter sharing) |
| `model_segnet_cross.py`     | Multi  | weight, dataroot, temp | our implementation of the [Cross Stitch Network](https://arxiv.org/abs/1604.03539) |
| `model_segnet_mtan.py`     | Multi  | weight, dataroot, temp | our approach |

For each flag, it represents

| Flag Name        | Usage  |  Comments |
| ------------- |-------------| -----|
| `task`     | pick one task to train: semantic (semantic segmentation, depth-wise cross-entropy loss), depth (depth estimation, l1 norm loss) or normal (normal prediction, cos-similarity loss)  | only availiable in single-task learning |
| `dataroot`   | directory root for NYUv2 dataset  | just put under the folder `im2im_pred` to avoid any concerns :D |
| `weight`   | weighting options for multi-task learning: equal (direct summation of all task losses), DWA (our proposal), uncert (our implementation of the [Weight Uncertainty Method](https://arxiv.org/abs/1705.07115))  |  only available in multi-task learning |
| `temp`   | hyper-parameter temperature in DWA weighting option  | to determine the softness of task weighting |
| `type`   | different versions of multi-task baseline split: standard, deep, wide  | only available in the baseline split |

To run any model, `cd im2im_pred/` and run `python MODEL_NAME.py --FLAG_NAME 'FLAG_OPTION'`.

### Visual Decathlon Challenge (Many-to-Many)
We have also provided source code for the recently proposed [Visual Decathlon Challenge](http://www.robots.ox.ac.uk/~vgg/decathlon/) for which we build MTAN based on [Wide Residual Network](https://arxiv.org/abs/1605.07146) from the implementation [here](https://github.com/meliketoy/wide-resnet.pytorch).

To run the code, first download the dataset and devkit at the official Visual Decathlon Challenge website [here](http://www.robots.ox.ac.uk/~vgg/decathlon/#download) and put it in the folder `visual_decathlon`. Then, put `decathlon_mean_std.pickle` into the folder of the downloaded dataset `decathlon-1.0-data`.

Finally, run `python model_wrn_mtan.py` for training `python model_wrn_eval.py --dataset 'imagenet' and 'notimagenet'` for evaluation and `python coco_results.py` for COCO format for online evaluation.

### Other Notices
1. The provided code is highly optimised for readibility. If you find any unusual behaviour, please post an issue or directly contact my email below.
2.  Training the provided code will result slightly better performances than the reported numbers in the paper for image-to-image prediction tasks (the rankings stay the same). If you want to compare any models in the paper for image-to-image prediction tasks, please re-run the model directly with your own training strategies (learning rate, optimiser, etc) and keep all training strategies consistent to ensure fairness. To compare results in Visual Decathlon Challenge, you may directly check out the results in the paper. To compare with your own research, please build your multi-task network with the same backbone architecture: SegNet for image-to-image tasks, and Wide Residual Network for the Visual Decathlon Challenge. 
3.  From my personal experience, designing a better architecture is usually more helpful (and easier) than finding a better task weighting in multi-task learning.

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
