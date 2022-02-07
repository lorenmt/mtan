# MTAN - Multi-Task Attention Network
This repository contains the source code of Multi-Task Attention Network (MTAN) and baselines from the paper, [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704), introduced by [Shikun Liu](https://shikun.io/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).

See more results in our project page [here](https://shikun.io/projects/multi-task-attention-network).

**Final Update** - This repository will not be further updated.  Check out our latest work: [Auto-Lambda](https://github.com/lorenmt/auto-lambda) for more multi-task optimisation methods.

## Experiments
### Image-to-Image Predictions (One-to-Many)
Under the folder `im2im_pred`, we have provided our proposed network along with all the baselines on `NYUv2` dataset presented in the paper. All models were written in `PyTorch`, and we have updated the implementation to PyTorch version 1.5 in the latest commit.

Download our pre-processed `NYUv2` dataset [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) which we evaluated in the paper. We use the pre-computed ground-truth normals from [here](https://cs.nyu.edu/~deigen/dnl/). The raw 13-class NYUv2 dataset can be directly downloaded in [this repo](https://github.com/ankurhanda/nyuv2-meta-data) with segmentation labels defined in [this repo](https://github.com/ankurhanda/SceneNetv1.0/).

*I am sorry that I am not able to provide the raw pre-processing code due to an unexpected computer crash.*

**Update - Jun 2019**: I have now released the pre-processing `CityScapes` dataset with 2, 7, and 19-class semantic labels (see the paper for more details) and (inverse) depth labels. Download [256x512, 2.42GB] version [here](https://www.dropbox.com/sh/vj349qgg57nthi9/AACdZmIuK-Qb_gP6w1HrA43ta?dl=0) and [128x256, 651MB] version [here](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0).

**Update - Oct 2019**: For pytorch 1.2 users: The mIoU evaluation method has now been updated to avoid "zeros issue" from computing binary masks. ~~Also, to correctly run the code, please move the `scheduler.step()` after calling the `optimizer.step()`, e.g. one line before the last performance printing step to fit the updated pytorch requirements. See more in the official pytorch documentation [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).~~ [We have fixed this in the latest commit.]

**Update - May 2020**: We now have provided our official MTAN-DeepLabv3 (or ResNet-like architecture) design to support more complicated and modern multi-task network backbone. Please check out `im2im_pred/model_resnet_mtan` for more details. One should easily replace this model with any training template defined in `im2im_pred`.

**Update - July 2020**: We have further improved the readability and updated all implementations in `im2im_pred` to comply the current latest version PyTorch 1.5. We fixed a bug to exclude non-defined pixel predictions for a more accurate mean IoU computation in semantic segmentation tasks. We also provided an additional option for users applying data augmentation in NYUv2 to avoid over-fitting and achieve better performances.

**Update - Nov 2020 [IMPORTANT!]**: We have updated mIoU and Pixel Accuracy formulas to be consistent with the standard benchmark from the [official COCO segmentation scripts](https://github.com/pytorch/vision/tree/master/references/segmentation). The mIoU for all methods are now expected to improve approximately 8% of performance. The new formulas compute mIoU and Pixel Accuracy based on the accumulated pixel predictions across all images, while the original formulas were based on average pixel predictions in each image across all images.

All models (files) built with SegNet (proposed in the original paper), are described in the following table:

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
| `task`     | pick one task to train: semantic (semantic segmentation, depth-wise cross-entropy loss), depth (depth estimation, l1 norm loss) or normal (normal prediction, cos-similarity loss)  | only available in single-task learning |
| `dataroot`   | directory root for NYUv2 dataset  | just put under the folder `im2im_pred` to avoid any concerns :D |
| `weight`   | weighting options for multi-task learning: equal (direct summation of all task losses), DWA (our proposal), uncert (our implementation of the [Weight Uncertainty Method](https://arxiv.org/abs/1705.07115))  |  only available in multi-task learning |
| `temp`   | hyper-parameter temperature in DWA weighting option  | to determine the softness of task weighting |
| `type`   | different versions of multi-task baseline split: standard, deep, wide  | only available in the baseline split |
| `apply_augmentation`   | toggle on to apply data augmentation in NYUv2 to avoid over-fitting  | available in all training models |

To run any model, `cd im2im_pred/` and simply run `python MODEL_NAME.py --FLAG_NAME 'FLAG_OPTION'` (default option is training without augmentation). Toggle on `apply_augmentation` flag to train with data augmentation: `python MODEL_NAME.py --FLAG_NAME 'FLAG_OPTION' --apply_augmentation`.

Please note that, we did not apply any data augmentation in the original paper.

### Benchmarking Multi-task Learning
Benchmarking multi-task learning is always a tricky question, since the performance and evaluation method for each task is different. In the original paper, I simply averaged the performance for each task from the last 10 epochs, assuming we do not have access to the validation data. 

For a more standardized and fair comparison, I would suggest researchers adopt the evaluation method defined in Section 5, Equation 4 of [this paper](https://arxiv.org/pdf/1904.08918.pdf), which computes the *average relative task improvements* over single task learning.

NYUv2 can be easily over-fitted due to its small sample size. In July's update, we have provided an option to apply data augmentation to alleviate the over-fitting issue (thanks to Jialong's help). We highly recommend to benchmark NYUv2 dataset with this data augmentation, to be consistent with other SOTA multi-task learning methods using the same data augmentation technique, such as [PAD-Net](https://arxiv.org/abs/1805.04409) and [MTI-Net](https://arxiv.org/abs/2001.06902).

### Visual Decathlon Challenge (Many-to-Many)
We also provided source code for [Visual Decathlon Challenge](http://www.robots.ox.ac.uk/~vgg/decathlon/) for which we build MTAN based on [Wide Residual Network](https://arxiv.org/abs/1605.07146) from the implementation [here](https://github.com/meliketoy/wide-resnet.pytorch).

To run the code, please follow the steps below.
1. Download the dataset and devkit at the official Visual Decathlon Challenge website [here](http://www.robots.ox.ac.uk/~vgg/decathlon/#download). Move the dataset folder `decathlon-1.0-data` under the folder `visual_decathlon`. Then, move `decathlon_mean_std.pickle` into the folder of the dataset folder `decathlon-1.0-data`.

2. Create a directory under `test` folder for each dataset, and move all test files into that created folder. (That is to comply the PyTorch dataloader format.)

3. Install `setup.py` in decathlon devkit under `code/coco/PythonAPI` folder. And then move `pycocotools` and `annotations` from devkit into `visual_decathlon` folder.

4. `cd visual_decathlon` and run `python model_wrn_mtan.py --gpu [GPU_ID] --mode [eval, or all]` for training. `eval` represents evaluating on validation dataset (normally for debugging or hyper-parameter tuning), and `all` represents training on all datasets (normally for final evaluating, or benchmarking). 

5. Run `python model_wrn_eval.py --dataset 'imagenet' and 'notimagenet'` (sequentially) for evaluating on Imagenet and other datasets. And finally, run `python coco_results.py` for converting into COCO format for online evaluation.

### Other Notices
1. The provided code is highly optimised for readability. If you find any unusual behaviour, please post an issue or directly contact my email below.
2.  Training the provided code will result different performances (depending on the type of task) than the reported numbers in the paper for image-to-image prediction tasks. But, the rankings stay the same. If you want to compare any models in the paper for image-to-image prediction tasks, please re-run the model on your own with your preferred training strategies (learning rate, optimiser, etc) and keep all training strategies consistent to ensure fairness. To compare results in Visual Decathlon Challenge, you may directly borrow the results presented in the paper. To fairly compare in your research, please build your multi-task network with the same backbone architecture.
3.  From my personal experience, designing a better architecture is usually more helpful (and easier) than finding a better task weighting in multi-task learning.

## Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```
@inproceedings{liu2019end,
  title={End-to-End Multi-task Learning with Attention},
  author={Liu, Shikun and Johns, Edward and Davison, Andrew J},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1871--1880},
  year={2019}
}
```

## Acknowledgement
We would like to thank Simon Vandenhende for his help on MTAN-DeepLabv3 design; Jialong Wu on his generous contribution to benchmarking MTAN-DeepLabv3, and implementation on data augmentation for NYUv2 dataset. 

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.
