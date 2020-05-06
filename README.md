# Multi-Task Attention Network (MTAN)
This repository contains the source code of Multi-Task Attention Network (MTAN) and baselines from the paper, [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704), introduced by [Shikun Liu](http://shikun.io/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).

## Experiments
### Image-to-Image Predictions (One-to-Many)
Under the folder `im2im_pred`, we have provided our proposed network alongside with all the baselines on `NYUv2` dataset presented in the paper. All the models were written in `pytorch`. So please first make sure you have  `pytorch 1.0` framework or above installed in your machine.

Download our pre-processed `NYUv2` dataset [here](https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=0) which we evaluated in the paper. We use the pre-computed ground-truth normals from [here](https://cs.nyu.edu/~deigen/dnl/). The raw 13-class NYUv2 dataset can be directly downloaded in [this repo](https://github.com/ankurhanda/nyuv2-meta-data) with segmentation labels defined in [this repo](https://github.com/ankurhanda/SceneNetv1.0/).

*I am sorry that I am not able to provide the raw pre-processing code due to an unexpected computer crash.*

**Update - Jun 2019**: I have now released the pre-processing `CityScapes` dataset with 2, 7, and 19-class semantic labels (see the paper for more details) and (inverse) depth labels. Download [256x512, 2.42GB] version [here](https://www.dropbox.com/s/q2333k4eyrnezbh/cityscapes.zip?dl=0) and [128x256, 651MB] version [here](https://www.dropbox.com/s/lg2ktu7o8hzwf99/cityscapes2.zip?dl=0).

**Update - Oct 2019**: For pytorch 1.2 users: The mIoU evaluation method has now been updated to avoid "zeros issue" from computing binary masks. Also, to correctly run the code, please move the `scheduler.step()` after calling the `optimizer.step()`, e.g. one line before the last performance printing step to fit the updated pytorch requirements. See more in the official pytorch documentation [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). 

**Update - May 2020**: We now have provided our official MTAN-DeepLabv3 (or ResNet-like architecture) design to support more complicated and modern multi-task network backbone. Please check out `im2im_pred/model_resnet_mtan` for more details. One should easily replace this model with any original training methods defined in `im2im_pred`.

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

To run any model, `cd im2im_pred/` and run `python MODEL_NAME.py --FLAG_NAME 'FLAG_OPTION'`.

### Benchmarking Multi-task Learning
Benchmarking multi-task learning is always a tricky question, since the performance and evaluation method for each task is different. In my original paper, I simply averaged the performance for each task from the last 10 epochs, assuming we do not have access to the validation data. 

For a more standardized and fair comparison, I would suggests researchers adopt the evaluation method defined in Section 5, Equation 4 of [this paper](https://arxiv.org/pdf/1904.08918.pdf), which computes the *average relative task improvements* over single task learning.

![Z(i,j)=X(i,k) * Y(k, j); k=1 to n](http://www.sciweavers.org/tex2img.php?eq=Z_i_j%3D%5Csum_%7Bi%3D1%7D%5E%7B10%7D%20X_i_k%20%2A%20Y_k_j&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

for which `l_i=1` if a lower value means a better performance for task `i`, and `l_i=0` otherwise; `T` is number of tasks; `M_m` represents the performance of evaluated multi-task learning method and `M_b` represents the baseline method. We normally choose the baseline method to be the single task learning using the same backbone architecture.

### Visual Decathlon Challenge (Many-to-Many)
We have also provided source code for the recently proposed [Visual Decathlon Challenge](http://www.robots.ox.ac.uk/~vgg/decathlon/) for which we build MTAN based on [Wide Residual Network](https://arxiv.org/abs/1605.07146) from the implementation [here](https://github.com/meliketoy/wide-resnet.pytorch).

To run the code, first download the dataset and devkit at the official Visual Decathlon Challenge website [here](http://www.robots.ox.ac.uk/~vgg/decathlon/#download) and put it in the folder `visual_decathlon`. Then, put `decathlon_mean_std.pickle` into the folder of the downloaded dataset `decathlon-1.0-data`.

Finally, run `python model_wrn_mtan.py` for training `python model_wrn_eval.py --dataset 'imagenet' and 'notimagenet'` for evaluation and `python coco_results.py` for COCO format for online evaluation.

### Other Notices
1. The provided code is highly optimised for readability. If you find any unusual behaviour, please post an issue or directly contact my email below.
2.  Training the provided code will result slightly different performances (depending on the type of task) than the reported numbers in the paper for image-to-image prediction tasks. But, the rankings stay the same. If you want to compare any models in the paper for image-to-image prediction tasks, please re-run the model directly with your own training strategies (learning rate, optimiser, etc) and keep all training strategies consistent to ensure fairness. To compare results in Visual Decathlon Challenge, you may directly check out the results in the paper. To compare with your own research, please build your multi-task network with the same backbone architecture: SegNet for image-to-image tasks, and Wide Residual Network for the Visual Decathlon Challenge. 
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
We would like to thank Simon Vandenhende for his help on MTAN-DeepLabv3 design. 

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.
