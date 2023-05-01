# Contour tracking
### [Project Page](https://junbongjang.github.io/projects/contour-tracking/index.html) | [Paper](https://arxiv.org/abs/2303.08364) | [Data](https://zenodo.org/record/7754589)

**[Unsupervised Contour Tracking of Live Cells by Mechanical and Cycle Consistency Losses](https://junbongjang.github.io/projects/contour-tracking/index.html)**
 <br />
 [Junbong Jang](https://junbongjang.github.io/)<sup>1</sup>,
 [Kwonmoo Lee](https://research.childrenshospital.org/kwonmoo-lee)\*<sup>2,3</sup>,
 [Tae-Kyun Kim](https://sites.google.com/view/tkkim/home)\*<sup>1,4</sup>
 <br /><sup>1</sup>KAIST, <sup>2</sup>Boston Children's Hospital, <sup>3</sup>Harvard Medical School, <sup>4</sup>Imperial College London
 <br />*denotes corresponding authors
 <br />Accepted to CVPR 2023

Tensorflow implementation of contour tracking of live cell and jellyfish videos.<br><br>

<div align="center">
    <img width="800" src='architecture.png' alt="architecture"/>
</div>



## Setup
We ran this code in the following setting.
* Python 3.6.9
* Tensorflow 2.4.3
* CUDA 11.0
* Ubuntu 18.04

## Installation
Build and run Docker using our Dockerfile.
* For instance:
    >sudo docker build -t tf24_contour_tracking .

    >sudo docker run -v /home/junbong/contour-tracking:/home/docker/contour-tracking -v /data/junbong/optical_flow/generated_3_5_2023:/data/junbong/optical_flow/generated_3_5_2023 -v /data/junbong/optical_flow/assets:/data/junbong/optical_flow/assets --gpus '"device=0"' -it tf24_contour_tracking

Then, install Python dependencies in the requirements.txt
>./install_setup.sh

## Data Pre-processing
We recommend doing this in a local PC in Windows 10/11

To create GT tracking points using our GUI,
>tracking_points_labeling_GUI.py

To create pseudo labels from the tracking results from MATLAB morphodynamics program, run
>MATLAB_tracking_points.py

To sample normal points,
>process_tracking_points.py --> sample_points() --> iterative normal line sampling

To get all points along the segmentation mask
>process_tracking_points.py --> sample_points() --> simply sample all points along the contour with order

To get correspondence tracking points to segmentation mask
Run this after MATLAB_tracking_points.py and process_tracking_points.py --> sample_points()
>process_tracking_points.py --> convert_GT_tracking_points_to_contour_indices()

## Data Conversion to TFRecord

To create tfrecords with images + seg_points + tracking_points
For training set
>python -m src.data_conversion_scripts.convert_custom_to_tfrecords --data_dir=src/assets/pc_5small_sparse_matlab_seg_all_points/ --shard=0 --num_shards=1 --img_format=png

For test set
>python -m src.data_conversion_scripts.convert_custom_to_tfrecords --data_dir=src/assets/pc_5small_dense_matlab_seg_all_points/ --shard=0 --num_shards=1 --img_format=png
>python -m src.data_conversion_scripts.convert_custom_to_tfrecords --data_dir=src/assets/HACKS_live/ --shard=0 --num_shards=1 --img_format=png

>python -m src.data_conversion_scripts.convert_custom_to_tfrecords --data_dir=src/assets/Jellyfish/ --shard=0 --num_shards=1 --img_format=png

To preprocess the segmentation masks
>python misc/get_edge_from_segmentation.py

## Training

First, edit src/uflow_flags.py to change hyperparameters such as learning rate and number of training steps.

###### MARS-Net dataset (all 200 frames)
>python -m src.contour_flow_main --train_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/training/ --valid_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test4/ --height=256 --width=256 --generated_dir=uflow/generated/pc_5small_dense_matlab_seg_all_points_match_lam3_batch8/ --use_tracking_points --use_seg_points --batch_size=8

>python -m src.contour_flow_main --train_on=custom:/data/junbong/optical_flow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/training/ --valid_on=custom:/data/junbong/optical_flow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test4/ --height=256 --width=256 --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_cycle_normal_batch8/ --use_tracking_points --use_seg_points --batch_size=8


###### HACKS dataset (all 200 frames)
>python -m src.contour_flow_main --train_on=custom:uflow/assets/HACKS_live/tfrecord/training/ --valid_on=custom:uflow/assets/HACKS_live/tfrecord/valid/ --height=512 --width=512 --generated_dir=uflow/generated/HACKS_live_cycle_normal_batch8/ --use_tracking_points --use_seg_points --batch_size=8

###### For comparison against UFlow, change the code in contour_flow_net.py before running this
>python -m src.contour_flow_main --train_on=custom:uflow/assets/pc_5small_sparse_matlab_seg_all_points/tfrecord/training/ --height=256 --width=256 --generated_dir=uflow/generated/pc_5small_sparse_matlab_seg_all_points_uflow_batch8/ --use_segmentations --use_tracking_points --batch_size=8

>python -m src.contour_flow_main --train_on=custom:uflow/assets/HACKS_live/tfrecord/training/ --height=512 --width=512 --generated_dir=uflow/generated/HACKS_live_uflow_batch8/ --use_segmentations --use_tracking_points --batch_size=8

###### For comparison against PoST, change the code in tracking_model.py before running this
>python -m src.contour_flow_main --train_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/training/ --valid_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test4/ --height=256 --width=256 --generated_dir=uflow/generated/pc_5small_dense_matlab_seg_all_points_post_batch8/ --use_tracking_points --use_seg_points --batch_size=8

>python -m src.contour_flow_main --train_on=custom:uflow/assets/HACKS_live/tfrecord/training/ --valid_on=custom:uflow/assets/HACKS_live/tfrecord/test9/ --height=512 --width=512 --generated_dir=uflow/generated/HACKS_live_post_batch8/ --use_tracking_points --use_seg_points --batch_size=8

>python -m src.contour_flow_main --train_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/training/ --valid_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test4/ --height=256 --width=256 --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_post_re_batch8/ --use_tracking_points --use_seg_points --batch_size=8

>python -m src.contour_flow_main --train_on=custom:/data/junbong/optical_flow/assets/HACKS_live/tfrecord/training/ --valid_on=custom:uflow/assets/HACKS_live/tfrecord/test9/ --height=512 --width=512 --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_post_re_batch4/ --use_tracking_points --use_seg_points --batch_size=4

###### For Jellyfish
>python -m src.contour_flow_main --train_on=custom:uflow/assets/Jellyfish/tfrecord/training/ --valid_on=custom:uflow/assets/Jellyfish/tfrecord/test/ --height=256 --width=256 --generated_dir=uflow/generated/Jellyfish_live_cycle_normal_batch8/ --use_tracking_points --use_seg_points --batch_size=8

>python -m src.contour_flow_main --predict_on=custom:uflow/assets/Jellyfish/tfrecord/test/ --generated_dir=uflow/generated/Jellyfish_live_cycle_normal_batch8/  --use_seg_points --use_tracking_points --width=512 --height=512

## Prediction

##### To predict Phase Contrast or Fluorescence Confocal live cell videos all at once, modify and run
>./batch_predict.sh

##### To predict an individual live cell dataset
>python -m src.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/valid/ --generated_dir=uflow/generated/pc_5small_dense_matlab_seg_all_points_match_lam3_batch8/  --use_seg_points --use_tracking_points --width=256 --height=256

>python -m src.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/valid/ --generated_dir=uflow/generated/HACKS_live_cycle_normal_batch8/  --use_seg_points --use_tracking_points --width=512 --height=512

## Citation
```
@inproceedings{jang2023contourtracking,
  title={Unsupervised Contour Tracking of Live Cells by Mechanical and Cycle Consistency Losses},
  author={Jang, Junbong and Lee, Kwonmoo and Kim, Tae-Kyun},
  booktitle={CVPR},
  year={2023}
}
```

## Acknowledgements
This program is built upon **[UFlow](https://github.com/google-research/google-research/tree/master/uflow)**, a library for research on unsupervised learning of optical flow from **[What Matters in Unsupervised Optical Flow](https://arxiv.org/pdf/2006.04902.pdf)**.
Also, we utilized the **[code](https://github.com/ghnam-ken/PoST)** from **[Polygonal Point Set Tracking](https://arxiv.org/pdf/2105.14584.pdf)** paper and **[cellular morphodynamics profiling software](https://github.com/DanuserLab/Windowing-Protrusion)** from **[Morphodynamic Profiling of Protrusion Phenotypes](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1367294/)** paper.
<br />We thank every member of KAIST Computer Vision and Learning (KCVL) lab for providing valuable feedbacks and moral supports and proofreading our manuscript.
