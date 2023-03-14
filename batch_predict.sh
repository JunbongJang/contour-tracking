#!/bin/bash
#for a_index in {1..9}
#do
#  python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test$a_index/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_cycle_normal_batch8_not/ --use_seg_points --use_tracking_points
#
#done

#for a_index in {1..4}
#do
#  python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test$a_index/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_cycle_normal_batch8_not/ --use_seg_points --use_tracking_points
#
#done

# For PC, run the following weights to replicate manuscript results
    # spatial2_batch8
    # cycle_batch8
    # photometric_spatial_batch8
    # normal_spatial_batch8
    # normal_linear_batch8
    # cycle_normal_linear_batch8
    # cycle_normal_decay_batch8

    # cycle_normal_nocross_batch8
    # cycle_normal_1cross_batch8
    # cycle_normal_circconv_batch8
    # cycle_normal_1dconv_batch8

    # post_batch8
    # HACKS_live_post_batch4 (train on HACKS, eval on PC dataset yields the best performance)

# For HACKS, run the following weights to replicate manuscript results
    # cycle_normal_batch8

    # post_batch4


project_name=post_batch4
python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test1/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/  --use_seg_points --use_tracking_points
python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test2/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/  --use_seg_points --use_tracking_points
python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test3/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/  --use_seg_points --use_tracking_points --width=576 --height=544
python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test4/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/  --use_seg_points --use_tracking_points --width=384 --height=480

# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test1/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/  --use_seg_points --use_tracking_points
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test2/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/  --use_seg_points --use_tracking_points
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test3/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/  --use_seg_points --use_tracking_points --width=576 --height=544
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/pc_5small_dense_matlab_seg_all_points/tfrecord/test4/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/  --use_seg_points --use_tracking_points --width=384 --height=480


# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test1/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points --width=800 --height=640
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test2/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test3/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points --width=416 --height=672
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test4/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points --width=544 --height=672
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test5/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points --width=512 --height=480
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test6/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points --width=672 --height=992
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test7/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test8/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points
# python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test9/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/HACKS_live_$project_name/ --use_seg_points --use_tracking_points

#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test1/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points --width=800 --height=640
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test2/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test3/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points --width=416 --height=672
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test4/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points --width=544 --height=672
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test5/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points --width=512 --height=480
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test6/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points --width=672 --height=992
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test7/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test8/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points
#python -m uflow.contour_flow_main --predict_on=custom:uflow/assets/HACKS_live/tfrecord/test9/ --generated_dir=/data/junbong/optical_flow/generated_3_5_2023/pc_5small_dense_matlab_seg_all_points_$project_name/ --use_seg_points --use_tracking_points