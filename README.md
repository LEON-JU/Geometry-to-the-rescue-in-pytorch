This is my code, based on monodepth2, removing monocular training related modules, only preserve stereo training.

Environment:
- python==3.8.18
- CUDA == 11.8
- torch == 2.1.0
- torchvision == 0.16.0
- tensorboardX == 1.4

dataset preparation:

first download KITTI dataset, in my case I download a subset of KITTI, containing the categories of city and road. Then use "modify_split_list.py" to remove items that are not contained in your dataset. **If you want to train my flipping model, use "modify_left" to remove left or right angles from the split list**

There's four models mentioned in my paper: simple, min, automasking, flipping

simple model training command:
```
python train.py --model_name my_model --frame_ids 0 --use_stereo --split eigen_full --avg_reprojection --disable_automasking
```

min model training command:
```
python train.py --model_name my_model --frame_ids 0 --use_stereo --split eigen_full --disable_automasking
```

automasking model training command:
```
python train.py --model_name my_model --frame_ids 0 --use_stereo --split eigen_full
```

flipping model training command:
*first uncommment line 208 - line 214 in trainer.py*
```
python train.py --model_name my_model --frame_ids 0 --use_stereo --split eigen_full
```

For evaluation, first run:
```
python export_gt_depth.py --data_path kitti_data --split eigen
```
then run
```
python evaluate_depth.py --load_weights_folder ~/tmp/my_model/models/weights_19/ --eval_stereo
```

If you want to see the loss curve or training data, run:
```
tensorboard --logdir C:\Users\YOURUSERNAME\tmp\my_model
tensorboard --logdir = ~\tmp\my_model
```
(windows and ubuntu commmand)