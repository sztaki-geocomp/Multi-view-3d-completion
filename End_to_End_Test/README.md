# Using and Testing the MVPCC-Net with pre-trained modles 


## Installation 

- Install python requirements:
```sh
pip install -r requirements.txt
```


## Testing 

You can test the model on all steps: 1) Multi-view 3D Representation, 2) Completion Model, 3) Re-projection, using the script `test_end_to_end.py`.
The pre-trained Completion Model usign 4 views can be downloaded using this [link](https://drive.google.com/file/d/1M3i2DIkBEferKenQFqGZip_XW04L_94-/view?usp=share_link), and copz it under your chechpoints directory `./checkpoints`. 

You can test the algorthim by runing:
```sh
python test_end_to_end.py --input [path to input directory or file]
```

We provide some test examples under `./in` directory, and you can try and run:

```sh
python test_end_to_end.py --input in/
```

By default, the script first projects the point cloud from multiple views and saves the restuls in `./out_project` directory, the Completion Model completes the missing regions of the images and the results will be save in `./out_completed` directory.
The completed 3D point cloud model is obtained by Re-projection of the inpainted multi-view images, and tge results are saved in `./pcd_completed`  directory. 


For showing the results and the input, you can visualize in parallel the 3D input point cloud and our method’s output rotating around their vertical axes, 
by run: 

```sh
python visualizer.py --input  [path to our result file]
```
 
For example: 

```sh
python visualizer.py --input pcd_completed/1a0bc9ab92c915167ae33d942430658c_4.pcd 
```
