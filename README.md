# MVPCC-Net: Multi-View Based Point Cloud Completion Network for MLS Data
Supplementary material to our  **[paper](https://www.sciencedirect.com/science/article/pii/S0262885623000495)** at  Image and Vision Computing Journal.


## Introduction 

We introduce a point cloud completion Network for generating high resolution, dense 3D object models from partial point cloud Mobile Laser Scanning (MLS) measurement data.

The dataflow of the model: 

![Dataflow15(2)](https://user-images.githubusercontent.com/101256004/229819119-200e260a-1cc0-4a3d-99f6-a2947ba4258c.PNG)




## Deom of the results:

In this video, we visualize in parallel the 3D input point cloud and our method’s output rotating around their vertical
axes, to show them from all perspectives.  

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/iQ_SWNuF2a4/0.jpg)](https://youtu.be/iQ_SWNuF2a4)






## Data Generation

## Synthetic Dataset
The full dataset will be available after publication. Some samples are given in  Data_samples folder

Our dataset contains a total of 4918 distinct models, of which 4580 are used to train our model and 338 are used as evaluation
data. Twenty partial samples were generated for each sample using twenty distinct perspectives, yielding a training set of 91 600 objects and a test set of 6760 samples.

The geometry inforamtion data can be downloaded using this [link](https://drive.google.com/file/d/1YVD8Na5LXrGLodpFXG33Cf_nl6Eifu3z/view?usp=share_link) (images in format .png)  (111 GB) 

The color information data can be downloaded using this [link](https://drive.google.com/file/d/1IwJ9BvT5oH4ui4EmigEv_y5T7Q8wVuzn/view?usp=share_link) (images in format .png)  (37,6 GB) 

The point cloud data can be downloaded using this [link](https://drive.google.com/file/d/1LzHrk0tu9kZqNVxvsul1MX2SfWx39Goe/view?usp=share_link) (in .pcd format) (15,8 GB) 

### Dataset structure
We provide the data in raw images for the XYZ channels and the RGB channels.

    ├── Data                        # Main dataset folder
    │   ├── GT                      # Contains Ground Truth samples
    │   │   ├── train               # Samples in raw image format
    │   │   │   ├── synsetId 
    │   │   │   │   ├── modelId 
    |   │   │   │   │   ├── images 
    │   │   ├── test             
    │   │   │   ├── synsetId 
    │   │   │   │   ├── modelId 
    |   │   │   │   │   ├── images 
    │   ├── in                     # Contains Input samples 
    │   │   ├── train              # Samples in raw image format
    │   │   │   ├── synsetId  
    │   │   │   │   ├── modelId 
    |   │   │   │   │   ├── samples 
    |   |   │   │   │   │   ├── images
    │   │   ├── test             
    │   │   │   ├── synsetId 
    │   │   │   │   ├── modelId 
    |   │   │   │   │   ├── samples
    |   |   │   │   │   │   ├── images
    └── ...


## Real Dataset
We created a real-world test set that consists of (mostly partial) vehicle point clouds extracted from  measurements of a Riegl VMX-450 MLS scanner. The raw MLS test data was provided by the  City Council's Road Management Department (Budapest Kőzút Zrt.) in Budapest, Hungary.
Our real MLS data collection consists  of 424 object samples in total. On one hand, 370 point clouds represent partial vehicle shapes, where the scans of the complete objects are not available in the MLS data, thus they can only be used for qualitative analysis of the proposed technique (can be downloaded using  this [link](https://drive.google.com/file/d/1qQAIYEWnkPKyl_7C6nGzLq2LBZp362VL/view?usp=share_link)). 

On the other hand, 54 samples depict almost entire vehicle shapes, which can be also used as ground truth similarly to the synthetic models. Here we generated four partial point cloud samples from each complete MLS vehicle shape generating overall 216 samples, each one was created by reprojecting an image created from a single virtual camera position, which was located in the front, behind, to the right, or to the left of the selected object of interest. The data can be downloaded using this [link](https://drive.google.com/file/d/1vlUZ6NdEK2aUgz7RH3CNoNpIofrP6gS_/view?usp=share_link).


## Code 

The project was implemented on Ubuntu 20.04 with CUDA 11.1 (with compatible CUDNN) using a 32-GB RAM and a NVIDIA GeForce RTX 3060 Ti GPU. All codes were implemented in python 3.7.11 with packages pytorch 1.12.0 using conda virtual environment. 

## Installation 

- Clone this repo: 
```sh
git clone https://github.com/sztaki-geocomp/Multi-view-3d-completion.git
cd Multi-view-3d-completion
```

- Install python requirements:
```sh
pip install -r requirements.txt
```

## 1) Training
To train the model, create a `config.yaml` file similar to the [example config file](./checkpoints/config.yml) and copy it under your checkpoints directory. 
To train the model:

```sh
python train.py  --checkpoints [path to checkpoints] \
  --views 5 6 7 8
```

## 2) Testng
To test the model, create a `config.yaml` file similar to the [example config file](./checkpoints/config.yml) and copy it under your checkpoints directory. To test the model:
```sh
python test.py \
  --checkpoints [path to checkpoints] \
  --views 5 6 7 8
  --input [path to input directory or file] \
  --output [path to the output directory]
```

For testing the algorthim with pretrained model you can see [End_to_End_Test](./End_to_End_Test) for geometry information completion and [End_to_End_Test_color](./End_to_End_Test_color) for geomerty and color information completion. 

## Citation
If you found this work helpful for your research, or used some part of the code, please cite our paper:

```text
@Article{mvpccnet,
AUTHOR = {Ibrahim, Yahya and Benedek, Csaba},
TITLE = {MVPCC-Net: Multi-View Based Point Cloud Completion Network for MLS Data},
JOURNAL = {Image and Vision Computing},
VOLUME = {}}


@INPROCEEDINGS{9956459,
  author={Ibrahim, Yahya and Nagy, Balázs and Benedek, Csaba},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Multi-view Based 3D Point Cloud Completion Algorithm for Vehicles}, 
  year={2022},
  volume={},
  number={},
  pages={2121-2127},
  doi={10.1109/ICPR56361.2022.9956459}}


```



## Authorship declaration
The code of this repository was implemented in the [Machine Perception Research Laboratory](https://www.sztaki.hu/en/science/departments/mplab), Institute of Computer Science and Control (SZTAKI), Budapest.\
<img src="https://user-images.githubusercontent.com/50795664/195994236-1579001a-e78e-4638-9cbe-496d4b9a73d2.png" width="200">
