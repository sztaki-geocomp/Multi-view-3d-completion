# MVPCC-Net: Multi-View Based Point Cloud Completion Network for MLS Data
Supplementary material to our submitted paper in  Image and Vision Computing Journal.





## Dataset
The full dataset will be available after publication.

## Code 

The code for training and inference our model will be available after publication.


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




# Multi-view Based 3D Point Cloud Completion Algorithm for Vehicles
Supplementary material to our paper published in the 26TH International Conference on Pattern Recognition August 21-25, 2022 • Montréal Québec. [link](https://ieeexplore.ieee.org/abstract/document/9956459)

[Video](https://youtu.be/_g9bZVt9Wc8) 
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/_g9bZVt9Wc8/0.jpg)](https://www.youtube.com/watch?v=_g9bZVt9Wc8)



## Citation
If you found this work helpful for your research, or used some part of the code, please cite our paper:

```text
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
<img src="https://epicinnolabs.hu/wp-content/uploads/2019/10/sztaki_logo_2019_uj_kek.png" width="200">
