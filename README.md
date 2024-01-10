## From 2D images to 3D model: Weakly Supervised Multi-View Face Reconstruction with Deep Fusion
##### Multi-view reconstruction with 3DMM texture
![avatar](./doc/example.jpeg#pic_center)

###### Landmarks: the blue dot is groundtruth
![avatar](./doc/example_lm.jpeg#pic_center)


#### Environment
RTX 3090 & RTX 2080 Ti  <br>
For 3090: We recommend CUDA 11.1 & Python v3.8.10  & Pytorch 1.9.0 
    
    conda create -n dfnet python=3.8.10
    source activate dfnet
    conda install -c conda-forge cudatoolkit=11.1 cudnn=8.1.0
    conda install -c pytorch pytorch=1.9.0 torchvision
    conda install tensorflow=2.5.0 numpy=1.20
    pip install -r lib/requirement.txt
  
#### Third Party lib:  
[pytorch3d](https://github.com/facebookresearch/pytorch3d)  
[involution](https://github.com/d-li14/involution)  
[face-parsing](https://github.com/zllrunning/face-parsing.PyTorch)  
[BFM Crop](https://github.com/sicxu/Deep3DFaceRecon_pytorch)

Pytorch3d:

    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    conda install pytorch3d=0.5.0 -c pytorch3d

Involution:

We simplified the code in the lib/involution directory, you don't need to download the source code. The following dependencies need to be installed:
    
    pip install mmcv
    pip install cupy-cuda111
    



#### Evaluation metric:
[Now_evalution](https://github.com/soubhiksanyal/now_evaluation)
  
We simplified the now_evaluation code in the 'lib/now_evaluation' directory. You needn't to download it.
    
Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh) within the virtual environment.

    cd lib/now_evaluation
    pip install -r requirements.txt
    git clone https://github.com/MPI-IS/mesh.git
    cd mesh
    conda install -c statiskit libboost-dev
    BOOST_INCLUDE_DIRS=/path/to/boost/include make all
    (Example: BOOST_INCLUDE_DIRS=/home/pointcloud/.conda/envs/dfnet/include/ make all)
    
Clone the [flame-fitting](https://github.com/Rubikplayer/flame-fitting) repository and copy the required folders by the following comments

    cd lib/now_evaluation
    git clone https://github.com/Rubikplayer/flame-fitting.git
    cp flame-fitting/smpl_webuser ./smpl_webuser -r
    cp flame-fitting/sbody ./sbody -r
    
Clone [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and copy the it to the following folder

    cd lib/now_evaluation
    git clone https://gitlab.com/libeigen/eigen.git
    cp eigen ./sbody/alignment/mesh_distance/eigen -r

Edit the file 'now_evaluation/sbody/alignment/mesh_distance/setup.py' to set EIGEN_DIR to the location of Eigen. Then compile the code by following command
    
    cd lib/now_evaluation/sbody/alignment/mesh_distance
    vim setup.py  
    (change c++11 to c++14, then save)
    make
    
    
#### Dataset
Please contact [pixel-face](https://github.com/zyg11/Pixel-Face) for 'Pixel_Face' dataset

#### Train
Download the face mask pretrain pth [face_mask.pth](https://drive.google.com/file/d/1wldpAhrPUVYy7JMtgt6aaYkMID_VsalI/view?usp=drive_link) into 'DF_MVR/pretrain/'


Download the BFM front face mat [BFM_model_front.mat](https://drive.google.com/file/d/1XJZVxpJLWQz5jSNJiFTjI6ZRkowMTGAp/view?usp=drive_link) into 'lib/BFM/model_basis/BFM2009/'

    python train.py

#### Test
Download our newest DF_MVR pretrain pth [000000279.pth](https://drive.google.com/file/d/1rA8V6-ZP8wLp531NuM2TF8PgbqdsaiT8/view?usp=drive_link) (RMSE 1.4040) into 'DF_MVR/pretrain/'
    
    python test_pixel_face

#### Test Pixel Face Samples

    python test_sample.py

## Citation
If you find this work useful in your research, please cite:
```
@article{zhao20222d,
  title={From 2D Images to 3D Model: Weakly Supervised Multi-View Face Reconstruction with Deep Fusion},
  author={Zhao, Weiguang and Yang, Chaolong and Ye, Jianan and Yan, Yuyao and Yang, Xi and Huang, Kaizhu},
  journal={arXiv preprint arXiv:2204.03842},
  year={2022}
}
```