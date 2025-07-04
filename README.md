<!-- # Freehand Sketch Generation from Mechanical Components

## ACM MM 2024 -->

<div align="center">
<h2>Freehand Sketch Generation from Mechanical Components </center> <br> <center>(ACM MM 2024)</h2>
<div align="center">  <img src='images/momohuhuhuluobo.jpg' style="height:200px"></img>  </div>

<a href='https://mcfreeskegen.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2408.05966'><img src='https://img.shields.io/badge/ArXiv-2408.05966-red'></a> 

<br>

[Zhichao Liao](https://lzc-sg.github.io/)🥕 🐟, [Di Huang](https://di-huang.github.io/)🥕 🐟, [Heming Fang]()🍎 🐟, [Yue Ma]()🥕, [Fengyuan Piao]()🥕 ✉, 
[Xinghui Li]()🥕 ✉, [Long Zeng]()🥕 ✉, [Pingfa Feng]()🥕 


🥕 Tsinghua University  🍎 Zhejiang University

🐟 Co-first authors (equal contribution)   ✉ Corresponding Author
</br>
</div>

<!--
[<a href="https://mcfreeskegen.github.io/">Project Website</a>] | [<a href="https://arxiv.org/abs/2408.05966">Paper</a>] -->


![Framework](images/framework.png)



# Part I
![](../archive/result_img.png)
### Setup
```
conda create --name=part_1 python=3.9
conda activate part_1

# If you are using RTX 4090:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# If you are using CPU only:
conda install pytorch torchvision

conda install -c conda-forge pythonocc-core=7.7.0
pip install -r requirements.txt
```
### Run
##### Edge Detector
```
# Tested on Windows platform.

• 3D -> 2D:
python main.py <input_folder>

• Remove duplicates:
# NOTE: backup your initial data before removing duplicates.
python main.py -rd 2 -rdm <hash or mse> <input_folder>

• Extra processing:
# The data that has undergone extra processing will be stored in the "out_ep" folder.
python main.py -epm <1 for padding-first, 2 for line-width-first> -pad <your_padding_size> -F <png or jpg> -W <desired_image_width> -H <desired_image_height> <input_folder>
```
##### Viewpoint Selector
```
Download ck.pth for ICNet: https://drive.google.com/drive/folders/1N3FSS91e7FkJWUKqT96y_zcsG9CRuIJw
Place "ck.pth" into the directory "models/saved_models/ck.pth".

# The selected data will be stored in the "out_vs" folder.
python main.py -vsm <1 for ICNet> <input_folder>
```


# Part II
![](../archive/architecture.png)
### Setup
```
conda create --name=part_2 python=3.8
conda activate part_2
# If using RTX 4090:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# If using RTX 3080 Ti:
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
cd ..

Download u2net.pth: https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ
Place "u2net.pth" into the directory "third_party/U2Net_/saved_models/".

cd third_party/clipasso/models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../../..

export CUDA_VISIBLE_DEVICES=<Your GPU ID>

# Generated results and logs are stored in the "logs/" directory.
```
### Run
##### Use label data generated by initial CLIPasso, and train our model
```
Place label data generated by initial CLIPasso into the directory "data/".

# NOTE: "<your_num_strokes>" should be the same as the number of strokes generated by CLIPasso. 
python main.py --config_path config/my_data.yaml --clip_model_name RN50 --label_data_source init_clipasso --num_strokes <your_num_strokes>
```
##### Generate label data using the optimized approach, and train our model
```
Place data generated by edge detector into the directory "data/train" as the following directory structure:
data/train
├── class_0
│   └── cad_model_0
│       └── view_0.png
├── class_1
│   └── cad_model_1
│       └── view_0.png
...

# Generate label data firstly. Set "<your_num_strokes>" (e.g. 30) before running.
# NOTE: set "--batch_size 4 --num_workers 4" if using RTX 3080 Ti
python generate_label_data.py --config_path config/my_data.yaml --output_dir logs/my_data --img_paths data/train/*/*/*.png --visualize --save_svgs --device gpu --num_strokes <your_num_strokes>

# Use chunk and run in background
nohup python generate_label_data.py ... --chunk <total_chunk_size> <chunk_index> > log_<total_chunk_size>_<chunk_index>.txt 2>&1 &
python merge_label_data.py --output_file logs/my_data/path.pkl --data_files logs/my_data/path_*

# The label data will be stored in ".pkl" format under "logs/my_data".

# Train the model. Note that "<your_num_strokes>" should be the same as what you set when running "generate_label_data.py".
python main.py --config_path config/my_data.yaml --clip_model_name RN50 --label_data_source pkl --num_strokes <your_num_strokes>
```
