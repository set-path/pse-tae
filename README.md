# 遥感作物分类

参考项目[pytorch-psetae](https://github.com/VSainteuf/pytorch-psetae)

论文地址[Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention](https://openaccess.thecvf.com/content_CVPR_2020/html/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.html)

## 配置环境

```shell
git clone https://github.com/set-path/pse-tae.git
cd PSE-TAE
conda create -n env_name python=3.8
conda activate env_name
pip install -r requirements.txt
```

## 项目文件说明

`data/`：存储推理需要使用的辅助数据

`learning/`：模型初始化方法，损失计算方法，指标计算方法

`models/`：模型文件

`checkpoints/`：模型参数文件

`dataset.py`：数据集文件

`example/`：示例数据

`predict_code.py`：推理接口

`preprocessing.py`：数据预处理接口


## 接口说明

接口支持数据格式：多个日期的tiff文件，以日期命名，例如*20220707.tiff*

接口参数说明：

`--data`：指向待推理的时序数据文件夹（预处理后）

`--weight_dir`：预训练参数文件夹（不需要手动指定）

`--fold`：选择使用的预训练参数（训练使用5折交叉验证，所以有5个预训练参数），可选1~5或all

`--device`：推理使用的设备

`--num_classes`：分类数量

其余参数一般不需要改变

## 推理一般流程

1. 首先将以日期命名的单个地块的tiff文件放到同一个文件夹下

2. 进行数据预处理，将多个维度为$(channel,width,height)$转换成$(sequence,channel,N)$，其中$N$是一幅图像中的所有有效像素数，格式为`.npy`，存储在`你的待推理地块文件夹/DATA`目录下，同时会在`你的待推理地块文件夹/META`目录下生成`geomfeat.json`文件，用于辅助模型推理

```shell
python preprocessing.py --path 你的待推理地块文件夹路径 --noData 0 --shpPath shp文件路径
```
参数说明

`--path`：指向待推理的时序数据文件夹

`--noData`：计算有效像素N，忽略noData的值，默认为0

`--shpPath`：用于计算地块的几何特征，在shp文件的相同路径下要有该地块的`.shx`、`.prj`、`.dbf`文件，且文件名相同

3. 推理

```shell
python predict_code.py --data 你的待推理地块文件夹路径
```

## 示例

```shell
python preprocessing.py --path example\youcai_050107953_23940 --noData 32767 --shpPath example\youcai_050107953_23940.shp

python predict_code.py --data example\youcai_050107953_23940

output:油菜
```