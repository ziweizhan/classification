利用pytorch实现图像分类，其中包含了58个基础网络模型。

## 实现功能
* 基础功能利用pytorch实现图像分类
* 利用flask + redis实现模型云端api部署

## 运行环境
* python3.6
* pytorch 1.7
* torchvision 0.8.0

## 代码仓库的使用

### 数据集形式
原始数据集存储形式为，同个类别的图像存储在同一个文件夹下，所有类别的图像存储在一个主文件夹data下。

```
|-- data
    |-- train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...

    |-- val
        |--*.jpg
```

利用preprocess.py将数据集格式进行转换（个人习惯这种数据集的方式）

```
python ./data/preprocess.py
```

转换后的数据集为，将训练集的路径与类别存储在train.txt文件中，测试机存储在val.txt中.
其中txt文件中的内容为

```
# train.txt

/home/xxx/data/train/label1/*.jpg   label

# val.txt

/home/xxx/data/train/label1/*.jpg
```

2. 在训练的时候使用验证集，得到验证集合的准确率
```shell
python train_val.py
```

### 预测
在cfg.py中`TRAINED_MODEL`参数修改为指定的权重文件存储位置,在predict文件中可以选定是否使用tta

```shell
python predict.py
```
