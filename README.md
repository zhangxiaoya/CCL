# CCL_LE

## 简介

这是关于CCL(Connected Component Label)的算法实现，目前只包含了其中的一个算法。详细的论文描述在[Parallel graph component labelling with GPUs and CUDA](https://www.sciencedirect.com/science/article/pii/S0167819110001055).

关于CCL的介绍在[WIKI](https://en.wikipedia.org/wiki/Connected-component_labeling).

## 使用
算法实现在CCL_LE_GPU.cu和CCL_LE_GPU.cuh中，其他的都是测试或者辅助文件，具体的使用在example文件中。

**更新：**
增加两种CCL实现方法，分别在CCL_NP.cu和CCL_DPL.cu文件中，定义分别在其cuh文件中。

初始化一个Mesh结构的原始数据：
```
  2   1   1   1   1   1   1   0   0
  2   0   0   0   1   1   1   1   0
  2   0   0   0   1   1   1   1   0
  2   0   0   0   0   1   1   1   1
  2   0   0   0   0   0   1   1   1
  2   0   0   0   1   1   1   1   1
  2   0   1   1   1   1   0   0   0
  2   0   1   0   0   0   0   0   0
```

最终输出标签Mask：
```
  0   1   1   1   1   1   1   7   7
  0  10  10  10   1   1   1   1   7
  0  10  10  10   1   1   1   1   7
  0  10  10  10  10   1   1   1   1
  0  10  10  10  10  10   1   1   1
  0  10  10  10   1   1   1   1   1
  0  10   1   1   1   1  60  60  60
  0  10   1  60  60  60  60  60  60
```

## Mesh Graph
本来是用于图像分割的label，借鉴mesh结构，把图像看做是一个图的mesh表示形式(不同于图的邻接链表或者稀疏矩阵的表示，Mesh本来就是一个网络结构)，然后应用算法实现Label。关于Mesh的定义在[这里](https://en.wikipedia.org/wiki/Lattice_graph)
