# 金融异常检测任务

## 1. 实验介绍

反欺诈是金融行业永恒的主题，在互联网金融信贷业务中，数字金融反欺诈技术已经得到广泛应用并取得良好效果，这其中包括了近几年迅速发展并在各个领域
得到越来越广泛应用的神经网络。本项目以互联网智能风控为背景，从用户相互关联和影响的视角，探索满足风控反欺诈领域需求的，可拓展、高效的神经
网络应用方案，从而帮助更好地识别欺诈用户。

本项目主要关于实现预测模型(**项目用图神经网络举例，具体实现可以使用其他模型**)，进行节点异常检测任务，并验证模型精度。而本项目基于的数据集[DGraph](https://dgraph.xinye.com/introduction)，[DGraph](https://dgraph.xinye.com/introduction)
是大规模动态图数据集的集合，由真实金融场景中随着时间演变事件和标签构成。

### 1.1 实验目的

- 了解如何使用 Pytorch 进行神经网络训练
- 了解如何使用 Pytorch-geometric 等图网络深度学习库进行简单图神经网络设计(推荐使用 GAT, GraphSAGE 模型)。
- 了解如何利用 MO 平台进行模型性能评估。

### 1.2 预备知识

- 具备一定的深度学习理论知识，如卷积神经网络、损失函数、优化器，训练策略等。
- 了解并熟悉 Pytorch 计算框架。
- 学习 Pytorch-geometric，请前往：<https://pytorch-geometric.readthedocs.io/en/latest/>

### 1.3 实验环境

- python = 3.9.5
- numpy = 1.26.4
- pandas = 
- pytorch = 2.3.1
- torch_geometric = 2.5.3
- torch_scatter = 2.1.2
- torch_sparse = 0.6.18

### 1.4 实验环境准备

使用 conda 进行环境准备，具体步骤如下：

1. 安装 Miniconda 或 Anaconda，请参考官方文档：<https://www.anaconda.com/>

2. 创建 conda 环境

   ```shell
   # 创建 conda 环境
   conda create -n mo_graph python=3.9.5
   # 激活环境
   conda activate mo_graph
   ```

3. 安装 Pytorch 环境，请参考官方文档：<https://pytorch.org/get-started/previous-versions/>

   ```shell
   # CUDA 11.8
   conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
   # CUDA 12.1
   conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   # CPU Only
   conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 cpuonly -c pytorch
   ```

4. 安装 pyg 相关包，包括 torch_geometric、torch_sparse、torch_scatter <https://github.com/pyg-team/pytorch_geometric>
   使用 conda 安装可能会出现版本问题，建议使用 pip 安装。

   ```shell
   # 1. 先安装扩展包
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+${CUDA}.html
   # where ${CUDA} should be replaced by either cpu, cu118, or cu121 depending on your PyTorch installation.
   # e.g. pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
   # 2. 安装 torch_geometric
   pip install torch_geometric==2.5.3
   ```

5. 安装 numpy 及其他依赖包

   ```shell
   conda install numpy=1.26.4 pandas
   ```
