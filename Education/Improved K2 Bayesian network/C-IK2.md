##  **New automatic concept map generation model C-IK2**
## introduction
自动化构建概念图模型是通过改进LPG算法和K2算法来自动化构建概念图，其实质是通过改进K2算法的输入序列以得到更加精确的网络结构，更好的网络质量和具有层次结构的概念图。实验对六个公开数据集进行评估，其中一个是医学数据集，另外五个是教育学相关数据集。
## Requirements

 

 - Pycharm
 - Rstudio
	 - bnlearn
 - Matlab
	 - FullBNT-1.0.4

## 核心内容

 - 核心模型C-IK2在new_k2_order.py
 - hc_mmhc文件夹中包含传统的基于分数的贝叶斯网络结构学习算法hc和mmhc，该文件夹内容使用R语言创建，生成结果为BIC值与C-IK2模型做对比实验
 - new_k2_order.py 的结果有K2算法的输入序列和评价标准BIC值
 - 由于论文中使用Matlab软件做对比实验，实验结合Matlab中K2算法和new_k2_order.py中生成的输入序列，最终得到一个图结构，用于与其他基于分数的贝叶斯网络结构学习算法做比较

## Training
hc_mmhc将此文件夹导入Rstudio 

 - 例如hc_algorithms:
  运行asia数据集，将100000数据集分为10个数据集进行分析，得到结果以下面为例：

  ![微信图片_20210411153439](C:\Users\LeiBaoXin\Desktop\C-IK2代码\微信图片_20210411153439.png)

  其中Asia 1[1]中显示的值是BIC值

 - new_K2_order.py（使用python）
 直接改变数据集就可以得到不同的结果topo 表示K2的输入序列，bic1 表示BIC值!
 
 ![微信图片_20210411153444](C:\Users\LeiBaoXin\Desktop\C-IK2代码\微信图片_20210411153444.png)
 
 - K2_test2（使用matlab）
 将new_K2_order.py中所获得的topo，作为K2_test2中order的值，最终得到相应的贝叶斯网络结构图
![微信图片_20210411153423](C:\Users\LeiBaoXin\Desktop\C-IK2代码\微信图片_20210411153423.png)

