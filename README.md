# CS231n 作业：本地运行版

本项目保存斯坦福 CS231n 课程的个人学习代码，并将原本面向 Google Colab 的笔记本调整为本地 Jupyter 环境运行。

| 目录 | 内容 |
| --- | --- |
| `assignment1/` | kNN、Softmax、全连接网络、图像特征 |
| `assignment2/` | Batch Normalization、Dropout、CNN、PyTorch、RNN Captioning |
| `assignment3/` | Transformer、SimCLR、DDPM、CLIP / DINO |

课程材料来源：[Stanford CS231n](https://cs231n.stanford.edu/)。

## 安装环境

使用 **Conda 创建 Python 3.11 环境**，并从项目根目录执行以下命令。安装全部依赖需要访问 PyPI 和 GitHub；OpenAI CLIP 从固定 Git 提交安装，因此还需要 Git。

```bash
conda create -n cs231n python=3.11 pip -y
conda activate cs231n
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name cs231n --display-name "Python (cs231n)"
jupyter lab
```


