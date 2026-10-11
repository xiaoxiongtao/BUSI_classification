# BUSI 乳腺超声图像分类

基于 PyTorch 的 **BUSI（Breast Ultrasound Images）** 三分类项目，对比 **ResNet** 与 **ViT（Vision Transformer）** 两种模型的性能。

- 类别：`benign`（良性）、`malignant`（恶性）、`normal`（正常）
- 任务：图像分类
- 框架：PyTorch
- 附带完整训练脚本、评估指标、可视化，以及一个 Jupyter Notebook 用于数据探索与结果分析。

## 数据集

[Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)

BUSI 数据集包含 780 张乳腺超声图像，共三类：

| 类别 | 数量 |
| --- | --- |
| benign | 437 |
| malignant | 210 |
| normal | 133 |

下载并解压后，将 `Dataset_BUSI_with_GT` 文件夹放到 `data/` 目录下：

```
data/
└── Dataset_BUSI_with_GT/
    ├── benign/
    ├── malignant/
    └── normal/
```

> 每个类别文件夹里还包含分割掩膜（文件名含 `_mask`），代码会自动忽略，仅使用原始图像。

## 项目结构

```
BUSI_classification/
├── data/
│   └── Dataset_BUSI_with_GT/   # 数据集（需自行下载放入）
├── models/
│   ├── resnet.py               # ResNet18/34/50/101（支持 ImageNet 预训练）
│   └── vit.py                  # ViT（从零实现 + torchvision 预训练）
├── utils/
│   ├── dataset.py              # 数据加载、划分、数据增强
│   ├── metrics.py              # 准确率/精确率/召回率/F1/混淆矩阵
│   └── visualization.py        # 可视化工具
├── notebooks/
│   └── BUSI_classification.ipynb  # 数据展示 + 模型对比 + 结果可视化
├── config.py                   # 全局配置
├── train.py                    # 训练入口（ResNet / ViT）
├── evaluate.py                 # 评估入口（加载权重并在测试集/验证集上评估）
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.10 ~ 3.12（PyTorch 对这些版本的预编译支持最完整）。

```bash
pip install -r requirements.txt
```

如需 GPU 版 PyTorch，请参考 [pytorch.org](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本。

## 训练

两个模型默认都会加载 ImageNet 预训练权重。训练 ResNet18（默认）：

```bash
python train.py --model resnet --depth 18 --epochs 50
```

训练预训练 ViT（默认 ViT-B/16）：

```bash
python train.py --model vit --vit-variant b16 --epochs 50
```

也可以训练更深的 ResNet、换其他 ViT 变体（b16/b32/l16/l32）、从零训练或关闭类别权重：

```bash
python train.py --model resnet --depth 50
python train.py --model vit --vit-variant l16
python train.py --model vit --no-pretrained
python train.py --model resnet --no-use-class-weights
```

常用命令行参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | `resnet` 或 `vit` | `resnet` |
| `--depth` | ResNet 深度（18/34/50/101） | `18` |
| `--vit-variant` | 预训练 ViT 变体（b16/b32/l16/l32） | `b16` |
| `--img-size` | 输入图像尺寸 | `224` |
| `--batch-size` | 批量大小 | `32` |
| `--epochs` | 训练轮数 | `50` |
| `--lr` | 学习率（ResNet 3e-4，预训练 ViT 1e-4） | 按模型 |
| `--pretrained` / `--no-pretrained` | 是否加载 ImageNet 预训练权重 | 开启 |
| `--use-class-weights` / `--no-use-class-weights` | 是否使用类别权重 | 开启 |
| `--label-smoothing` | 标签平滑系数 | `0.1` |
| `--early-stop` | 早停耐心值（0 关闭） | `15` |
| `--device` | `auto` / `cuda` / `cpu` | `auto` |

其余参数（图像尺寸、ViT 结构、学习率等）见 `config.py` 和 `python train.py --help`。

### 训练输出

训练过程会打印每个 epoch 的 train/val 损失与指标（Accuracy / Precision / Recall / F1），并输出到 `outputs/`：

```
outputs/
├── checkpoints/
│   ├── resnet18_best.pt      # 验证集 F1 最高的权重（命名与模型对应）
│   ├── resnet50_best.pt
│   ├── vit_b16_best.pt       # 预训练 ViT-B/16
│   └── vit_best.pt           # 从零实现的轻量 ViT
└── logs/
    ├── resnet18_history.json # 每轮 loss/指标曲线数据
    ├── vit_b16_curves.png    # 训练曲线图
    ├── vit_b16_confusion.png # 测试集混淆矩阵
    └── ...
```

## 评估

训练完成后，用 `evaluate.py` 加载权重并在测试集上评估（默认加载 `outputs/checkpoints/` 下的最佳权重）：

```bash
python evaluate.py --model resnet --depth 18
python evaluate.py --model vit
```

可指定其他权重或评估验证集：

```bash
python evaluate.py --model resnet --checkpoint outputs/checkpoints/resnet18_last.pt
python evaluate.py --model resnet --split val
```

它会打印总体指标（loss / Accuracy / macro Precision / Recall / F1）、各类别的分类报告，并把混淆矩阵图与指标 JSON 保存到 `outputs/logs/`。

## 可视化与对比（Jupyter Notebook）

启动 Jupyter：

```bash
jupyter notebook
```

打开 `notebooks/BUSI_classification.ipynb`，依次执行即可：

1. 数据集展示（类别分布、样本图像）
2. 自动发现 `outputs/checkpoints/` 中所有已训练的模型（各种 ResNet 深度、预训练 ViT 变体、从零实现的 ViT）
3. 在测试集上统一评估，得到一张按 F1 排序的对比表（Accuracy / Precision / Recall / F1）
4. 并排对比混淆矩阵、预测样例
5. 逐个查看训练曲线

> Notebook 会自动扫描 `outputs/checkpoints/*_best.pt`，因此训练了哪几个模型就对比哪几个，无需手动改代码。请先完成训练，再运行"发现并加载模型"及其后的单元。

## 实现说明

- **ResNet**：实现了 ResNet18/34/50/101（`BasicBlock` / `Bottleneck`），
  支持任意类别数；默认加载 ImageNet 预训练权重，也可用 `--no-pretrained` 从零训练。
- **ViT**：提供两套实现——从零实现的轻量 ViT（Patch Embedding + 可学习 class token +
  位置编码 + Transformer Encoder + MLP Head），以及 torchvision 预训练 ViT
  （b16/b32/l16/l32，默认 b16）；`--no-pretrained` 切换为从零实现。
- **数据划分**：按类别分层划分 70% / 15% / 15%（train / val / test），
  保证各类比例一致。
- **指标**：每个 epoch 计算并记录损失、Accuracy、macro Precision / Recall / F1，
  以及测试集混淆矩阵。
- **公平对比**：不同模型使用相同的划分、数据增强、优化器和训练流程。

## 备注

- BUSI 数据集存在类别不平衡，类别权重默认已开启（`--use-class-weights`），
  有助于提升小类的召回率；可用 `--no-use-class-weights` 关闭。
- 小数据量下，ResNet 依靠卷积的归纳偏置通常更容易收敛；ViT 通常需要更多数据
  或更强数据增强才能充分发挥，实际结论以本地训练结果为准。

## 作者与版权

- **作者**：肖雄涛
- **版权**：Copyright © 2026 肖雄涛，保留所有权利。
- **用途许可**：本项目（含代码、训练脚本与结果）仅供**学习、教学、科研等非商业用途**使用，**禁止用于任何商业目的**；未经作者书面许可，不得将本项目用于商业用途，或进行销售、再分发、商业部署。