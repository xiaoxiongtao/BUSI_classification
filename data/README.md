# 数据集目录

将 BUSI（乳腺超声图像）数据集放在本目录下，目录结构应如下：

```
data/
└── Dataset_BUSI_with_GT/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png
    │   └── ...
    ├── malignant/
    │   ├── malignant (1).png
    │   ├── malignant (1)_mask.png
    │   └── ...
    └── normal/
        ├── normal (1).png
        ├── normal (1)_mask.png
        └── ...
```

下载地址（Kaggle）：
https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset

说明：

- 三个类别：`benign`（良性）、`malignant`（恶性）、`normal`（正常）。
- 每个类别文件夹中同时包含原始超声图像和对应的分割掩膜
  （文件名含 `_mask`），代码会自动忽略掩膜，仅使用原始图像做分类。
- 如果解压后的目录名不是 `Dataset_BUSI_with_GT`，请修改
  `config.py` 中的 `DATA_DIR` 指向实际目录。
