# 模型训练、构造技巧

## 实用技巧

带 batch 的矩阵乘法

```python
batch_size = 4
embedding_dim = 8
e1=torch.randn(batch_size, embedding_dim)
e2=torch.randn(batch_size, embedding_dim)
m=torch.sum(torch.mul(e1, e2), dim=1)  # 计算每个batch的内积
```

## 实用库

### Albumentations

用于**图象增强**的库，相比 torchvision.transforms，albumentations 拥有更强的工业化能力，包括：

- 更多的功能，如分割掩码(mask)、边界框(bbox)和关键点(keypoints)的同步增强
- 提供了丰富的增强方法，包括几何变换、颜色变换和高级增强（如弹性变换、网格失真等）
- 基于 OpenCV 和 NumPy，性能高效，尤其适合大规模数据集

适合需要高效处理大规模数据集的场景，如 Kaggle 竞赛和工业级应用。提供了 ToTensorV2 方法，将增强后的图像转换为 PyTorch 张量
