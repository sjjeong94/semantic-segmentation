# Semantic Segmentation
This is a PyTorch implementation of semantic segmentation models on [comma10k dataset](https://github.com/commaai/comma10k)

---
## Dataset
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [comma10k](https://github.com/commaai/comma10k)


## comma10k

### 1-Epoch Training Results
| Case | Model                     | Duration (s) | Train Loss | Val Loss |
| ---- | ------------------------- | ------------ | ---------- | -------- |
| 000  | lraspp_mobilenet_v3_large | 1419.1662    | 0.1236     | 0.1616   |
| 001  | fcn_resnet50              | 1796.9648    | 0.1395     | 0.1984   |
