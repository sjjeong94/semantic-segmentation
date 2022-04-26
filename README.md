# Semantic Segmentation
This is a PyTorch implementation of semantic segmentation models on [comma10k dataset](https://github.com/commaai/comma10k)

---
## Dataset
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [comma10k](https://github.com/commaai/comma10k)


## comma10k

### 1-Epoch Training Results
| Case  | Model                               | Duration (s) | Train Loss | Val Loss |
| ----- | ----------------------------------- | ------------ | ---------- | -------- |
| 000   | lraspp_mobilenet_v3_large (bs=2)    | 1419.1662    | 0.1236     | 0.1616   |
| 001   | fcn_resnet50 (bs=2)                 | 1796.9648    | 0.1395     | 0.1984   |
| 002   | deeplabv3_mobilenet_v3_large (bs=4) | 1812.6592    | 0.1099     | 0.0848   |
| 002_1 | bs=8                                | 1731.3506    | 0.1141     | 0.0814   |
| 002_2 | input normalization                 | 1688.1650    | 0.1144     | 0.0795   |
