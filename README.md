# sleep-staging-edge

睡眠分期模型的端侧部署实验仓库。

## 目标
- PyTorch → ONNX → INT8 量化 → NCNN / MNN 端侧推理
- 记录模型大小、延迟、内存、精度四项指标

## 目录
- `scripts/` 实验脚本
- `models/` 模型定义
- `reports/` 实验记录

## 进度
- [ ] 环境搭建
- [x] WSL + PyTorch 环境（torch 2.14.0+cu130, CUDA True）
- [ ] baseline 跑通
- [ ] ONNX 导出
- [ ] INT8 量化
- [ ] 端侧部署
