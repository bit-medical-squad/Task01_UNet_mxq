# 网络复现作业一：复现UNet

## 任务
分割眼球

## 数据
头部CT数据15例，切分成566张切片（from lwt）
- 训练集：12例数据（465slice）
- 验证集：3例数据（101slice）

## 结果
每个切片验证
- mean dice: 0.8015592903205554

global 验证
- 31dice 0.8888420182680636
- 32dice 0.9070051716031969
- 34dice 0.8408024089983173
