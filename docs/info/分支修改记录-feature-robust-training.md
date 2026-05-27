# feature/robust-training 分支修改记录

## 1. 分支目标

本分支用于实现“只改训练策略、不改网络结构”的鲁棒训练版本，重点解决：

- 模态缺失
- 模态弱化
- 轻度噪声退化

本分支与其他分支边界固定为：

- 不改 [model/MSFMamba.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/model/MSFMamba.py)
- 不改 [modules/Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)
- 不引入 `ReliabilityGate`
- 不引入 `Cross-State Modulation`

## 2. 当前已完成内容

### 2.1 新增退化构造工具层

已新增：

- [setting/robustness.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting/robustness.py)

当前已实现的工具函数：

- `degrade_hsi_batch(...)`
- `degrade_aux_batch(...)`
- `apply_modality_dropout(...)`
- `build_degraded_batch(...)`
- `summarize_degradation_stats(...)`

当前退化口径为：

- HSI band-wise dropout
- HSI PCA 高斯噪声
- 辅助模态高斯噪声
- 单模态 dropout

并且保证：

- 输入输出 shape 不变
- dtype 不变
- device 不变
- 不会双模态同时全零

### 2.2 训练入口已接入 clean/degraded 双分支

已修改：

- [setting/options.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting/options.py)
- [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py)

当前已新增的训练参数：

- `--robust_train`
- `--lambda_deg`
- `--lambda_cons`
- `--modality_dropout_prob`
- `--hsi_dropout_prob`
- `--hsi_noise_std`
- `--aux_noise_std`

当前训练损失固定为：

```text
L_total = L_ce_clean + lambda_deg * L_ce_deg + lambda_cons * L_cons
```

其中：

- `L_ce_clean`：完整输入交叉熵
- `L_ce_deg`：退化输入交叉熵
- `L_cons`：退化预测与完整预测之间的 KL 一致性

日志中已支持输出：

- `CleanCE`
- `DegCE`
- `Cons`
- `DropHSI`
- `DropAux`
- `HSINoise`
- `AuxNoise`
- `HSIBandDrop`

`epoch_history.csv` 中也已追加这些字段。

### 2.3 评估入口已支持退化评估模式

已修改：

- [eval.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/eval.py)

当前支持的评估模式：

- `none`
- `drop_hsi`
- `drop_aux`
- `noise`
- `combined`

当前行为为：

- 默认 `none` 时，与原始评估保持一致
- 指定退化模式时，对评估 batch 先做退化再推理
- 输出文件前缀自动带模式名，例如：
  - `test_drop_hsi_metrics_summary.json`
  - `test_combined_classification_report.txt`

## 3. 当前验证情况

### 3.1 工具层验证

已完成：

- `setting/robustness.py` 语法检查
- 张量 shape / dtype / device 检查
- dropout 行为检查
- 统计字段范围检查

本地验证输出示例：

```text
DropHSI: 0.1250 DropAux: 0.0000 HSINoise: 1.0000 AuxNoise: 1.0000 HSIBandDrop: 0.1354
shape/dtype/device/dropout checks ok
```

### 3.2 双分支训练 smoke test

已完成 `robust_train=1` 的最小训练验证，示例结果为：

```text
CleanCE_avg: 2.8502
DegCE_avg: 2.8954
Cons_avg: 0.0592
DropHSI_avg: 0.1000
DropAux_avg: 0.0750
```

这说明：

- clean/degraded 双分支训练已接通
- consistency loss 量级正常
- 退化统计与设定概率基本一致

## 4. 当前分支结论

到目前为止，`feature/robust-training` 已经完成：

1. 退化工具层
2. 双分支训练接入
3. 退化评估接口

当前仍未实现的内容：

- 退化条件下的可视化扩展
- 更复杂的遮挡或错位扰动
- 与 `gate` / `cross-state` 的整合版本

## 5. 下一步建议

本分支后续最合理的动作是：

1. 在 WSL 中分别跑：
   - 普通评估
   - `drop_hsi`
   - `drop_aux`
   - `noise`
   - `combined`
2. 观察不同退化模式下的指标下降幅度
3. 形成一张“鲁棒性实验表”
4. 再决定是否进入最终整合分支
