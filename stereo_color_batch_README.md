# 双目图像批量色差校正

脚本递归扫描左右目目录，按相同的相对路径和文件名配对，并以左目图像作为参考校正右目图像。

当前版本采用“先备份、后覆盖”：

1. 将所有待处理的右目原图完整复制到独立备份目录；
2. 逐文件检查大小并进行字节级比较；
3. 生成包含 SHA-256 的 `backup_manifest.csv`；
4. 只有全部备份验证成功后，才开始矫正；
5. 矫正结果通过临时文件原子替换右目原图；
6. 左目目录始终只读。

## 依赖

```powershell
python -m pip install numpy opencv-python
```

建议使用 Python 3.9 或更高版本、OpenCV 4.5 或更高版本。

## 目录示例

```text
dataset/
├─ left/
│  └─ scene01/
│     ├─ 000001.png
│     └─ 000002.png
└─ right/
   └─ scene01/
      ├─ 000001.png
      └─ 000002.png
```

## 推荐命令

```powershell
python .\stereo_color_batch.py `
  --left-dir 'D:\dataset\left' `
  --right-dir 'D:\dataset\right'
```

脚本会自动在 `right` 的同级目录创建带时间戳的备份，例如：

```text
D:\dataset\right_original_backup_20260723_151500
```

备份目录保存：

- 所有被覆盖前的右目原图，目录结构不变；
- `backup_manifest.csv`：原文件路径、备份路径、大小和 SHA-256；
- `stereo_color_model.npz`：共享校正模型；
- `stereo_color_report.csv`：匹配数量、视差、亮度参数、校正误差和失败原因。

如果希望指定备份位置：

```powershell
python .\stereo_color_batch.py `
  --left-dir 'D:\dataset\left' `
  --right-dir 'D:\dataset\right' `
  --backup-dir 'E:\backup\right_original_20260723'
```

`--backup-dir` 必须是尚不存在的新目录，以防误用旧备份或覆盖已有文件。

## 默认校正模式

默认 `hybrid` 模式：

- 每对图片根据可靠对应点单独匹配亮度；
- 整批图片共享一个 3×3 颜色矩阵和一个平滑空间色偏场；
- 对固定双目镜头更稳定，视频序列不容易出现逐帧颜色闪烁。

整批使用同一个模型、速度更快：

```powershell
python .\stereo_color_batch.py `
  --left-dir 'D:\dataset\left' `
  --right-dir 'D:\dataset\right' `
  --mode shared
```

每对图片独立拟合完整模型：

```powershell
python .\stereo_color_batch.py `
  --left-dir 'D:\dataset\left' `
  --right-dir 'D:\dataset\right' `
  --mode per-pair
```

逐对模式适合相机设置或现场光照变化很大的数据，但相邻帧可能出现轻微颜色跳动。

## 常用参数

```text
--backup-dir PATH             指定新的原图备份目录
--max-calibration-pairs 30   共享模型最多抽取多少对图片
--min-colour-pairs 80        单对图片所需的最少可靠对应点
--degree 2                   空间色偏场阶数；2 更稳健，3 更灵活
--failure skip               单对失败时保留该右目原图；也可使用 error
--no-recursive               不扫描子目录
--linear-input               输入为线性 RGB，而不是普通 sRGB/JPEG/PNG
```

## 恢复原图

备份目录与右目目录具有相同的图片相对路径。需要恢复时，从备份目录将图片复制回右目目录即可。不要把 `backup_manifest.csv`、模型或报告复制进右目目录。

输出保持右目图像的尺寸、位深和 Alpha 通道。OpenCV 重编码后的图片不会保留 EXIF 等文件元数据，但备份中的原始文件是字节级副本，保留全部原始数据和元数据。
