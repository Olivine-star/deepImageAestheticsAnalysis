# 项目推理说明（图片美学评分）

本文给出在本仓库中使用 `python/caffe_prediction.py` 对图片做美学评分推理的最短步骤。

## 1. 准备环境

1. 安装 **Caffe（带 Python 接口）**，并确认能在 Python 中 `import caffe`。
2. 安装依赖：

```bash
pip install numpy opencv-python
```

说明：`python/caffe_prediction.py` 是 Python 2 风格代码（`print` 语法），建议用 Python 2 运行，或自行改成 Python 3 语法后再运行。

## 2. 准备模型文件

从仓库根目录 `README.md` 提供的 Google Drive 链接下载 demo 模型（`initModel.zip`），解压后至少需要：

- `initModel.prototxt`
- `initModel.caffemodel`

本仓库已自带均值文件：

- `mean_AADB_regression_warp256.binaryproto`

建议放置方式（示例）：

```text
<repo>/models/initModel.prototxt
<repo>/models/initModel.caffemodel
<repo>/mean_AADB_regression_warp256.binaryproto
```

## 3. 修改推理脚本路径

编辑 `python/caffe_prediction.py` 顶部常量：

- `CAFFE_ROOT`：你的 caffe 根目录（里面有 `python/` 子目录）
- `AVA_ROOT`：图片目录（脚本会读取该目录下 `*.jpg`）
- `IMAGE_MEAN`：均值文件路径
- `DEPLOY`：`initModel.prototxt` 路径
- `MODEL_FILE`：`initModel.caffemodel` 路径
- `IMAGE_FILE`：待推理图片通配符（如 `/path/to/images/*.jpg`）

可参考（按你本机实际路径替换）：

```python
CAFFE_ROOT = '/opt/caffe/caffe-master/'
AVA_ROOT = '/Users/xxx/code/deepImageAestheticsAnalysis/demoRatingImages/'
IMAGE_MEAN = '/Users/xxx/code/deepImageAestheticsAnalysis/mean_AADB_regression_warp256.binaryproto'
DEPLOY = '/Users/xxx/code/deepImageAestheticsAnalysis/models/initModel.prototxt'
MODEL_FILE = '/Users/xxx/code/deepImageAestheticsAnalysis/models/initModel.caffemodel'
IMAGE_FILE = AVA_ROOT + '*.jpg'
```

## 4. 运行推理

在仓库根目录执行：

```bash
python python/caffe_prediction.py
```

## 5. 查看结果

脚本会逐张输出：

- 每张图的 `fc11_score`（美学评分）
- 最后输出分数最高图片：`Best image, based only on fc11_score = ...`

另外，`out` 中也包含属性分支分数（如 `fc9_ColorHarmony`、`fc9_RuleOfThirds` 等）。
