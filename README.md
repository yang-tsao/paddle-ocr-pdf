# paddle-ocr-pdf
A python script which employs PaddleOCR to add a hidden text layer to picture pdfs.

利用PaddleOCR对图像PDF进行OCR，相较于OCRmyPDF，不会在中文间乱加空格。

## 安装方法

### **安装`paddlepaddle`**

具体参见<https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html>，尤其是拥有GPU的用户。

```bash
python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### **安装`paddleocr`**

```bash
pip install paddleocr
```

### **安装`PyMuPDF`**

```bash
pip install PyMuPDF
```

## 运行


### 基本用法

三个脚本（`pocr.py`、`pocr-pixmap.py` 和 `pocr-inplace.py`）均可为图像型 PDF 添加隐藏但可复制的文本层，以下是通用命令格式：

```bash
python <script_name> input.pdf output.pdf
```

将 `script_name` 替换为具体脚本名，`input.pdf` 为输入的图像型 PDF 文件，`output.pdf` 为处理后的输出文件。

### 命令行参数说明

- **`-p` 或 `--pure`**：生成只包含文本层的纯文本 PDF 文件，文件名以 `-pure.pdf` 结尾。

```bash
python script_name.py -p input.pdf output.pdf
```

- **`-c` 或 `--cv`**：处理过程中显示提取的图像。

```bash
python script_name.py -c input.pdf output.pdf
```

- **`-n` 或 `--no-ocr`**（仅 `pocr-pixmap.py` 和 `pocr-inplace.py` 支持）：跳过 OCR 处理。

```bash
python script_name.py -n input.pdf output.pdf
```

- **`-l` 或 `--lang`**：指定 OCR 识别语言，默认为 `ch`。

```bash
python script_name.py -l eng input.pdf output.pdf
```
