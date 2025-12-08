# Digital Governance Topic Modeling Pipeline

本仓库包含对多个城市政府治理文本进行处理、转换、主题建模（BERTopic）的完整工作流。模型文件体积较大未上传，但通过本仓库的代码、数据结构与环境文件，可完全复现分析流程。

---

## 📁 项目结构

```
project/
├── bertopic_pipeline.py        # 主运行脚本：清洗文本 + 加载模型 + 主题建模
├── bertopic_run.log            # 模型运行日志
├── environment.yml             # Conda 环境（可复现）
├── result.csv                  # 第一次主题建模输出
├── result2.csv                 # 第二次主题建模输出
│
├── data/
│   ├── docx2txt.py             # DOCX → TXT 转换脚本
│   ├── docxV/                  # 原始 docx 文档（未上传）
│   ├── txt_doc_level/          # 每个城市的 txt 文本
│   └── ref.txt / ref2 / ...    # 中间文件
│
├── models/                     # 需手动放置的 embedding 模型（如 text2vec）
└── output/                     # BERTopic 输出（未上传）
```

---

## 🚀 工作流概览（Workflow Overview）

本项目的主题建模流程包含四个阶段：

### 1. 文档预处理：DOCX → TXT

```bash
python data/docx2txt.py
```

将所有城市 docx 文档转换为 txt，存入：

```
data/txt_doc_level/
```

---

### 2. 文本加载与清洗（在 bertopic_pipeline.py 内）

* 遍历 txt 文档
* 清洗文本（去空行、特殊符号等）
* 汇总为 corpus 列表
* Logging 记录文本条数

---

### 3. 加载本地 embedding 模型

将下载好的中文句向量模型（如 text2vec-base-chinese）放入：

```
models/
```

示例：

```
models/text2vec-base-chinese/
```

BERTopic 支持本地 embedding 模型，推荐方式参考官方文档（https://github.com/MaartenGr/BERTopic）。

#### **1. 下载 Sentence-Transformers 模型（推荐）**

以中文模型为例：

```bash
huggingface-cli download --repo-type model BAAI/bge-base-zh-v1.5 --local-dir models/bge-base-zh-v1.5
```

或 text2vec:

```bash
huggingface-cli download --repo-type model GanymedeNil/text2vec-large-chinese --local-dir models/text2vec-large-chinese
```

#### **2. BERTopic 官方推荐模型类型**

来自官方仓库：[https://github.com/MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)

支持：

* Sentence-Transformers 模型
* 🤗 Transformers 任意 encoder 模型
* 使用 `EmbeddingModel` 自定义加载路径

#### **3. 将模型放入正确目录**

模型目录结构：

```
project/models/
    └── text2vec-large-chinese/
        ├── config.json
        ├── model.safetensors / pytorch_model.bin
        ├── tokenizer.json
        └── ...
```

#### **4. bertopic_pipeline.py 如何加载模型**

在脚本中类似：

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("models/text2vec-large-chinese")
topic_model = BERTopic(embedding_model=embedding_model)
```

---

### 4. 运行 BERTopic 主题建模

运行主脚本：

```bash
python bertopic_pipeline.py
```

输出包括：

* `result.csv`
* `result2.csv`
* 可视化/模型（若开启保存）

---

## 🧪 可复现步骤（从零开始）

### 1. Clone 仓库

```bash
git clone https://github.com/JiangGufan/digital-governance.git
cd digital-governance/project
```

### 2. 创建 conda 环境

```bash
conda env create -f environment.yml
conda activate topiccity
```

### 3. 下载中文 embedding 模型（手动）



将模型放入：

```
project/models/
```

推荐模型：

* text2vec-base-chinese
* sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

### 4. 运行主题建模

```bash
python bertopic_pipeline.py
```

---

## 📊 输入输出格式

### 输入

```
data/docxV/*.docx
```

经转换：

```
data/txt_doc_level/*.txt
```

### 输出

```
result.csv      # 主题结果（版本 1）
result2.csv     # 主题结果（版本 2）
output/*        # 可选可视化与模型
```

---

## 📦 环境依赖

使用 Conda 自动创建：

```bash
conda env create -f environment.yml
conda activate topiccity
```

模型文件未上传，需要手动放置。

---

## 📜 License

本项目采用 **MIT License**，允许商业使用、修改、分发。
