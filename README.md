# 🐎 2026丙午 灵马送福 - 智能春联生成器

![Project Banner](assets/eg.jpg)

> **融合 RAG (检索增强生成) + DeepSeek V3 + FLUX.1 的 AI 创意贺岁应用**

## 📖 项目简介

**2026 灵马送福** 是一个结合传统春节民俗与硬核 AI 科技的创意项目。

不同于市面上普通的随机生成器，本项目引入了 **RAG (检索增强生成)** 技术，能够“读懂”用户的**身份**（如程序员、医生、科研人员、二次元爱好者等）。通过检索本地的高质量语料库，它能精准击中行业痛点与爽点，生成既工整对仗，又充满“懂行”幽默感的个性化春联。

此外，项目集成了 **DeepSeek** 进行格律控制，并调用 **FLUX.1** 生成高清“门神”年画，完美复刻传统大门的**“竖排对联 + 居中斗方”**视觉体验。

## ✨ 核心功能

* **🎯 懂行 RAG 引擎**：内置 50+ 种职业/身份的知识库，精准匹配行业梗（如程序员的“代码无Bug”、医生的“夜班无急诊”）。
* **🧠 硬核格律创作**：接入 **DeepSeek V3**，通过专家级 Prompt 严格控制平仄格律（仄起平收），拒绝“只有押韵没有灵魂”。
* **🎨 视觉盛宴**：调用 **FLUX.1 (SiliconFlow)** 生成 1024x1024 高清年画，支持赛博朋克、水墨国风、3D卡通等多种风格。
* **🖼️ 沉浸式布局**：前端复刻传统“门神”贴法，实现**竖排对联 + 居中插画 + 顶部横批**的完整视觉效果。
* **✍️ 名字藏头**：智能尝试将用户的名字融入对联之中，打造独一无二的专属祝福。

## 📷 效果展示

以名字为“哈基米”，身份场景为“医疗-医生”为例，四种风格的对联和年画效果如下所示。

|          **传统典雅**          |         **幽默搞怪**         |
|:--------------------------:|:------------------------:|
| ![Vertical](assets/eg.jpg) | ![Funny](assets/eg4.jpg) |

|       **互联网黑话**        |         **赛博朋克**          |
|:----------------------:|:-------------------------:|
| ![RAG](assets/eg2.jpg) | ![Doctor](assets/eg3.jpg) |

## 🛠️ 技术栈

* **Frontend**: Streamlit (Python Web Framework)
* **LLM**: DeepSeek-V3 (via OpenAI SDK)
* **Image Gen**: FLUX.1-schnell (via SiliconFlow API)
* **RAG**: Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Vector Search**: Cosine Similarity (Local Calculation)

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone [https://github.com/littlefish116/Horse-Couplet-2026.git](https://github.com/littlefish116/Horse-Couplet-2026.git)
cd Horse-Couplet-2026
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 API Key
```bash
cp .env.example .env
```
打开 .env 文件，填入你的 API Key：
* **DEEPSEEK_API_KEY**: 获取自 DeepSeek 开放平台
* **SILICONFLOW_API_KEY**: 获取自 SiliconFlow (用于免费绘图)

### 4. 运行应用
```bash
streamlit run app.py
```

浏览器自动打开 http://localhost:8501 即可体验。

* ⚠️ 注意：首次运行时，程序会自动下载 paraphrase-multilingual-MiniLM-L12-v2 语义模型（约 400MB+）。请保持网络畅通，耐心等待几分钟。下载完成后，之后再次运行将瞬间启动。若不支持程序自动下载，可进入 [paraphrase-multilingual-MiniLM-L12-v2 官方链接](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)自行按照其目录结构下载。

## 📂 项目结构

```Plaintext
Horse-Couplet-2026/
├── app.py                  # 主程序入口
├── knowledge_base.json     # RAG 行业语料库
├── requirements.txt        # 项目依赖
├── .env.example            # 环境变量模板
├── README.md               # 项目说明书
└── assets/                 # 演示图片资源
```

## ⚡️ 常见问题 (FAQ)

**Q: 首次运行模型下载太慢怎么办？**

A: 本项目依赖 Hugging Face 模型。如果您在中国大陆地区，建议在终端先设置镜像环境变量，再运行程序：

**Windows (PowerShell):**
```powershell
$env:HF_ENDPOINT = "[https://hf-mirror.com](https://hf-mirror.com)"
streamlit run app.py
```

**Linux / Mac:**
```bash
export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)
streamlit run app.py
```
