# Peblo

✨ **Peblo** 是一组实用、简洁、面向日常使用的 AI 工具集合 —— 轻量、快速，并且非常适合命令行环境。

Peblo 正在积极开发中，已经提供了一组经过测试的工具，覆盖文本、图片、文档处理以及问答功能。

中文 | [English](./README.md)

---

## ✨ 功能特性

Peblo 目前包含：

* **Translate** — 使用本地或远程 LLM 进行文本翻译
* **Summarize** — 生成短摘要与关键词
* **OCR** — 从图片中提取文字
* **Image Caption** — 自动生成图片描述
* **Quote Tools** — 引文搜索与验证
* **Peek** — 预览文本、JSON、图片与文档
* **Q&A** — 针对文本或文件进行问答
* **PDF / Document Tools** — 文本提取与文档元数据读取
* **Chat** — 基于 LLM 的交互式命令行聊天，可对文档或 URL 进行对话

以上所有功能均可直接在命令行中使用。

---

## 📦 安装

```bash
pip install peblo
```

---

## 🖥 CLI 用法

查看帮助：

```bash
peblo --help
```

查看版本：

```bash
peblo --version
```

---

## 🔧 已测试的示例命令

以下命令均已在真实环境中测试通过：

```bash
peblo
peblo --help
peblo --version

peblo summary < story.txt
peblo translate -t ja -m ollama:qwen3:4b-instruct 'hello, world'

peblo ocr image.png

peblo quote verify --author 鲁迅 '其实地上本没有路，走的人多了，也便成了路'
peblo quote search '世上本没有路；走的人多了，也就慢慢有了路'

peblo peek main.py
peblo peek all-models.json
peblo peek '世上本没有路；走的人多了，也就慢慢有了路'

peblo qa --json '世上本没有路；走的人多了，也就慢慢有了路，这是谁说的？'
peblo qa --json 'List the functions defined in this file.' main.py
peblo qa --json 'How to fix this error?' 'ERROR: FileNotFoundError: config.yaml not found'

peblo --debug --log-file run.log qa 'How to fix this error?' 'ERROR: FileNotFoundError: config.yaml not found'

peblo --debug pdftext 'Qwen3-VL Technical Report (2025.11).pdf'
peblo --debug docmeta 'Qwen3-VL Technical Report (2025.11).pdf'

peblo --debug chat
peblo --debug chat main.py
peblo --debug chat --session peblo_cli main.py
peblo --debug chat https://arxiv.org/abs/2512.02556
```

---

## 📄 许可证

MIT License。

---

## 🤝 参与贡献

欢迎贡献！
欢迎在 GitHub 上提 issue 或提交 Pull Request。
