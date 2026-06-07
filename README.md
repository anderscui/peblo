# Peblo

✨ **Peblo** is a collection of practical, everyday AI-powered utilities — simple, fast, and CLI-friendly.

Peblo is under active development and provides a set of tested tools for text, images, documents, and question-answering.

English | [中文文档](./README_zh.md)

---

## ✨ Features

Peblo currently includes:

- **Translate** — text translation via local or remote LLMs  
- **Summarize** — short summaries and keywords
- **OCR** — extract text from images
- **Image Caption** — describe images  
- **Quote Tools** — quote search & verification  
- **Peek** — preview text, JSON, images, and documents  
- **Q&A** — ask questions about text and files  
- **PDF / Document Tools** — extract text and metadata  
- **Chat** — an interactive LLM-powered CLI chat session, you can chat with docs or urls.  

All features can be used directly from the command line.

---

## 📦 Installation

```bash
pip install peblo
````

---

## 🖥 CLI Usage

Show help:

```bash
peblo --help
```

Show version:

```bash
peblo --version
```

---

## 🔧 Tested Examples

Below are real, tested commands:

```bash
peblo
peblo --help
peblo --version

peblo summary < story.txt
peblo translate -t ja -m ollama:gemma4:12b 'hello, world'

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

## 📄 License

MIT License.

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open an issue or submit a pull request on GitHub.
