# EvidenceMind

EvidenceMind 是一个本地优先的证据检索与问答项目。它把文档位置、Chunk ID、检索结果和回答引用串成可检查的链路：模型可以选择调用本地工具，但工具执行、预算限制和引用校验由程序负责。项目使用纯 Python 实现 Agent 循环，不依赖 LangChain 或 LangGraph。

## 能做什么

- 单 PDF：提取有文本层的页面、切块、建立本地向量索引、检索与页码引用问答。
- 多来源知识库：只读导入带版本号的 DOCX、Obsidian Markdown/文本层 PDF、Zotero 集合中的题录、笔记和本地 PDF；检索结果保留来源、版本与位置。
- 研究 Agent：模型选择计算器或知识库检索工具，程序校验参数、执行工具、回传观察，并限制决策步数和检索次数。
- 回答校验：文档回答必须引用本轮工具实际返回的 Chunk ID；无证据、伪造引用或耗尽步数时明确停止。
- 可选重排：先用多语言 E5 召回候选，再用本地 CrossEncoder 重新排序；默认关闭，不影响原有检索。

```text
用户问题 → 模型选择工具 → 程序限定范围并检索 → 可选候选重排
         → 证据观察回传模型 → 回答 → 程序核对本轮 Chunk ID
```

引用校验只证明 ID 来自本轮证据，**不证明引用段落在语义上支持每一句话**。相似度也不是答案正确率。

## 安装与测试

建议使用 Python 3.12。在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

当前本地回归为 97 项通过。自动化测试主要使用合成文档和模拟模型，不会上传私人资料，也不等同于真实问答准确率。

## 从自己的 PDF 开始

以下命令中的路径和问题都是示例，请换成你有权使用的文件：

```powershell
python -m app.index_cli .\my-document.pdf data\document-index.json `
  --strategy char_overlap --chunk-size 800 --overlap 120

python -m app.retrieval_cli data\document-index.json `
  "这份文档的方法是什么？" --top-k 3 --min-similarity 0.5
```

首次运行会下载公开的嵌入模型；模型已缓存时可加 `--offline`，禁止本地模型下载。只有运行问答命令时，命中的受限证据才会发送给已配置的模型接口。

### 配置在线问答

复制公开模板并只在本地 `.env` 中填写自己的凭据：

```powershell
Copy-Item .env.example .env
```

`config.yaml` 只存模型名、预算和环境变量名，不存密钥。当前模板使用 Step Plan 的 OpenAI 兼容 Chat Completions 接口；如使用其他兼容服务，请自行调整配置。`.env`、索引和原始资料默认不进入 Git。

```powershell
python -m app.answer_cli data\document-index.json `
  "这份文档的方法是什么？" --top-k 3 --min-similarity 0.5 --offline
```

`--offline` 只限制本地模型加载；上面的回答命令**仍会调用在线接口**。只想查看检索结果时使用 `app.retrieval_cli`。

## 多来源知识库与 Agent

下例使用你自己的论文版本目录、Obsidian vault 和 Zotero 数据库。建库会读取本地来源并把原文写入被忽略的索引文件；不会修改源文件，也不会上传整库。

```powershell
python -m app.kb_cli build data\kb-index.json `
  --papers-root "D:\my-paper-versions" `
  --obsidian-root "C:\my-vault" `
  --zotero-db "C:\my-zotero\zotero.sqlite" `
  --current-version 5.1

python -m app.kb_cli ask data\kb-index.json `
  "文档中的关键方法是什么？" --scope paper --retrieval-only --top-k 3
```

多来源索引默认使用 `intfloat/multilingual-e5-small`；Zotero 集合名、排除集合名和论文版本均可通过 `app.kb_cli build --help` 调整。旧版论文只有显式加 `--history` 才进入 Agent 检索。

要让真实模型自主选择工具，并在 Dense Top-20 候选上做可选重排：

```powershell
python -m app.agent_cli data\kb-index.json `
  "请先检索，再根据证据回答这个问题。" `
  --scope paper --top-k 3 --min-similarity 0.5 `
  --rerank --candidate-k 20 --offline
```

首次启用重排时需下载公开的 CrossEncoder 模型；`--offline` 要求模型已经缓存在本机。重排只看第一阶段召回的候选，不能找回候选池之外的证据。Agent 命令会把受长度限制的命中片段送到配置的在线模型；请只对你有权发送的资料使用。要保持原来的 Dense 排序，去掉 `--rerank --candidate-k 20`。

## 评测与边界

`evaluation/` 提供页码级 Dense 检索和多来源知识库评测脚本。真实研究资料、对应索引及私人 10 问标签不随公开仓库发布；运行评测前需用自己的数据准备相应文件。仓库自带的 97 项自动化测试可独立运行。

当前只读取 PDF 文本层，不做 OCR；本地 JSON 索引线性扫描，不是生产级向量数据库；Agent 不保存跨提问会话。没有自动语义支持判定，也没有宣称通用问答准确率。

## 隐私与发布

本仓库只应包含代码、合成测试、公开配置模板和说明。不要提交 `.env`、API 密钥、原始 DOCX/PDF、Obsidian vault、Zotero 数据库及附件、`data/` 索引或模型缓存。发布前请检查 `git status` 和 `git diff --cached --name-only`。当前尚未附许可证；如果要明确授权他人复用，请先选择并添加 `LICENSE`。
