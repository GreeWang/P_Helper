# P-Helper

P-Helper 是一个本地论文整理工具：输入 PDF 或 PDF 目录，生成固定格式、带物理页码证据的中文 Markdown 摘要，并建立本地关键词索引用于跨论文问答。

摘要会遍历论文全部分页文本；问答使用 SQLite FTS5 关键词检索。项目不依赖向量数据库、Embedding、QAnything、Git 发布或 Web 服务。

## 环境与安装

- Python `>=3.10,<3.14`，推荐 Python 3.11。
- Marker 首次运行会下载本地 PDF 解析模型，需要较长时间和数 GB 可用磁盘空间。
- Apple Silicon 可使用 CPU/MPS；批量解析始终串行，`--workers` 只控制解析后的摘要并发。

使用 Poetry：

```bash
poetry env use python3.11
poetry install
poetry run p-helper --help
```

也可以安装构建出的 wheel：

```bash
python3.11 -m pip install dist/p_helper-1.1.0-py3-none-any.whl
```

## 模型配置

复制 `.env.example` 为 `.env`，填写 OpenAI-compatible Chat Completions 配置：

```dotenv
P_HELPER_API_KEY=your-key
P_HELPER_API_URL=https://provider.example/v1/chat/completions
P_HELPER_MODEL=your-model
```

`P_HELPER_API_URL` 必须是完整的 Chat Completions 地址。密钥只从环境变量或 `.env` 读取，不会写入日志、清单、索引或会话数据库。

## 生成摘要与索引

```bash
# 单个 PDF
p-helper paper.pdf

# 递归处理目录
p-helper papers/ -o summaries

# 英文摘要、并发摘要
p-helper papers/ --language en --workers 2

# 无条件重做解析、摘要与索引
p-helper papers/ --force
```

每篇解析成功后会自动建立关键词索引。摘要失败不影响正文进入问答库；索引失败不会删除成功摘要，但批次返回退出码 `1`，下次运行会自动重试。

旧输出库可以单独补建索引，不重新生成摘要：

```bash
p-helper index papers/ -o summaries
p-helper index papers/ -o summaries --force
```

索引优先复用 `.phelper/cache` 的分页解析；缓存缺失或解析器版本变化时重新解析原 PDF。

## 论文问答

单轮问答：

```bash
p-helper ask "这些论文使用了哪些数据集？" -o summaries
```

交互式多轮问答：

```bash
p-helper ask -o summaries
```

程序会显示会话 ID。输入 `exit`、`quit` 或发送 EOF 退出。

限制到一篇或多篇论文：

```bash
p-helper ask "主要局限是什么？" -o summaries --paper path/to/paper.pdf
p-helper ask "比较它们的方法" -o summaries --paper abc123 --paper def456
```

`--paper` 接受精确来源路径、别名或唯一 PDF 指纹前缀。默认检索整个输出库；`--top-k` 默认 `8`，允许 `1-20`。

## Web 工作台

启动只监听本机的浏览器界面：

```bash
poetry run p-helper web -o summaries
```

然后打开 `http://127.0.0.1:8765`。可在页面中上传 PDF 或提交本机 PDF 路径、浏览固定摘要、选择论文范围进行问答，以及恢复和删除历史会话。指定其他端口使用 `--port`。

Web 服务仍从 `.env` 或环境变量读取模型配置，密钥不会发送给浏览器。它面向单用户本机使用，默认不监听局域网地址，也不提供账户、远程部署或多用户隔离。

继续会话：

```bash
p-helper ask "它的实验结果呢？" -o summaries --session SESSION_ID
p-helper sessions list -o summaries
p-helper sessions delete SESSION_ID -o summaries
```

会话固定语言、论文范围和 Top-K；恢复时传入冲突参数会报配置错误。默认最多保留最近使用的 100 个会话且数据库不超过 100 MB，任一超限即删除最久未使用会话。

## 证据边界

- PDF 文本、事实、检索块和会话历史都作为不可信数据传入；其中夹带的指令不会被当作系统指令执行。
- 摘要先逐页抽取原文子串事实，再组装结构化摘要；标题、作者、一句话总结、关键词等全部字段也必须引用已抽取事实。
- 每条摘要和问答结论除了校验证据支持，还会校验它是否符合对应摘要字段或回答当前问题。
- 查询会先由模型改写成独立问题并展开为中英文关键词，再通过 SQLite `unicode61` 和 `trigram` 双 FTS 检索。
- 最近 6 轮历史仅用于消解“它、这个方法”等指代；历史回答不作为论文证据，也不会进入最终回答提示词。
- 每条回答必须引用实际召回的证据块，来源和 `PDF 第 N 页` 由本地代码渲染。
- 跨论文比较允许有限综合，但会明确标记为“综合推断”。
- 回答数字必须出现在其引用证据中；未知证据、无引用结论、无依据数字或不相关结论会被拒绝并受约束修复一次。
- 检索不到足够证据时明确拒答，并建议调整关键词或论文范围，不使用模型常识补全。

关键词 RAG 不等价于语义向量检索。查询扩展能改善中文问题检索英文论文，但复杂同义表达仍可能漏召回；v1.1 明确不引入 Embedding。证据链、数字比对和语义裁判能显著降低幻觉风险，但语义裁判仍由同一模型完成，不构成形式化的事实保证。

## 输出结构

```text
summaries/
├── README.md
├── manifest.json
├── papers/<文件名>-<SHA256前12位>.md
├── images/<完整指纹>/figure-01.png
└── .phelper/
    ├── cache/<完整指纹>/
    └── qa/
        ├── index.sqlite3
        └── sessions.sqlite3
```

`manifest.json` 分别记录摘要状态和问答索引状态。已经从输入目录删除的历史论文摘要与索引不会自动删除。

缓存默认最多保留 50 篇且总计不超过 5 GB；任一超限即按 LRU 删除可再生解析缓存，不删除摘要、代表图片、索引、会话、README 或 manifest。

## 重跑与退出码

- 未变化的摘要和索引会跳过。
- PDF、解析器、模型、语言或摘要模板变化时重做摘要。
- 索引只在 PDF、解析器或索引模板变化时重建，模型变化不触发索引重建。
- `--force` 对目标命令执行无条件重做。
- `0`：成功、跳过或正常证据不足。
- `1`：至少一篇处理失败或问答执行失败。
- `2`：参数、配置、论文选择器或索引缺失。
- `130`：用户中断。

## 故障排查

- 返回 HTML 而不是 JSON：检查 `P_HELPER_API_URL` 是否以 `/v1/chat/completions` 结尾。
- Marker 首次启动超时：确认 Hugging Face/Marker 模型站可访问，并适当设置 `HF_HUB_ETAG_TIMEOUT` 和 `HF_HUB_DOWNLOAD_TIMEOUT`。
- 本地 Marker 服务被代理：将 `127.0.0.1,localhost` 加入 `NO_PROXY` 和 `no_proxy`。
- 问答提示索引不存在：先运行摘要命令或 `p-helper index INPUT -o OUTPUT`。
- 关键词没有命中：尝试论文中的英文术语、完整技术名、三字以上中文短语或精确实验数字。

## 测试与构建

```bash
pytest -q
python -m build --wheel
```

真实 Marker 验收测试默认跳过，使用 `pytest -m marker` 单独运行，避免普通 CI 下载模型。配置 `.env` 后可用 `P_HELPER_RUN_MODEL_TESTS=1 pytest -m model` 运行真实模型的提示词对抗验收；该组测试会产生外部 API 调用。
