# P-Helper

P-Helper 是面向研究人员、学生和论文阅读者的本地论文整理工具。它读取单篇 PDF 或 PDF 目录，生成固定格式、带物理页码证据的 Markdown 摘要，并建立本地关键词索引，用于单篇或跨论文问答。

项目的重点不是生成一段无法追溯的概述，而是让摘要和回答中的结论能够回到 PDF 原文页。摘要会遍历全部分页文本；问答使用 SQLite FTS5 检索，不需要向量数据库、Embedding 服务或独立 RAG 平台。PDF、摘要、索引和会话保存在本机，但摘要和问答仍需要调用一个 OpenAI-compatible Chat Completions API。

## 当前状态

- 当前版本：`1.1.0`。
- 当前定位：本地、单用户的 CLI 和浏览器工作台；适合个人论文库整理、证据核查和有限的跨论文比较。
- 自动化测试覆盖本地管线、索引、问答约束、HTTP 接口和 Web 行为。真实 Marker PDF 和真实模型测试是单独启用的外部验收，不包含在普通测试中。
- 这不是面向公网或多租户的生产服务。项目没有账户、远程访问控制、服务级健康检查、监控告警或容量保证。

与常见的“PDF 对话”或向量 RAG 工具相比，P-Helper 的关键特点是：固定摘要结构、物理页码引用、本地渲染引用、数字证据校验和明确拒答；代价是关键词检索不能覆盖所有同义表达，也不提供语义向量召回。

## 功能与边界

已支持：

- 处理单个 PDF，或递归处理目录内的 PDF。
- 生成中文或英文 Markdown 摘要，并提取代表图片。
- 基于 PDF 指纹和处理签名跳过未变化内容，支持强制重跑。
- 为正文建立 SQLite FTS5 本地索引；已有摘要库可以单独补建索引。
- 单轮或多轮问答，可限制论文范围，并可列出、恢复和删除本地会话。
- 仅监听本机地址的 Web 工作台，可上传 PDF、提交本机路径、浏览摘要和执行问答。

暂不支持：

- Windows、向量检索、Embedding、OCR/解析模型的远程托管配置。
- 账户体系、多用户隔离、局域网或公网部署、横向扩容。
- 自动删除已经从输入目录移除的历史论文。
- 对模型输出正确性的形式化保证。

已知限制：

- Marker 的解析质量决定后续证据质量；扫描件、复杂公式和异常版式可能解析不完整。
- 中文问题会扩展为中英文关键词，但复杂同义表达仍可能漏召回。
- 语义裁判由同一个外部模型完成，只能降低而不能消除错误。
- 批量 PDF 解析始终串行；`--workers` 只并发执行解析后的摘要步骤。
- 同一输出目录的摘要和补索引操作使用进程锁串行执行。

## 前置条件

- 操作系统：macOS 或 Linux。代码依赖 POSIX `fcntl` 文件锁，Windows 当前不支持。
- Python：`>=3.10,<3.14`，推荐 Python 3.11。
- SQLite：Python 自带的 SQLite 必须启用 FTS5，并支持 `unicode61` 和 `trigram` tokenizer。
- 外部服务：一个 OpenAI-compatible Chat Completions API。
- 网络：首次解析时需要访问 Hugging Face/Marker 模型源；摘要和问答时需要访问模型 API。模型下载完成后，索引和本地浏览不需要外部数据库。
- 权限：输入 PDF 的读取权限、输出目录的创建和写入权限，以及访问上述服务的网络权限。
- 资源：Marker 首次运行会下载模型，需要较长时间和数 GB 可用磁盘空间；默认解析缓存上限为 50 篇或 5 GB。实际内存和时间取决于 PDF 数量、页数及硬件。

Apple Silicon 可以使用 CPU/MPS。仓库没有声明 CUDA、特定 GPU 或容器为必需条件，也没有提供 Docker 镜像。

## 快速开始

以下是步骤最少的源码安装路径，不要求预先安装 Poetry：

```bash
git clone https://github.com/GreeWang/P_Helper.git
cd P_Helper
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

编辑 `.env`，填入自己的模型地址、模型名和密钥。不要把真实密钥提交到 Git、写入命令参数或粘贴到问题报告中。然后处理一篇 PDF：

```bash
p-helper /absolute/path/to/paper.pdf -o summaries
```

首次运行会下载 Marker 模型并调用配置的模型 API。命令退出码为 `0` 后，应能看到：

```text
summaries/
├── README.md
├── manifest.json
├── papers/<文件名>-<SHA256前12位>.md
├── images/<完整指纹>/figure-01.png
└── .phelper/qa/index.sqlite3
```

`manifest.json` 中对应论文的摘要和索引状态应为成功，生成的 Markdown 中应包含 `PDF 第 N 页` 证据引用。代表图片只在解析结果包含合适图片时生成，所以没有 `figure-01.png` 不一定表示失败。

随后可执行一次核心问答：

```bash
p-helper ask "这篇论文使用了什么数据集？" -o summaries
```

成功时会先输出 `Session: <会话 ID>`，然后给出带来源和 `PDF 第 N 页` 引用的回答；证据不足时会明确拒答，这也是正常结果。

如果已经安装 Poetry，也可以使用锁定依赖环境：

```bash
poetry env use python3.11
poetry install
poetry run p-helper --help
```

Poetry 不是运行时依赖，需要使用者自行安装。构建 wheel 后也可以执行 `python -m pip install dist/p_helper-1.1.0-py3-none-any.whl`。

## 配置

复制仓库中的 [`.env.example`](.env.example) 为 `.env`。运行入口会读取当前目录的 `.env`；同名进程环境变量优先于文件内容。

| 名称 | 必填 | 用途 | 默认值 |
| --- | --- | --- | --- |
| `P_HELPER_API_KEY` | 是 | OpenAI-compatible API 密钥 | 无 |
| `P_HELPER_API_URL` | 是 | 完整的 Chat Completions 地址 | 无 |
| `P_HELPER_MODEL` | 是 | 请求使用的模型名 | 无 |
| `P_HELPER_OUTPUT` | 否 | 默认输出目录，可被 `-o/--output` 覆盖 | `summaries` |

示例只使用占位值：

```dotenv
P_HELPER_API_KEY=replace-with-your-key
P_HELPER_API_URL=https://provider.example/v1/chat/completions
P_HELPER_MODEL=replace-with-your-model
P_HELPER_OUTPUT=summaries
```

摘要命令还支持 `--language zh|en`、`--workers N`、`--force`、`--cache-max-papers N`、`--cache-max-size 5GB` 和 `--verbose`。问答支持 `--paper`、`--session`、`--top-k 1..20` 和 `--language`。使用 `p-helper --help` 或对应子命令的 `--help` 查看当前参数，避免在 README 中重复容易过期的完整参数表。

## 使用示例

生成摘要并自动建索引：

```bash
# 单个 PDF
p-helper paper.pdf

# 递归处理目录
p-helper papers/ -o summaries

# 英文摘要，并发执行解析后的摘要步骤
p-helper papers/ --language en --workers 2

# 无条件重做解析、摘要与索引
p-helper papers/ --force
```

摘要失败不影响已解析正文进入问答库；索引失败不会删除成功摘要，但批次返回退出码 `1`，下次运行会自动重试。旧输出库可以单独补建索引：

```bash
p-helper index papers/ -o summaries
p-helper index papers/ -o summaries --force
```

索引优先复用 `.phelper/cache` 的分页解析；缓存缺失或解析器版本变化时重新解析原 PDF。

单轮、交互式和限定论文范围的问答：

```bash
p-helper ask "这些论文使用了哪些数据集？" -o summaries
p-helper ask -o summaries
p-helper ask "主要局限是什么？" -o summaries --paper path/to/paper.pdf
p-helper ask "比较它们的方法" -o summaries --paper abc123 --paper def456
```

交互模式显示会话 ID；输入 `exit`、`quit` 或发送 EOF 退出。`--paper` 接受精确来源路径、别名或唯一 PDF 指纹前缀。默认检索整个输出库；`--top-k` 默认 `8`，允许 `1-20`。

恢复和管理会话：

```bash
p-helper ask "它的实验结果呢？" -o summaries --session SESSION_ID
p-helper sessions list -o summaries
p-helper sessions delete SESSION_ID -o summaries
```

会话固定语言、论文范围和 Top-K；恢复时传入冲突参数会报配置错误。默认最多保留最近使用的 100 个会话且数据库不超过 100 MB，任一超限即删除最久未使用的会话。

### Web 工作台使用教程

1. 确认 `.env` 已配置，然后启动本机工作台：

```bash
p-helper web -o summaries
```

2. 打开 `http://127.0.0.1:8765`。左下角应显示“模型已配置”；如果显示“模型未配置”，停止服务，补全三个必填模型变量后重新启动。使用 `--port` 可以更改端口，例如 `p-helper web -o summaries --port 8766`。

3. 在“论文库”点击右上角“导入 PDF”，选择一种导入方式：

   - “上传 PDF”支持选择或拖入多个 PDF，单批最多 100 个文件、总大小不超过 1 GB。
   - “本地路径”接受这台机器上的单个 PDF 或目录路径；运行 P-Helper 的用户必须能够读取该路径。

   按需选择摘要语言、摘要并发数和“强制重新处理”，然后点击“开始处理”。摘要并发范围为 `1-16`；它不改变 Marker 串行解析 PDF 的行为。

4. 返回“论文库”查看任务状态。“排队中”和“处理中”表示任务尚未结束；失败任务会显示错误详情。成功后页面提示“处理任务已完成”，论文条目应显示“摘要就绪”和“索引就绪”。只有“摘要就绪”时可以浏览摘要，只有“索引就绪”时可以参与问答。点击刷新按钮可以重新读取任务和论文状态。

5. 在论文列表中点击一篇“摘要就绪”的论文，右侧会显示生成的 Markdown 摘要和代表图片。可以用顶部搜索框按来源路径筛选论文。如果摘要或索引显示失败，先查看任务详情和终端日志，再按需要使用“强制重新处理”。

6. 切换到“论文问答”。默认范围是“整个论文库”；点击“范围”可以只选择部分“索引就绪”的论文，并通过“召回证据”滑块设置 Top-K。输入问题后点击发送，或使用页面提供的示例问题。成功时回答会附论文来源和 `PDF 第 N 页`；证据不足时的明确拒答是正常结果。第一次回答会创建会话，此后范围和 Top-K 被锁定；需要更换范围时点击“新建会话”。

7. 切换到“历史会话”可以恢复已有对话或删除不再需要的会话。恢复后继续提问会沿用该会话原有的论文范围、语言和 Top-K。删除操作只删除问答会话，不删除论文、摘要或索引。

页面提示“页面令牌无效”时刷新浏览器后重试；页面持续显示“处理中”时查看启动终端的日志；任务失败时保留错误信息并参考“故障排查”。Web 服务只绑定 `127.0.0.1`，页面令牌用于防止本机跨站请求，不是用户身份认证，不应通过端口转发将它直接暴露到局域网或公网。

## 证据边界

- PDF 文本、事实、检索块和会话历史都作为不可信数据传入；其中夹带的指令不会被当作系统指令执行。
- 摘要先逐页抽取原文子串事实，再组装结构化摘要；标题、作者、一句话总结、关键词等字段也必须引用已抽取事实。
- 每条摘要和问答结论除了校验证据支持，还会校验它是否符合对应摘要字段或回答当前问题。
- 查询先由模型改写成独立问题并展开为中英文关键词，再通过 SQLite `unicode61` 和 `trigram` 双 FTS 检索。
- 最近 6 轮历史只用于消解“它、这个方法”等指代；历史回答不作为论文证据，也不会进入最终回答提示词。
- 每条回答必须引用实际召回的证据块，来源和物理页码由本地代码渲染。
- 跨论文比较允许有限综合，但会明确标记为“综合推断”。
- 回答数字必须出现在引用证据中；未知证据、无引用结论、无依据数字或不相关结论会被拒绝，并进行一次受约束修复。
- 检索不到足够证据时明确拒答，并建议调整关键词或论文范围，不使用模型常识补全。

这些措施降低幻觉风险，但语义裁判仍由外部模型完成，不构成事实保证。关键词 RAG 也不等价于语义向量检索。

## 架构与关键目录

核心流程如下：

```text
CLI / 本机 Web
      │
      ├─ PDF discovery → Marker parsing → page/block cache
      │                                  │
      │                                  ├─ evidence extraction → validation → Markdown summary
      │                                  └─ chunking → SQLite FTS5 index
      │
      └─ question rewrite → keyword retrieval → evidence validation → cited answer
```

- `frame/main.py`：CLI 参数、子命令分发和退出码。
- `frame/pipeline.py`、`parser.py`、`summarize.py`：PDF 处理和摘要管线。
- `frame/indexing.py`、`qa_index.py`、`retrieval.py`、`qa.py`：索引、检索和证据问答。
- `frame/web.py`、`frame/web_static/`：仅本机使用的 Web 工作台。
- `frame/manifest.py`、`cache.py`、`sessions.py`：处理状态、可再生缓存和会话持久化。
- `tests/`：本地单元、管线、HTTP/Web 和可选外部验收测试。

运行产物默认位于 `summaries/`。`manifest.json` 分别记录摘要和问答索引状态；`.phelper/qa/` 中的 SQLite 数据库保存索引和会话。

## 重跑与退出码

- 未变化的摘要和索引会跳过。
- PDF、解析器、模型、语言或摘要模板变化时重做摘要。
- 索引只在 PDF、解析器或索引模板变化时重建，模型变化不触发索引重建。
- `--force` 对目标命令执行无条件重做。
- `0`：成功、跳过或正常证据不足。
- `1`：至少一篇处理失败或问答执行失败。
- `2`：参数、配置、论文选择器或索引缺失。
- `130`：用户中断；已经完成的论文仍会保留。

## 测试与验证

开发依赖可通过 Poetry 安装，或在已激活的虚拟环境中安装：

```bash
python -m pip install -e . pytest build httpx
```

本地自动化测试和构建：

```bash
pytest -q
python -m build --wheel
```

仓库当前没有配置独立的 lint、格式化或静态类型检查命令，因此 README 不宣称这些检查已经执行。普通 `pytest -q` 使用模拟解析器和模型，不需要真实密钥，但真实 Marker 用例会因未提供 PDF fixture 而跳过，真实模型组也默认跳过。

真实 Marker 验收：

```bash
P_HELPER_MARKER_DIGITAL_FIXTURE=/absolute/path/to/digital.pdf \
P_HELPER_MARKER_SCANNED_FIXTURE=/absolute/path/to/scanned.pdf \
pytest -m marker
```

真实模型验收会产生外部 API 调用和费用，只在确认 `.env` 指向测试允许的模型后运行：

```bash
P_HELPER_RUN_MODEL_TESTS=1 pytest -m model
```

本地测试通过只证明模拟依赖下的行为；真实 Marker 通过只证明指定 fixture 的解析；真实模型通过只证明当次模型端点的约束表现。它们都不等价于公网部署或生产可用。

当前文档验证基线（2026-08-13，macOS、Python 3.11）：源码可编辑安装成功；根命令及 `index`、`ask`、`sessions`、`web` 的 `--help` 均返回 `0`；SQLite FTS5 的 `unicode61` 和 `trigram` 可用；普通测试结果为 `118 passed, 7 skipped`；wheel 构建成功；本机 `/api/bootstrap` 返回预期 JSON。本次没有调用真实模型，也没有运行真实 Marker fixture，因此快速开始中的真实 PDF 结果仍取决于使用者的 PDF、网络和模型端点。

## 部署与运维

推荐方式是每位用户在自己的 macOS/Linux 工作站中使用虚拟环境运行 CLI 或本机 Web 工作台。当前没有受支持的 Docker、systemd、云部署或远程多用户方案。

- 健康确认：运行 `p-helper --help` 验证安装；Web 启动后访问 `http://127.0.0.1:8765/api/bootstrap`，能够返回 JSON 说明本机进程可响应。该接口是启动检查，不是带依赖探测的生产健康检查。
- 日志：CLI 和 Web 日志输出到标准错误；使用 `--verbose` 查看调试日志。项目没有日志轮转、指标或告警。
- 持久化：备份整个输出目录，至少包括 `manifest.json`、`papers/`、`images/` 和 `.phelper/qa/`。`.phelper/cache/` 是可再生解析缓存，可以不备份。
- SQLite 备份：`index.sqlite3` 和 `sessions.sqlite3` 使用 WAL。先停止 P-Helper 进程再复制输出目录，避免得到不一致的数据库副本。
- 升级：停止运行中的进程，备份输出目录和当前环境，拉取目标版本后重新安装。首次在副本上运行并检查 `manifest.json` 和核心问答，再替换原环境。
- 回滚：恢复旧代码/虚拟环境和升级前的完整输出目录备份。仓库当前没有独立数据库迁移或自动回滚命令，因此不要只降级代码后继续写入已被新版本修改的输出库。

生产或远程化之前仍需另行设计身份认证、TLS、网络隔离、密钥管理、健康探测、日志与指标采集、资源限额、并发策略、备份恢复演练和容量测试；这些不属于当前支持范围。

## 故障排查

| 现象 | 常见原因 | 处理方法 |
| --- | --- | --- |
| `poetry: command not found` | 机器未安装 Poetry | 使用“快速开始”的标准 `venv + pip` 路径，或先自行安装 Poetry |
| `Missing required model configuration` | 三个模型配置有缺失 | 检查 `.env` 是否位于运行目录，且 `P_HELPER_API_KEY`、`P_HELPER_API_URL`、`P_HELPER_MODEL` 均非空 |
| 返回 HTML 而不是 JSON | API 地址不是 Chat Completions 端点 | 确认 `P_HELPER_API_URL` 是完整地址，通常以 `/v1/chat/completions` 结尾 |
| Marker 首次启动超时 | 模型源不可达或下载慢 | 检查 Hugging Face/Marker 网络；按需设置 `HF_HUB_ETAG_TIMEOUT` 和 `HF_HUB_DOWNLOAD_TIMEOUT` |
| 本地 Marker 请求经过代理 | 代理截获回环地址 | 将 `127.0.0.1,localhost` 加入 `NO_PROXY` 和 `no_proxy` |
| `Address already in use` | Web 端口被占用 | 使用 `p-helper web --port 8766`，或停止占用原端口的进程 |
| `Permission denied` | 无权读取 PDF 或写入输出目录 | 检查输入文件和输出目录权限，改用当前用户可读写的路径 |
| SQLite `locked` 或命令长时间等待 | 另一个进程正在使用同一输出库 | 等待该进程结束；不要让多个 Web/CLI 写入者共享同一输出目录 |
| 问答提示索引不存在 | 尚未成功建索引 | 先运行摘要命令，或执行 `p-helper index INPUT -o OUTPUT` |
| 关键词没有命中 | 查询词与论文措辞差异较大 | 尝试英文术语、完整技术名、三字以上中文短语或精确实验数字 |

先用 `--verbose` 重现问题，并检查命令退出码、终端日志和 `manifest.json` 对应论文的状态。提交问题时请附最小复现命令、操作系统、Python/P-Helper 版本、退出码和已脱敏日志，不要附 `.env`、API 密钥或包含敏感论文内容的数据库。问题入口：[GitHub Issues](https://github.com/GreeWang/P_Helper/issues)。

## 协作与维护

- 反馈问题或提出功能建议：[GitHub Issues](https://github.com/GreeWang/P_Helper/issues)。
- 贡献流程：从 `master` 创建分支，保持改动聚焦，为行为变化补充测试，运行普通测试和 wheel 构建，再提交 Pull Request。提交信息应简洁说明行为变化；不要提交 `.env`、模型密钥、受限论文或生成的本地数据库。
- 版本：包版本以 `pyproject.toml` 为准。项目当前没有单独的 CHANGELOG 或正式稳定性承诺；发布前应在 PR 中说明兼容性和数据格式变化。
- 维护者：`pyproject.toml` 中登记的项目作者为 GreeWang；日常反馈统一通过 Issue 跟踪。
- License：本项目采用 [GNU GPL v3 或更高版本](LICENSE)。第三方依赖和随包静态资源适用各自许可证。
