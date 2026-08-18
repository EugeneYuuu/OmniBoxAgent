# OmniBoxAgent · 私人收藏 AI 问答大脑

**项目网站：[OmniBoxAgent](<https://omnibox-parse.online/OmniBoxAgent/>)**

**OmniBoxAgent** 为 OmniBox 私人收藏管理产品提供 AI 问答能力：将各平台收藏的内容转化为统一的知识库，支持自然语言提问，回答均基于用户自有内容生成并附原文出处；复杂问题自动拆解执行，用户偏好持续沉淀，问答体验随使用不断优化。

技术上，服务以 **FastAPI + LangGraph** 编排 QA 与 Creative 两条流式管线，通过 **ChromaDB 混合检索 + 多轮 LLM 判定**（查询理解、复杂度分类、澄清、质量门控、反思）输出"边算边输出"（NDJSON 流式）回答，并支持 **SKILL 渐进式加载**、**MCP 工具接入**与**会话记忆（Session Tree + Compaction）**。

---

## 业务场景（Business Scenario）

OmniBoxAgent 解决一个核心问题：用户跨平台收藏的内容是碎片化的非结构化数据，无法直接被大模型检索。它把这些收藏变成可问答的知识库，并在问答中沉淀记忆——越用越懂你。

- **多平台收藏聚合**：各主流内容平台的笔记、视频统一摄取。每条内容从多维度建立语义索引（正文、媒体文字、评论互动），形成统一的可查询知识库。
- **基于知识库的智能问答**：用户用自然语言提问，系统通过语义检索找到相关收藏，再由大模型生成 grounded 回答，所有引用都可溯源到原始笔记。
- **会话内连续追问**：会话树支撑指代式追问——「总结上面的内容」由会话记忆直接作答；「第三点有哪些推荐的」自动消解指代再检索。多轮对话不丢上下文。
- **跨会话长期记忆**：三层长期记忆随问答自动沉淀：收藏口味、作答风格等画像常驻注入；「上次整理的考研资料」等情景按需召回。偏好说过一次，下次不再重复。
- **流式输出 + 主动澄清**：思考进度、引用与回答逐字流式呈现。当查询模糊时主动发起澄清卡片，内联在对话流中呈现，并通过计数控制避免过度打断用户。
- **多租户数据隔离**：每位用户有独立的租户标识（用户代码），贯穿语义索引、内容检索、会话上下文，确保所有操作按用户边界隔离，无横向越权风险。

---

## 核心特性

- **流式问答**：`/v1/ask/stream` 以 NDJSON 逐 token 输出，支持 `thinking / references / token / clarify / done / error` 事件。
- **复杂度路由**：LLM-as-a-Judge 判定复杂度，简单问题走 QA 子图，复杂问题走 Creative DAG 子图。
- **混合检索**：向量检索（embedding）+ RRF 融合（tag 加权 + 新鲜度衰减），与召回集一致地统计数量/平台分布。
- **Creative Plan-Solve-Reflect-Synthesize**：复杂问题拆解成 DAG 子任务，波浪式并行求解、四维反思（覆盖/质量/合规/幻觉）、多轮重规划，收敛后合成最终答案。
- **澄清机制（Clarify）**：在 QA Reason / DAG Plan·Reflect·Synthesize 节点按需追问，支持链内多次澄清、resume 恢复、Reflect 强制澄清（replan 多轮仍有 poor/conflicts 时绕过置信度门控触发）。
- **会话记忆（可选）**：Session Tree + 同步 Compaction，支持会话内指代查询（"上面说的什么"）与跨轮上下文。
- **SKILL 渐进式加载**：三级匹配（关键词 → 语义向量 → LLM 仲裁），命中才注入指令，未命中零侵入；支持技能依赖与资源注入。
- **MCP 工具接入**：作为客户端连接外部 MCP Server（stdio / streamable_http），支持运行时增删/重连。
- **可观测与安全**：全链路 Ask 追踪、熔断、限流、CORS、AI 配置 AES 加密、多租户隔离。

---

## 在线演示：Simple 与 Complex 双管线

系统先用大模型判断查询的复杂度，再路由到对应流程。Simple 走线性问答流程，Complex 走规划-求解-评估循环。详见官网演示：<https://omnibox-parse.online/OmniBoxAgent/>

### SIMPLE QA · 线性流程 · 单轮问答

适合计数、存在性、单一主题检索、翻译、通用列表查询。一次分类判定 + 一次知识库检索 + 一次生成。

- **7 节点线性编排**：查询理解 → 账号检查 → 知识库检索 → 质量过滤 → 推理组装 → 流式生成 → 完成输出
- **早期分支机制**：无账号中止、无相关降级、查询模糊澄清，保障可用性
- **引用提前下发**：质量过滤后第一时间把引用发到前端，减少等待
- **流式逐字输出**：最后阶段直接流式生成，用户体验流畅

### DAG CREATIVE · 规划循环 · 多轮合成

适合任务规划、跨平台对比、跨文档综合、深度分析、知识体系整理。先拆解子任务，再并行求解，评估后决定合成或重规划。

- **5 节点 + 3 条件分支**：以规划-求解-评估为核心的状态机结构
- **并行求解**：无依赖的子任务同时执行，并通过并发控制保障稳定
- **四维评估**：覆盖度、合规度、质量、一致性四个维度综合打分
- **三种重规划策略**：局部修复、指令修正、全量重规划（限次保护）

---

## 四层 Harness 架构

四层架构：**接入层接收请求 → 调度层两判分发 → 子图层四条作答管线各司其职 → 服务与插件层提供共享能力**，结果逐层回传。Harness 作为**插件宿主**统一挂载 MCP 工具、SKILL 技能、记忆管理与后台任务，共享同一生命周期，支持运行时增删与热重载。

- **记忆双轨（Memory）**：短期会话记忆管会话内窗口，长期三层记忆管跨会话个性化，读按预算注入、写全搭便车。
- **插件注册表（Plugin Registry）**：Harness 作为插件宿主统一挂载 MCP 工具、SKILL 技能、记忆管理与后台任务，共享生命周期、失败非致命降级。
- **工具支持（MCP Tools）**：工具调用回合注入 schema，模型决策调用、执行、回填直至成答，摄取与富化复用同批工具。
- **技能注入（SKILL）**：与工具正交——渐进式加载，命中才注入指令文本，未命中零行为变化。
- **热插拔与安全边界**：MCP 与技能支持运行时增删、重连、重载，配套鉴权、路径校验与调用预算。

---

## RAG 知识库检索流程

Simple 与 Complex 两条流程都共享同一套检索管线。从查询进入到最终交付结构化上下文，共经历 **5 个连续阶段**，每一步都有明确输入输出和兜底策略：

```
向量搜索 → 融合排序 → 质量门控 → 精炼 → Token 预算
```

1. **向量搜索**：三维（正文 / 媒体文字 / 评论）独立向量检索，评论预算独立、max-pool 聚合。
2. **融合排序**：RRF 融合（tag 软提升 / 记忆 ×1.1 / 条件衰减）。
3. **质量门控**：三级门控（三标签判定 / fail-open / 兜底链每级 ≤ 1）。
4. **精炼（CRAG）**：句子级精炼（>300 字或失败回退原文）。
5. **Token 预算**：首条完整、聚合独立大预算，按余量截断。

---

## 长短期记忆管理

记忆分两条轨道：**短期会话记忆**在会话内维持上下文窗口（超限自动压缩成摘要），**长期跨会话记忆**沉淀用户画像、偏好与情景（下次不再重复澄清）。读取按需注入，写入全部**搭便车**，任一环节失败都不影响问答主路径。

- **短期 · 会话窗口**：会话内维持**摘要 + 最近对话**的恒定窗口，超限自动压缩、到期自动清理。
- **长期 · 三层记忆**：**画像 / 偏好 / 情景**跨会话生效，同类问题不再重复澄清。
- **写入搭便车**：三路写入均搭便车——澄清回答映射偏好、摘要压缩顺带提炼、每 5 轮轻量提取，零额外模型调用。
- **安全可控**：会话 / 长期记忆**独立开关**即时生效；敏感信息不落库，用户可随时查看删除。

---

## 端到端问答流程

一次问答请求的完整生命周期：用户查询进入 → **会话指代判定**（纯回顾直接走会话记忆作答，不检索、不判复杂度）→ 复杂度分类 → Simple 或 Complex 路由 → 知识库检索 → 节点编排 → 流式输出。

- **入口 & 分类**：先判会话指代（纯回顾直接会话作答），否则按复杂度判定 Simple / Complex。
- **Simple 线性流程**：解析 → 检查 → 检索 → 门控 → 推理 → 生成，6 节点顺序执行。
- **Complex 规划循环**：规划 → 求解 → 评估 → 合成，评估不合格限次重规划回环。
- **共享 RAG 管线**：两条流程共用同一检索管线（见上文「RAG 知识库检索流程」）。
- **澄清 & 恢复**：推理 / 规划 / 评估可发起澄清，内联卡片呈现，响应后恢复断点，单流 ≤ 5 次。
- **统一事件输出**：统一事件格式流式下发 token、引用、澄清、完成、异常，一条 SSE 通道还原交互。

---

## 技术栈

| 层 | 技术 |
|---|---|
| 框架 | FastAPI + Uvicorn |
| 编排 | LangGraph（QA / Creative 子图）、LangChain |
| 向量库 | ChromaDB（embedded / http） |
| 元数据库 | MySQL（SQLAlchemy + PyMySQL，只读共享 OmniHub_server） |
| 模型调用 | OpenAI 兼容接口（智谱 GLM 系列 / DeepSeek / Qwen 等任意自选） |
| 工具 | MCP（Python SDK）、SKILL（本地指令包） |
| 其他 | APScheduler、jieba、cryptography、pydantic-settings |

---

## 快速开始

### 1. 环境要求

- Python ≥ 3.11
- MySQL
- ChromaDB（embedded 模式无需独立服务）

### 2. 安装

```bash
pip install -e ".[dev]"
```

### 3. 配置

复制环境变量模板并填写密钥：

```bash
cp .env.example .env
```

关键配置说明见下文「配置」一节。至少需要：

- `MYSQL_HOST` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE`
- `EMBEDDING_API_KEY`（系统级，仅 embedding 任务使用）
- `QU_API_KEY`（查询理解 / 评估，可复用同一把 key）

### 4. 启动

```bash
python -m omnibox_agent.main
```

或

```bash
uvicorn omnibox_agent.api.app:app --host 0.0.0.0 --port 8100
```

启动时自动校验 MySQL 与 ChromaDB 依赖（关键依赖失败则 fail-fast），并拉起 MCP / SKILL / 视频富化 worker 等组件。

健康检查：

```bash
curl http://localhost:8100/health
```

---

## 配置（环境变量）

完整变量见 [.env.example](.env.example)，核心配置分组如下：

| 分组 | 关键变量 | 说明 |
|---|---|---|
| MySQL | `MYSQL_HOST` `MYSQL_USER` `MYSQL_PASSWORD` `MYSQL_DATABASE` | 只读共享 OmniHub_server |
| ChromaDB | `CHROMA_MODE` `CHROMA_PERSIST_DIR` `CHROMA_COLLECTION_NAME` | embedded / http |
| Embedding | `EMBEDDING_BASE_URL` `EMBEDDING_MODEL` `EMBEDDING_DIMENSION` `EMBEDDING_API_KEY` | 系统级 key，唯一使用系统 key 的任务 |
| 查询理解 | `QU_BASE_URL` `QU_MODEL` `QU_API_KEY` | 用户级 key 下运行 |
| 服务 | `AGENT_HOST` `AGENT_PORT` | 默认 `0.0.0.0:8100` |
| 检索 | `RETRIEVAL_TOP_N` `RETRIEVAL_CANDIDATE_N` `UNBOUNDED_*` `RRF_*` | 融合检索参数 |
| MCP | `MCP_ENABLED` `MCP_SERVERS` | 外部 MCP Server JSON 配置 |
| 熔断 | `CB_FAILURE_THRESHOLD` `CB_OPEN_SECONDS` | 按 base_url 熔断 |
| 限流 | `RATE_LIMIT_ENABLED` `RATE_LIMIT_RPM` | `/v1/ask/stream` 每分钟上限 |
| 澄清 | `CLARIFY_*` `REFLECT_FORCE_*` | 澄清计数 / 置信度 / 强制澄清阈值 |
| Creative | `CREATIVE_MODE` `ABSOLUTE_MAX_ROUNDS` `CONVERGENCE_PATIENCE` | DAG 轮数 / 收敛控制 |
| 门控 | `GATE_ENABLED` `REFINEMENT_MIN_CHARS` `MAX_ONDEMAND_IMAGES` | CRAG 质量门控 |
| 生成 | `CONTEXT_TOKEN_BUDGET` `SYNTHESIZE_TOKEN_BUDGET` | token 预算 |
| SKILL | `SKILL_ENABLED` `SKILL_MATCH_MODE` `SKILL_ADMIN_KEY` | 渐进式加载开关与鉴权 |
| 记忆 | `MEMORY_ENABLED` `MEMORY_WHITELIST_USERS` | 默认关闭，灰度开启 |
| 其他 | `AI_CONFIG_ENCRYPTION_KEY` `FINGERPRINT_SALT` | 加密 / 指纹盐 |

---

## Ask 流式协议

`POST /v1/ask/stream` 返回 `application/x-ndjson`，每行一个 JSON 事件：

| event | data | 说明 |
|---|---|---|
| `meta` | `requestId, route, clarifySupported` | 首帧，回传请求 ID |
| `thinking` | `phase, message` | 阶段进度（parsing/checking/retrieving/filtering/reasoning/generating/planning/solving/reflecting…） |
| `references` | `items[]` | 召回条目（标题/封面/平台/作者/链接） |
| `token` | `delta` | 逐 token 增量 |
| `clarify` | `question, options, importance, recommendedKey, context` | 澄清追问 |
| `done` | `sessionId, text, metadata` | 最终回答，含 `llm_calls / elapsed_s / confidence / skills` 等观测字段 |
| `error` | `reason, code` | 错误 |

QA 管线事件顺序：

```
thinking(parsing) → thinking(checking) → thinking(retrieving) → thinking(filtering)
→ [references] → thinking(reasoning) → [clarify | thinking(generating) → token* → done/error]
```

Creative DAG 管线：

```
thinking(planning) → thinking(solving)* → thinking(reflecting)* → thinking(generating) → token* → done
```

会话内指代查询（"上面说的什么"）走独立会话记忆管线，跳过检索。

---

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/ask/stream` | 流式问答（核心） |
| DELETE | `/v1/session/{session_id}` | 级联删除会话树 |
| GET | `/v1/embed/status` | 向量覆盖度 |
| POST | `/v1/embed/delete` | 删除向量 |
| POST | `/v1/ingest` | 多模态摄取（图片解析 + 异步视频富化） |
| POST | `/v1/ingest/backfill` | 全量回填 |
| GET | `/v1/ingest/video-tasks` | 视频任务状态 / MCP 日用量 |
| GET/POST/DELETE | `/v1/mcp/*` | MCP Server 管理（列表/增删/重连/工具） |
| GET/POST | `/v1/skills/*` | SKILL 管理（写操作需 `X-Skill-Admin-Key` 鉴权） |
| GET/POST | `/v1/eval/*` | 评估框架（模板生成 / 运行） |
| POST | `/v1/task/cancel` | 任务取消登记 |
| GET | `/health` | 聚合健康检查 |

---

## 项目结构

```
omnibox_agent/
├── main.py                      # 应用入口（uvicorn）
├── api/
│   ├── app.py                   # FastAPI 装配（CORS / 限流 / 路由 / 生命周期）
│   ├── lifecycle.py             # Harness 生命周期 + 依赖校验
│   └── routes/                  # ask / embed / ingest / mcp / skills / eval / task
├── agent/
│   ├── harness.py               # AgentHarness：registry + 生命周期 + MCP/SKILL 挂载
│   ├── ask_agent.py             # QA step 实现（Parse/Guard/Retrieve/Gate/Reason）
│   ├── graph_qa.py              # LangGraph QA 子图
│   ├── graph_creative.py        # LangGraph Creative 子图（含强制澄清）
│   ├── graph_skill.py           # 共享 skill 节点
│   ├── stream_pipeline.py       # 流式管线（QA / Creative / 会话记忆 / resume）
│   ├── orchestration/router.py  # 复杂度路由
│   ├── mcp_client.py            # MCP 客户端管理
│   ├── context.py               # AgentContext
│   └── loop.py                  # 执行器 / 阻塞转异步
├── services/                    # 业务服务（检索/摄取/澄清/创意编排/会话记忆/技能…）
├── skills/                      # SKILL 子系统（model/store/loader/manager/validator）
├── models/                      # Pydantic 数据模型
├── core/                        # config / database / tracing / auth / 熔断 / 加密
└── evaluation/                  # 评估框架（eval set / runner / skill 匹配评估）
```

---

## 核心机制

### Creative DAG 管线

```
skill → plan → solve → reflect → { synthesize | replan → solve } → done
```

- **Plan**：拆解子任务，失败则回退 QA。
- **Solve**：波浪式并行求解子任务（含 stale 重跑）。
- **Reflect**：四维评估（coverage / quality / compliance / hallucination），产出 replan 动作或冲突。
- **Replan**：Strategy 1/2 普通重规划，Strategy 3 冻结已完成节点全量重规划（有次数上限）。
- **Synthesize**：all-empty 短路或完整合成，逐 token 流式输出。

收敛保证：轮数 ≤ 10、连续 2 轮无改善即止损、单任务最多 2 次重跑、无时间上限（用户指令）。

### 澄清机制（Clarify）

- 澄清点：QA Reason / DAG Plan·Reflect·Synthesize。
- 双维计数：整流上限（`max_total_per_stream`，默认 5）+ 每节点上限（QA 3 / DAG 各 2），`try_incr` 原子占位防并发超限。
- Resume：用户选择/补充后经 `resume_context` 恢复，simple QA 直接合成，DAG 走 augmented query 重跑或**增量 resume**（局部重做 / 直接合成）。
- **Reflect 强制澄清**：replan 多轮（累计 ≥ min_rounds）仍有 poor/conflicts 时，绕过置信度门控强制触发，LLM 仅负责措辞（选项为固定 A/B/C 模板）。

### 会话记忆（默认关闭）

- Session Tree 存储问答节点，`ask.py` 在请求生命周期内接入（append_user / append_assistant / clarify 上树 / interrupted 标记）。
- 同步 Compaction：超阈值时压缩摘要，保证树一致。
- 会话内指代查询（R7/R8/R9）走独立会话记忆管线，结合条件检索。

### SKILL

- 三级匹配：Level0 关键词（零成本召回）→ Level1 语义向量（兜底召回 + 精排）→ Level2 LLM 仲裁。
- 渐进式：命中才懒加载 `SKILL.md` 指令注入 prompt，未命中零侵入。
- 能力：技能依赖（`{{skill:name}}`）、资源注入（`{{resource:path}}`）、YAML front matter、管理侧鉴权（`SKILL_ADMIN_KEY`，fail-closed）。

---

## 测试

```bash
# 冒烟测试：QA 子图 / Creative 子图 / LLM 层
python scripts/smoke_qa_graph.py
python scripts/smoke_creative_graph.py
python scripts/regression_llm_layer.py

# 单元测试
pytest
```

---

## 相关链接

- 🌐 产品官网 / 在线演示：<https://omnibox-parse.online/OmniBoxAgent/>
- 📄 许可证：[MIT](LICENSE)

---

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
