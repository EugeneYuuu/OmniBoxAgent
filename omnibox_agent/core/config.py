"""Application configuration via environment variables / .env."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "omnibox_agent"
    password: str = ""
    database: str = "omnihub"

    @property
    def url(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ChromaConfig:
    mode: str = "embedded"  # "embedded" or "http"
    persist_dir: str = "./chroma_data"
    collection_name: str = "omnihub_items"
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class EmbeddingConfig:
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "embedding-3"
    dimension: int = 2048
    api_key: str = ""


@dataclass
class QUConfig:
    """Query Understanding configuration.

    No timeout — QU is not time-limited (user directive).
    """
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "glm-4-flash"
    api_key: str = ""


@dataclass
class AgentConfig:
    host: str = "0.0.0.0"
    port: int = 8100


@dataclass
class SyncConfig:
    batch_size: int = 50


@dataclass
class RetrievalConfig:
    top_n: int = 5
    # 粗召回 topk（第一阶段，向量召回）：50~100。这是语义相似度召回预算，
    # 决定进 RRF 融合的候选数量上限。时间列举类查询会被全库召回覆盖以保新收藏。
    candidate_n: int = 50
    recency_top_n: int = 50
    # 精召回 topk（第二阶段，排序后截取）：20~50。粗召回融合并排序后，
    # 用户未显式指定条数时按此截取进入门控/生成。用户问题自带 topk（limit_count）
    # 时优先使用用户值，覆盖本默认。
    refine_top_n: int = 20
    # Unbounded mode (used when the user did NOT ask for a specific count or
    # time window — e.g. aggregation / summary queries). Widens recall so the
    # generator can see the whole主题范畴 instead of being capped at top_n.
    unbounded_top_n: int = 100
    unbounded_candidate_n: int = 150
    # Max number of reference items shown to the user (display only; the
    # generator may receive more via the unbounded path).
    reference_display_limit: int = 20
    rrf_k: int = 60
    rrf_vector_weight: float = 0.8
    rrf_bm25_weight: float = 0.2
    # Comment supplement channel: comment vectors join the SAME retrieval
    # pipeline step as main/media, but with an INDEPENDENT top-k budget
    # (comment_candidate_n) — they are never squeezed by the primary
    # candidate_n pool. They enter RRF weighted below rrf_vector_weight and
    # marked is_comment_match, so they can supplement the candidate pool but
    # can never displace main/media matches in ranking.
    # Set RRF_COMMENT_WEIGHT=0 to disable the comment channel entirely.
    rrf_comment_weight: float = 0.3
    # 评论补充通道独立召回预算（用户指令：评论向量不要限制 top-k）。
    # 仅作 ChromaDB 查询的必要 n_results 值，不与主通道共享 top-k；
    # 命中后仍经 RRF 降权 + gate + token 预算收口。
    comment_candidate_n: int = 200
    # 默认不限制 top-k（用户指令）：无显式条数限制时按用户库全量召回。
    # 仅作 ChromaDB 查询必要的 n_results 防御上限（取 min(库向量总数, 该值)），
    # 要真正覆盖更大库就调大该值。
    retrieval_max_candidates: int = 5000
    tag_boost_factor: float = 0.15
    freshness_lambda: float = 0.01
    # Comment enrichment is now per-item, per-sub-task inside the DAG
    # creative solver — NOT a global threshold fallback. See creative_solver.py
    # for _supplement_docs_with_comments().


@dataclass
class McpServerConfig:
    """Configuration for a single MCP server connection."""
    name: str = ""
    transport: str = "stdio"  # "stdio" or "sse"
    command: str = ""         # for stdio
    args: list[str] = field(default_factory=list)  # for stdio
    url: str = ""             # for sse
    env: dict[str, str] = field(default_factory=dict)  # for stdio subprocess env


@dataclass
class MCPConfig:
    """MCP client configuration -- agent connects to external MCP servers as a client."""
    enabled: bool = False
    servers: list[McpServerConfig] = field(default_factory=list)
    result_max_chars: int = 2000
    tool_timeout_s: float = 10.0


@dataclass
class CircuitBreakerConfig:
    """Per-base_url circuit breaker configuration."""
    failure_threshold: int = 5
    open_seconds: int = 30
    half_open_max_requests: int = 1
    max_breakers: int = 128


@dataclass
class IngestionConfig:
    """v4.1 Layer 0 ingestion pipeline configuration."""
    summary_budget: int = 300            # 摘要长度预算(字), 超 300 字才调 LLM 摘要
    mcp_daily_budget: int = 2000         # 每日 MCP 调用上限, 超限熔断转排队
    max_concurrent_mcp: int = 3          # MCP 并发限流(semaphore)
    video_poll_interval: float = 2.0     # 后台轮询间隔(秒), 任务无超时, 轮询至成功/失败
    video_max_consecutive_errors: int = 8  # 视频任务连续异常上限, 超过则放弃该任务
    index_version: str = "v4"            # 索引格式版本, 格式/模型变更时递增触发全库重建
    mcp_retry_max: int = 2              # MCP 调用重试次数(指数退避)


@dataclass
class GateConfig:
    """v4.1 Layer 2 quality gate configuration."""
    enabled: bool = True                  # Gate on/off (rollback switch)
    max_ondemand_images: int = 6         # 查询期补解析图片数上限
    refinement_min_chars: int = 300      # CRAG 知识精炼触发阈值(字)
    refinement_fallback_mode: str = "keep_full"  # 无句子通过阈值时的兜底模式
    gate_retry_max: int = 2             # 批量判定 JSON 解析失败重试次数
    # Skip the (expensive, and for very large sets useless) relevance judge
    # when the candidate set exceeds this size — keep everything and let
    # refinement + budget fit proceed (e.g. unbounded aggregation queries).
    max_judge_docs: int = 120
    # ── 评论区兜底（comment fallback drill）──
    # 用户原则：内容正文找不到问题的语义时（judge 无 relevant、仅
    # topic_relevant），再去该内容的评论向量找。0 = 关闭兜底（回滚开关）。
    comment_drill_max_items: int = 5     # 最多为前 N 条条目附评论文本
    comment_drill_max_chars: int = 400   # 每条评论文本注入截断（字符）


@dataclass
class CreativeConfig:
    """v4.1 creative task Plan-Solve-Reflect configuration."""
    mode: str = "on"                    # off / whitelist / all
    absolute_max_rounds: int = 10        # 轮数天花板
    convergence_patience: int = 2        # 连续 N 轮无改善则止损
    rerun_cap: int = 2                   # 单任务最大执行次数(含 stale 重跑)


@dataclass
class EvalConfig:
    """v4.1 evaluator LLM configuration."""
    model: str = "glm-4-flash"           # 门控/精炼/Reflect 批量判定用的评估器模型
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""                    # 留空则回退到 QU_API_KEY


@dataclass
class GenerationConfig:
    """v4.1 Layer 3 generation configuration."""
    context_token_budget: int = 3000     # 检索内容注入上限（bounded 查询）
    synthesize_token_budget: int = 6000  # 合成输入上限
    # Wider budget used for unbounded aggregation / summary queries so the
    # whole主题范畴 actually reaches the LLM after the gate.
    unbounded_context_token_budget: int = 12000


@dataclass
class SkillConfig:
    """SKILL 渐进式加载配置（docs/skill-support-design.md §7）。

    - enabled: 灰度开关，关闭时 skill 节点零开销降级
    - dir: 技能根目录（默认 ./skills）
    - match_mode: keyword | embedding | hybrid | llm
    - max_inject: 最多注入的技能数
    - max_instruction_chars: 所有命中技能指令（含资源）拼接后的总字符上限
    - select_top_k: Level1 收敛后送入 Level2 的候选数上限
    - similarity_threshold: Level1 余弦相似度阈值
    - allowed_source_roots: source 模式允许的源目录根（默认 [dir]）
    - admin_key: 管理侧鉴权密钥，空则写操作默认拒绝（fail-closed）
    - max_resource_chars: 单个资源文件注入上限（§4.7）
    - selector_*: Level2 LLM 仲裁规格，留空回退 evaluator
    """
    enabled: bool = False
    dir: str = "./skills"
    match_mode: str = "hybrid"            # keyword | embedding | hybrid | llm
    max_inject: int = 3
    max_instruction_chars: int = 6000     # 所有命中技能拼接后的总字符上限
    select_top_k: int = 6                 # Level2 候选数上限（= max_inject × 2）
    similarity_threshold: float = 0.5     # Level1 余弦相似度阈值
    keyword_min_hits: int = 1             # Level0 命中的最小信心下限（keyword 模式的置信保障）
    max_dep_depth: int = 5                 # 技能间依赖的最大嵌套深度（防循环/过深）
    allowed_source_roots: list[str] = field(default_factory=lambda: [])
    admin_key: str = ""                   # 管理侧鉴权密钥，空则写操作默认拒绝
    max_resource_chars: int = 2000        # 单个资源文件注入上限
    selector_model: str = ""
    selector_base_url: str = ""
    selector_api_key: str = ""


@dataclass
class CorsConfig:
    """CORS configuration (issue #11)."""
    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class RateLimitConfig:
    """API rate limiting configuration (issue #15)."""
    enabled: bool = True
    requests_per_minute: int = 60       # /v1/ask/stream 每分钟上限
    ingest_per_minute: int = 30          # /v1/ingest 每分钟上限
    admin_per_minute: int = 30           # /admin/* 每分钟上限（无 token 鉴权，靠限流兜底）


@dataclass
class PipelineConfig:
    """Pipeline budget/behavior configuration (no timeouts — not time-limited)."""
    min_query_length: int = 2
    max_sub_queries: int = 8
    max_tool_rounds: int = 3


@dataclass
class ClarifyConfig:
    """Ask 中间澄清（Clarify）机制配置（docs/clarify-mid-ask-design.md）。

    - enabled: 是否启用澄清判定（灰度开关，后端亦可通过 meta.clarifySupported
      与 X-Clarify-Enabled 请求头透传控制）
    - max_total_per_stream: 同一 Stream（同一 clarify_session_id 的整个问答链路）
      内最多允许的总澄清次数；达到上限后，无论哪个澄清点都强制兜底作答。
      （max_per_ask 为 v2 起废弃的兼容字段，不再参与上限计算）
    - max_per_phase_qa: Simple QA 管线同一澄清点（Reason 节点）内最多允许连续澄清次数
    - max_per_phase_dag: DAG/Creative 管线每个节点（Plan/Reflect/Synthesize 各算一个
      独立计数点）内最多允许连续澄清次数
    - confidence_threshold: LLM 判定置信度阈值，低于该值视为不需要澄清
    - reflect_force_*: Reflect 阶段强制澄清——replan 多轮（默认 ≥2 轮，即 round≥3，
      与 max_rounds 钳制）仍有 poor/conflicts 时绕过 confidence 门控强制触发，
      LLM 仅负责措辞；计数上限仍生效（见 judge_dag_clarification 的 forced 参数）
    """
    enabled: bool = True
    max_per_ask: int = 5                  # ← v2 起废弃：仅向后兼容旧调用方，不再参与上限计算
    max_total_per_stream: int = 5         # ← 新权威：整流上限 5
    max_per_phase_qa: int = 3             # Simple: 单节点最多 3
    max_per_phase_dag: int = 2            # DAG: plan/reflect/synthesize 各点最多2
    confidence_threshold: float = 0.6
    # LLM 判定使用的模型规格（复用 complexity_classifier 同规格）
    judge_model: str = "glm-4-flash"
    judge_max_tokens: int = 1024
    # Reflect 强制澄清（v3）：replan 多轮仍有 poor/conflicts 时，规则先行强制触发
    # —— 绕过 confidence 门控，LLM 只负责措辞；两层计数上限（total/phase）仍然生效
    reflect_force_enabled: bool = True
    # round N 的 reflect 发生在 N-1 次 replan 之后：3 = 严格"已重规划 2 轮"。
    # 运行时与 max_rounds 钳制（max_rounds=2 → 最后一轮 reflect 也触发；=1 → 永不触发），
    # 且下限为 2，保证零次 replan 时不追问
    reflect_force_min_rounds: int = 3
    reflect_force_on_poor: bool = True       # quality=poor 的任务 ≥1 → 触发
    reflect_force_on_conflicts: bool = True  # conflicts ≥1 → 触发（replan 修不了，交用户裁决）

    def effective_max_total(self) -> int:
        """整流上限（v2/R2 修复）：直接返回 max_total_per_stream。

        旧实现取 max(max_per_ask, max_total_per_stream)，导致收紧配置永远被旧字段
        顶回（如设 CLARIFY_MAX_TOTAL_PER_STREAM=3 仍得 5）——上限无法调小。
        """
        return self.max_total_per_stream


# 模型上下文窗口映射（保守值；未命中用 MemoryConfig.default_window_tokens）。
# 用户自带模型窗口各异，Compaction 触发与上下文预算按此映射，避免小窗口模型溢出。
MODEL_WINDOWS: dict[str, int] = {
    "glm-4-flash": 32768,
    "glm-4": 128000,
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
    "qwen-plus": 32000,
    "qwen-max": 32000,
}


@dataclass
class MemoryConfig:
    """Agent 侧记忆系统（Session Tree + Compaction）配置（MEMORY_SYSTEM_DESIGN.md v1.1 §6
    + MEMORY_HARNESS_INTEGRATION_DESIGN.md V2.0 §4.2 / §13.3）。

    - enabled: ⚠️ V2.0 起默认开（原灰度默认关）；env MEMORY_ENABLED，false=部署层熔断。
    - whitelist_users: V2.0 废弃（由 flag 平台取代，§17）——读入仅记 warning，不参与判定。
    - reply_max_tokens: 必须 ≥ 生成侧 stream_chat 的 max_tokens（4096），否则回复溢出窗口。
    - keep_recent_tokens + max_summary_tokens 之和必须 < context_budget。
    - cleanup_interval_hours / cleanup_enabled: harness 托管的后台清理（§6.5 所有权）。
    - long_term_*: 长期记忆（L1 画像 / L2 偏好 / L3 向量，第二部分 §13.3）。
    """
    enabled: bool = True              # ⚠️ V2.0 默认开（原 False）；env: MEMORY_ENABLED，false=部署层熔断
    whitelist_users: list[str] = field(default_factory=list)   # 废弃：读入仅记 warning，不参与判定（§17）
    default_window_tokens: int = 32768
    reply_max_tokens: int = 4096
    reserve_tokens: int = 500
    keep_recent_tokens: int = 6000
    max_summary_tokens: int = 1200
    compact_threshold: float = 0.9
    qu_history_hours: int = 12
    cleanup_interval_hours: int = 24   # 后台清理周期（harness 托管，替代手动脚本）；load_config 中 max(1,...) 兜底
    cleanup_enabled: bool = True       # 进程内清理开关；false 时交回 cron/后端 @Scheduled 托管（§6.5）
    # —— 以下为第二部分长期记忆配置（§13.3）——
    long_term_enabled: bool = True     # env: MEMORY_LONG_TERM_ENABLED（默认开；false=部署层熔断）
    long_term_reserve: int = 800       # 注入预算（token，三层合计硬上限）
    lt_extract_interval: int = 5       # 未达压缩阈值时每 N 轮轻量提取
    lt_batch_users: int = 100          # 周期任务单批用户数

    async def is_enabled_for(self, user_id: str | None) -> bool:
        """用户级判定：薄代理 → await flag_service.is_enabled(uid, "memory_session")（§17）。

        flag 平台未落地（模块不存在）时回退 self.enabled（env kill 语义，§13.3）。
        注：flag_service.is_enabled 含 DB I/O 为 async，本代理必须 async 且调用方 await——
        直接返回协程对象会被 bool() 判恒真，导致用户级关闭静默失效。
        """
        if self.whitelist_users:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "MEMORY_WHITELIST_USERS is deprecated (V2.0, replaced by flag platform); ignored")
        try:
            from omnibox_agent.services.flag_service import is_enabled as flag_is_enabled
            return await flag_is_enabled(user_id or "", "memory_session")
        except (ImportError, ModuleNotFoundError):
            return self.enabled

    async def is_enabled_for_lt(self, user_id: str | None) -> bool:
        """长期记忆用户级判定：await flag_service.is_enabled(uid, "memory_long_term")（§17）。

        开发期 fallback:flag_service 未实现时返回 self.long_term_enabled（纯 env kill 语义）。
        """
        try:
            from omnibox_agent.services.flag_service import is_enabled as flag_is_enabled
            return await flag_is_enabled(user_id or "", "memory_long_term")
        except (ImportError, ModuleNotFoundError):
            return self.long_term_enabled

    def window_for(self, model_name: str | None) -> int:
        return MODEL_WINDOWS.get(model_name or "", self.default_window_tokens)

    def context_budget_for(self, model_name: str | None,
                           long_term_tokens: int = 0) -> int:
        """历史+摘要可占用的窗口预算 = 窗口 - 回复预留 - 安全余量 - 长期记忆实际注入（§6 / §12.3）。

        长期记忆按**实际注入**扣减（min(long_term_tokens, long_term_reserve)）：
        无长期产出时预算与会话记忆现状一致（零副作用），不是开关级全局扣减。
        """
        w = self.window_for(model_name)
        return (w - self.reply_max_tokens - self.reserve_tokens
                - min(max(0, long_term_tokens), self.long_term_reserve))

    def should_compact_at(self, model_name: str | None,
                          long_term_tokens: int = 0) -> int:
        """触发压缩的 token 阈值：context_budget × compact_threshold（含 LT 扣减，§12.3）。"""
        return int(self.context_budget_for(model_name, long_term_tokens) * self.compact_threshold)


@dataclass
class Config:
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qu: QUConfig = field(default_factory=QUConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    cors: CorsConfig = field(default_factory=CorsConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    fingerprint_salt: str = "omnihub-ask-agent-v1"
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    creative: CreativeConfig = field(default_factory=CreativeConfig)
    evaluator: EvalConfig = field(default_factory=EvalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    clarify: ClarifyConfig = field(default_factory=ClarifyConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        mysql=MySQLConfig(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "omnibox_agent"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "omnihub"),
        ),
        chroma=ChromaConfig(
            mode=os.getenv("CHROMA_MODE", "embedded"),
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "omnihub_items"),
            host=os.getenv("CHROMA_HOST", "127.0.0.1"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        ),
        embedding=EmbeddingConfig(
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            model=os.getenv("EMBEDDING_MODEL", "embedding-3"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "2048")),
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
        ),
        qu=QUConfig(
            base_url=os.getenv("QU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            model=os.getenv("QU_MODEL", "glm-4-flash"),
            api_key=os.getenv("QU_API_KEY", ""),
        ),
        agent=AgentConfig(
            host=os.getenv("AGENT_HOST", "0.0.0.0"),
            port=int(os.getenv("AGENT_PORT", "8100")),
        ),
        sync=SyncConfig(
            batch_size=int(os.getenv("VECTOR_SYNC_BATCH_SIZE", "50")),
        ),
        retrieval=RetrievalConfig(
            top_n=int(os.getenv("RETRIEVAL_TOP_N", "5")),
            candidate_n=int(os.getenv("RETRIEVAL_CANDIDATE_N", "50")),
            recency_top_n=int(os.getenv("RECENCY_TOP_N", "50")),
            refine_top_n=int(os.getenv("RETRIEVAL_REFINE_TOP_N", "20")),
            unbounded_top_n=int(os.getenv("RETRIEVAL_UNBOUNDED_TOP_N", "100")),
            unbounded_candidate_n=int(os.getenv("RETRIEVAL_UNBOUNDED_CANDIDATE_N", "150")),
            reference_display_limit=int(os.getenv("RETRIEVAL_REFERENCE_DISPLAY_LIMIT", "20")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            rrf_vector_weight=float(os.getenv("RRF_VECTOR_WEIGHT", "0.6")),
            rrf_bm25_weight=float(os.getenv("RRF_BM25_WEIGHT", "0.4")),
            rrf_comment_weight=float(os.getenv("RRF_COMMENT_WEIGHT", "0.3")),
            comment_candidate_n=int(os.getenv("RETRIEVAL_COMMENT_CANDIDATE_N", "200")),
            retrieval_max_candidates=int(os.getenv("RETRIEVAL_MAX_CANDIDATES", "5000")),
            tag_boost_factor=float(os.getenv("TAG_BOOST_FACTOR", "0.15")),
            freshness_lambda=float(os.getenv("FRESHNESS_LAMBDA", "0.01")),
        ),
        mcp=MCPConfig(
            enabled=os.getenv("MCP_ENABLED", "false").lower() == "true",
            servers=_parse_mcp_servers(os.getenv("MCP_SERVERS", "")),
            result_max_chars=int(os.getenv("MCP_RESULT_MAX_CHARS", "2000")),
            tool_timeout_s=float(os.getenv("MCP_TOOL_TIMEOUT_S", "10.0")),
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=int(os.getenv("CB_FAILURE_THRESHOLD", "5")),
            open_seconds=int(os.getenv("CB_OPEN_SECONDS", "30")),
        ),
        cors=CorsConfig(
            allow_origins=_parse_cors_origins(os.getenv("CORS_ALLOW_ORIGINS", "*")),
            allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
            allow_methods=_parse_list_env(os.getenv("CORS_ALLOW_METHODS", "*")),
            allow_headers=_parse_list_env(os.getenv("CORS_ALLOW_HEADERS", "*")),
        ),
        rate_limit=RateLimitConfig(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
            ingest_per_minute=int(os.getenv("RATE_LIMIT_INGEST_RPM", "30")),
            admin_per_minute=int(os.getenv("RATE_LIMIT_ADMIN_RPM", "30")),
        ),
        pipeline=PipelineConfig(
            min_query_length=int(os.getenv("PIPELINE_MIN_QUERY_LENGTH", "2")),
            max_sub_queries=int(os.getenv("PIPELINE_MAX_SUB_QUERIES", "8")),
            max_tool_rounds=int(os.getenv("PIPELINE_MAX_TOOL_ROUNDS", "3")),
        ),
        skills=SkillConfig(
            enabled=os.getenv("SKILL_ENABLED", "false").lower() == "true",
            dir=os.getenv("SKILL_DIR", "./skills"),
            match_mode=os.getenv("SKILL_MATCH_MODE", "hybrid"),
            max_inject=int(os.getenv("SKILL_MAX_INJECT", "3")),
            max_instruction_chars=int(os.getenv("SKILL_MAX_INSTRUCTION_CHARS", "6000")),
            select_top_k=int(os.getenv("SKILL_SELECT_TOP_K", "6")),
            similarity_threshold=float(os.getenv("SKILL_SIMILARITY_THRESHOLD", "0.5")),
            keyword_min_hits=int(os.getenv("SKILL_KEYWORD_MIN_HITS", "1")),
            max_dep_depth=int(os.getenv("SKILL_MAX_DEP_DEPTH", "5")),
            allowed_source_roots=_parse_list_env(os.getenv("SKILL_ALLOWED_SOURCE_ROOTS", "")),
            admin_key=os.getenv("SKILL_ADMIN_KEY", ""),
            max_resource_chars=int(os.getenv("SKILL_MAX_RESOURCE_CHARS", "2000")),
            selector_model=os.getenv("SKILL_SELECTOR_MODEL", ""),
            selector_base_url=os.getenv("SKILL_SELECTOR_BASE_URL", ""),
            selector_api_key=os.getenv("SKILL_SELECTOR_API_KEY", ""),
        ),
        fingerprint_salt=os.getenv("FINGERPRINT_SALT", "omnihub-ask-agent-v1"),
        ingestion=IngestionConfig(
            summary_budget=int(os.getenv("INGESTION_SUMMARY_BUDGET", "300")),
            mcp_daily_budget=int(os.getenv("MCP_DAILY_BUDGET", "2000")),
            max_concurrent_mcp=int(os.getenv("MAX_CONCURRENT_MCP", "3")),
            video_poll_interval=float(os.getenv("VIDEO_POLL_INTERVAL", "2")),
            video_max_consecutive_errors=int(os.getenv("VIDEO_MAX_CONSECUTIVE_ERRORS", "8")),
            index_version=os.getenv("INDEX_VERSION", "v4"),
            mcp_retry_max=int(os.getenv("MCP_RETRY_MAX", "2")),
        ),
        gate=GateConfig(
            enabled=os.getenv("GATE_ENABLED", "true").lower() == "true",
            max_ondemand_images=int(os.getenv("MAX_ONDEMAND_IMAGES", "6")),
            refinement_min_chars=int(os.getenv("REFINEMENT_MIN_CHARS", "300")),
            refinement_fallback_mode=os.getenv("REFINEMENT_FALLBACK_MODE", "keep_full"),
            gate_retry_max=int(os.getenv("GATE_RETRY_MAX", "2")),
            max_judge_docs=int(os.getenv("GATE_MAX_JUDGE_DOCS", "120")),
        ),
        creative=CreativeConfig(
            mode=os.getenv("CREATIVE_MODE", "on"),
            absolute_max_rounds=int(os.getenv("ABSOLUTE_MAX_ROUNDS", "10")),
            convergence_patience=int(os.getenv("CONVERGENCE_PATIENCE", "2")),
            rerun_cap=int(os.getenv("RERUN_CAP", "2")),
        ),
        evaluator=EvalConfig(
            model=os.getenv("EVALUATOR_MODEL", os.getenv("QU_MODEL", "glm-4-flash")),
            base_url=os.getenv("EVALUATOR_BASE_URL", os.getenv("QU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")),
            api_key=os.getenv("EVALUATOR_API_KEY", os.getenv("QU_API_KEY", "")),
        ),
        generation=GenerationConfig(
            context_token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000")),
            synthesize_token_budget=int(os.getenv("SYNTHESIZE_TOKEN_BUDGET", "6000")),
            unbounded_context_token_budget=int(os.getenv("UNBOUNDED_CONTEXT_TOKEN_BUDGET", "12000")),
        ),
        clarify=ClarifyConfig(
            enabled=os.getenv("CLARIFY_ENABLED", "true").lower() == "true",
            max_per_ask=int(os.getenv("CLARIFY_MAX_PER_ASK", "5")),
            max_total_per_stream=int(os.getenv("CLARIFY_MAX_TOTAL_PER_STREAM", "5")),
            max_per_phase_qa=int(os.getenv("CLARIFY_MAX_PER_PHASE_QA", "3")),
            max_per_phase_dag=int(os.getenv("CLARIFY_MAX_PER_PHASE_DAG", "2")),
            confidence_threshold=float(os.getenv("CLARIFY_CONFIDENCE_THRESHOLD", "0.6")),
            judge_model=os.getenv("CLARIFY_JUDGE_MODEL", "glm-4-flash"),
            judge_max_tokens=int(os.getenv("CLARIFY_JUDGE_MAX_TOKENS", "1024")),
        ),
        memory=MemoryConfig(
            enabled=os.getenv("MEMORY_ENABLED", "true").lower() == "true",
            whitelist_users=_parse_list_env(os.getenv("MEMORY_WHITELIST_USERS", "")),
            default_window_tokens=int(os.getenv("MEMORY_DEFAULT_WINDOW_TOKENS", "32768")),
            reply_max_tokens=int(os.getenv("MEMORY_REPLY_MAX_TOKENS", "4096")),
            reserve_tokens=int(os.getenv("MEMORY_RESERVE_TOKENS", "500")),
            keep_recent_tokens=int(os.getenv("MEMORY_KEEP_RECENT_TOKENS", "6000")),
            max_summary_tokens=int(os.getenv("MEMORY_MAX_SUMMARY_TOKENS", "1200")),
            compact_threshold=float(os.getenv("MEMORY_COMPACT_THRESHOLD", "0.9")),
            qu_history_hours=int(os.getenv("MEMORY_QU_HISTORY_HOURS", "12")),
            cleanup_interval_hours=max(1, int(os.getenv("MEMORY_CLEANUP_INTERVAL_HOURS", "24"))),
            cleanup_enabled=os.getenv("MEMORY_CLEANUP_ENABLED", "true").lower() == "true",
            long_term_enabled=os.getenv("MEMORY_LONG_TERM_ENABLED", "true").lower() == "true",
            long_term_reserve=int(os.getenv("MEMORY_LONG_TERM_RESERVE", "800")),
            lt_extract_interval=int(os.getenv("MEMORY_LT_EXTRACT_INTERVAL", "5")),
            lt_batch_users=int(os.getenv("MEMORY_LT_BATCH_USERS", "100")),
        ),
    )



def _parse_mcp_servers(raw: str) -> list[McpServerConfig]:
    """Parse MCP_SERVERS env var as JSON list of server configs.

    Example:
      MCP_SERVERS='[{"name":"filesystem","transport":"stdio","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}]'
    """
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        servers = []
        for item in data:
            if not isinstance(item, dict):
                continue
            env = item.get("env", {})
            if not isinstance(env, dict):
                env = {}
            servers.append(McpServerConfig(
                name=item.get("name", ""),
                transport=item.get("transport", "stdio"),
                command=item.get("command", ""),
                args=item.get("args", []),
                url=item.get("url", ""),
                env={str(k): str(v) for k, v in env.items()},
            ))
        return servers
    except (json.JSONDecodeError, TypeError) as e:
        import logging
        logging.getLogger(__name__).warning("Failed to parse MCP_SERVERS: %s", e)
        return []


def _parse_cors_origins(raw: str) -> list[str]:
    """Parse CORS_ALLOW_ORIGINS: comma-separated or '*' for all."""
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _parse_list_env(raw: str) -> list[str]:
    """Parse a comma-separated env var into list, '*' means all."""
    if raw.strip() == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


# Singleton
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
