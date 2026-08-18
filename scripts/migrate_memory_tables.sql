-- =====================================================================
-- OmniBoxAgent 记忆系统建表脚本（MEMORY_SYSTEM_DESIGN.md v1.1 §3；
-- V2.0 追加长期记忆三表：MEMORY_HARNESS_INTEGRATION_DESIGN.md §10 +
-- USER_FLAG_PLATFORM_DESIGN.md §3.1）
--
-- ⚠️ 执行要求：
--   1. 需以具备 DDL 权限的账号执行（root 或 omnihub 用户）；
--      Agent 应用账号（MYSQL_USER=omnibox_agent）通常只有 DML 权限。
--   2. 本地 MySQL 与 ECS 生产库（39.96.9.213 侧）是两份独立数据，
--      本脚本需在两处各执行一次。
--   3. 幂等：使用 CREATE TABLE IF NOT EXISTS，可重复执行。
--   4. 回滚：只需关闭 MEMORY_ENABLED / MEMORY_LONG_TERM_ENABLED 开关，表可保留不删。
-- =====================================================================

USE omnihub;

-- ---------------------------------------------------------------------
-- 会话表：会话元数据 + leaf 指针
-- 归属校验：所有读写必须带 user_id（sessionId 为客户端生成，可猜测/复用）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_session (
  session_id     VARCHAR(64)  PRIMARY KEY COMMENT '沿用后端生成的 sessionId（客户端生成，见设计 §3.1 冲突说明）',
  user_id        VARCHAR(64)  NOT NULL COMMENT '租户隔离 (user_code)',
  leaf_entry_id  VARCHAR(64)  NULL     COMMENT '树指针：当前对话节点 ID',
  title          VARCHAR(128) NULL,
  status         VARCHAR(16)  NOT NULL DEFAULT 'active',
  last_active_at DATETIME(3)  NOT NULL,
  created_at     DATETIME(3)  NOT NULL,
  KEY idx_user (user_id, last_active_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='OmniBoxAgent 会话树元数据（记忆系统 v1.1）';

-- ---------------------------------------------------------------------
-- 会话节点表：会话树节点
-- 约束语义：内容只增不改（content/role/entry_type/meta 不可变）；
--          结构指针可变更（parent_id 允许 UPDATE——Compaction 上树重定向、
--          中断/完成标记 status）。
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_session_entry (
  id         BIGINT      AUTO_INCREMENT PRIMARY KEY,
  entry_id   VARCHAR(64) NOT NULL COMMENT '业务 ID (e.g., e_xxx)',
  session_id VARCHAR(64) NOT NULL,
  user_id    VARCHAR(64) NOT NULL COMMENT '租户隔离 (user_code)，防跨用户',
  parent_id  VARCHAR(64) NULL COMMENT '父节点 ID，认父不认子；结构指针允许变更',
  entry_type VARCHAR(32) NOT NULL COMMENT 'message/compaction/branch_summary/meta',
  role       VARCHAR(16) NULL COMMENT 'user/assistant (仅 message 类型)',
  status     VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT '仅 message 使用: pending(待答复)/complete(已答复)/interrupted(客户端中断)',
  content    MEDIUMTEXT   NULL COMMENT '正文 / 摘要文本',
  meta       JSON         NULL COMMENT '引用条目 content_ids、clarify 标记、request_id、意图等',
  request_id VARCHAR(64) NULL COMMENT '幂等去重：同一请求不重复 append（一问一答两个节点共用 rid，按 role 区分；NULL 不参与唯一性）',
  token_est  INT          NOT NULL DEFAULT 0 COMMENT '估算 Token 数（中英分语言启发式，见设计 §6）',
  created_at DATETIME(3)  NOT NULL,
  UNIQUE KEY uk_entry (session_id, entry_id),
  UNIQUE KEY uk_request_role (session_id, request_id, role),
  KEY idx_path (session_id, id),
  KEY idx_user_entry (user_id, session_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='OmniBoxAgent 会话树节点（记忆系统 v1.1）';

-- ---------------------------------------------------------------------
-- 长期记忆 L1：用户画像（MEMORY_HARNESS_INTEGRATION_DESIGN.md §10.1，每用户一行）
-- user_id 存 user_code（与 agent_session.user_id 同标准，见 §10 键约定）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_user_profile (
  user_id     VARCHAR(64) PRIMARY KEY COMMENT '租户隔离 (user_code)',
  profile_json  JSON     COMMENT 'LLM 提炼画像（§11.2）；NULL=尚未生成',
  stats_json    JSON     COMMENT 'SQL 统计画像（零 LLM 成本，底座）',
  version     INT NOT NULL DEFAULT 1,
  lt_round_count INT NOT NULL DEFAULT 0   COMMENT '未达压缩阈值时的轻量提取轮次计数（§11.2）',
  last_lt_extract_at DATETIME(3) NULL     COMMENT '上次轻量提取时间（§11.2）',
  updated_at  DATETIME(3) NOT NULL,
  KEY idx_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='长期记忆 L1:用户画像（第二部分 §10.1）';

-- ---------------------------------------------------------------------
-- 长期记忆 L2/L3 索引：跨会话偏好与情景记忆（§10.2，行式 + supersede 链）
-- 召回只取 status='active'；superseded 不物理删（可审计）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_user_memory (
  id            BIGINT AUTO_INCREMENT PRIMARY KEY,
  memory_id     VARCHAR(64) NOT NULL COMMENT 'uuid hex（与 Chroma id 对齐）',
  user_id       VARCHAR(64) NOT NULL COMMENT '租户隔离 (user_code)',
  mem_type      VARCHAR(16) NOT NULL COMMENT 'preference / fact / episodic',
  content       TEXT NOT NULL COMMENT '如 "平台偏好:优先 bilibili"',
  meta          JSON COMMENT 'source_session_id / importance / confidence / evidence / superseded_by',
  status        VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT 'active / superseded / deleted',
  hit_count     INT NOT NULL DEFAULT 0,
  last_accessed_at DATETIME(3) NULL,
  created_at    DATETIME(3) NOT NULL,
  UNIQUE KEY uk_memory (memory_id),
  KEY idx_user_type (user_id, mem_type, status),
  KEY idx_access (user_id, last_accessed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='长期记忆 L2/L3 索引:跨会话偏好与情景记忆（第二部分 §10.2）';

-- ---------------------------------------------------------------------
-- 用户级功能开关（USER_FLAG_PLATFORM_DESIGN.md §3.1，开关平台 F1）
-- 缺席即默认：无行 = FLAG_REGISTRY 默认值；表只存"显式覆盖"（预计 <1% 用户）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_user_feature_flag (
  id         BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id    VARCHAR(64) NOT NULL COMMENT 'user_code,与 agent_session 同标准',
  flag_name  VARCHAR(64) NOT NULL COMMENT '见 flag_service.FLAG_REGISTRY',
  enabled    TINYINT NOT NULL COMMENT '1=显式开 0=显式关;无行=默认值',
  reason     VARCHAR(255) NULL COMMENT '操作原因(审计,如"记忆污染投诉临时关闭")',
  updated_by VARCHAR(64) NULL COMMENT '操作者(admin token 持有者标识)',
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  UNIQUE KEY uk_user_flag (user_id, flag_name),
  KEY idx_flag (flag_name, enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='用户级功能开关(监控平台管理,USER_FLAG_PLATFORM_DESIGN.md)';

-- ---------------------------------------------------------------------
-- 一致性校验（可选，部署后执行确认）：
--   SELECT COUNT(*) FROM agent_session;
--   SELECT COUNT(*) FROM agent_session_entry;
--   SELECT COUNT(*) FROM agent_user_profile;
--   SELECT COUNT(*) FROM agent_user_memory;
--   SELECT COUNT(*) FROM agent_user_feature_flag;
--
-- Chroma 集合 omnihub_user_memories 由 MemoryManager.startup() 首次
-- get_named_collection() 自动创建（§10.4），无需手动建。
-- ---------------------------------------------------------------------
