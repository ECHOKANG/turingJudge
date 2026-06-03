<div align="center">

# Turing Eval · 多轮对话评估平台（Web）
**⚠️声明**：本项目为2026美团AI HACKSON参赛作品🦘🦘🦘，该仓库未处于公开状态，仅被授权的人员可访问

一个用于“接入被测 Agent → 生成/配置评估 CASE → 批量运行评估 → 查看评估报告”的轻量级 Web 工具。

### 👉 推进入口：产品说明网站：**[Turing Eval](http://140.143.211.100:5001/static/turing-platform/turing-platform.html)**


</div>

## 创新点

**动态对话模拟** ：现有的Eval对用户模拟的策略几乎都是基于上下文休息进行的，而真实人类的每一轮对话是基于多重因素动态变化的；每一轮对话产生的上下文信息都在改变用户的心理状态（情绪、价值、信任、目的\.\.\.），这一秒的用户已经不再是上一秒的用户。Turing通过每轮对话结束后模拟用户多个维度的心理状态，更新用户状态作为模拟器参数来构建下一轮对话内容无限逼近最真实的业务场景。

**个性化用户模拟**：突破"固定 Prompt 池"局限，** 由Agent 的 System Prompt 抽取业务语境**（角色、任务、知识点、约束），动态生成高度耦合的测试用户身份与对话目标。一次批量样本，人群特征分布前端可视化配置，实现高效批跑。

**规则命中 \+ LLM 评分双轨: **硬规则（rule\_flag）检测红线词、信息泄漏、脱离平台引导等可机器判断项；软评分（LLM Judge）评估体验、共情、解决度等语义维度。两者交叉验证。

**并行提效** ：把 "一次批量评估" 拆成可独立调度的 subagent 任务单元，由 \*\* 编排层（Orchestrator）\*\* 统一管理生命周期、并发、重试、进度；并行16线程，一次case评估提速2\.12x

## 实测效果

|**指标**|**数值**|**说明**|
|---|---|---|
|Kappa coefficient|κ\>0\.7|与人工评测达到实质性一致水平|
|规则检测精确率|\>95%|规则中风险检查点覆盖率|
|单 CASE 评测时延|21\.5～28\.7s |单case耗时（默认最大输出长度）|
|平均测试时长|28\.47min|100个Case总耗时|
|cross\-batch stabilit|*σ\<0\.3*|评分波动稳定性\(同 Prompt \+ 同场景多次评测分数一致\)<br>|

## 平台板块：

|Tab|组件|类型|功能|
|---|---|---|---|
|接入Agent|Agent配置|输入|配置字段：Agent 名称，System Prompt，上下文变量，模型，温度，Max Tokens|
||保存样本|按钮|保存当前Agent配置并存入登记列表|
||测试连通性|按钮|测试模型API连通性|
||登记列表|切换列表|切换已保存的不同Agent配置|
|配置CASE|生成模式|切换列表|配置CASE生成数据源，提供了三种<br>1. 模拟CASE：所有CASE由模型模拟<br>2. 扩写：根据线上CASE作为种子进行生成<br>3. 回归：复用线上对话进行测试|
||样本量（回归隐藏）|滑块|选择测试轮次，1\-100|
||模拟配比<br>（回归隐藏）|切换列表<br>|选择用户模拟画像配比来源：<br>1. AI智能分配：模拟分析PE后自行模型<br>2. Persona池，在池子中选择配比|
||Persona配置<br>（百分比问题）|滑块|配置不同群体画像在本次测试的分布|
||预置Persona库|下拉选项|选择预置的Persona库|
||自定义persona|输入|自定义persona库，上传后在上方配置|
||生成多样性|可视化滑块|年龄群体<br>场景关键词<br>目标清晰度<br>语言风格|

## 功能概览

- **被测 Agent 配置**：在网页中填写 Agent 名称、System Prompt、模型参数、上下文字段等，并保存为样本。
- **CASE 生成与配置**：
  - 支持多种 CASE 生成模式（如 LLM 生成 / 参考扩写 / 线上回归）。
  - 支持 **Persona Pool**：用“用户群体 + 权重”的方式批量控制模拟用户分布。
  - 支持自然语言配置“生成多样性要求”。
- **批量评估（V2）**：后端以 `asyncio + semaphore` 执行并发评估，支持超时与重试。
- **实时进度监控**：评估过程中写入 JSONL 事件流，可通过 API 实时拉取，便于定位“卡住/尾延迟”等问题。
- **报告查看**：支持查看历史运行与报告详情。

## 快速开始（本地）

### 1) 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
cp .env.example .env
```

关键变量：

- `OPENAI_API_KEY`：模型服务 Key
- `OPENAI_BASE_URL`：模型服务 Base URL（OpenAI 兼容）
- 批量评估（可选，均有默认值）
  - `EVAL_MAX_CONCURRENCY`：并发数（默认 16）
  - `EVAL_CASE_TIMEOUT`：单 case 超时秒数（默认 180）
  - `EVAL_MAX_RETRIES`：失败重试次数（默认 1）

### 3) 启动

```bash
python3 app.py
```

访问：

- http://localhost:5001

## 配置说明

### 模型角色配置（models.json）

除被测 Agent 外，其它角色（simulator/judge/allocator/refiner/scenario_generator 等）通过 `models.json` 集中配置：

- 文件：`models.json`
- 读取逻辑：`config/config.py`

修改 `models.json` 后，需重启服务生效。

### 实时进度日志

- 写入目录：`output/progress_logs/`
- 格式：JSONL（每行一个事件，写入后立即 flush+fsync）
- 典型事件：
  - `batch_start` / `batch_total` / `case_start` / `case_done` / `case_fail` / `case_timeout` / `batch_done`

## API 说明（常用）

- `POST /api/v2/evaluate_batch`：V2 批量评估（推荐）
- `POST /api/evaluate_batch`：旧批量评估（保留用于回退）
- `GET /api/batch_progress`：获取最近一次 run 的进度日志（或列出已有 runs）
- `GET /api/batch_progress?run_id=...`：按 run_id 获取进度日志
- `GET /api/batch_progress/list`：列出历史 runs

## 关于 Langfuse

项目默认不依赖 Langfuse 进行运行（为了避免在不同环境中因 SDK 版本差异导致评估失败）。
仓库内提供了一个 `langfuse.py` 的 no-op shim，使得已有埋点调用不会影响主流程。

## 目录结构（简要）

- `app.py`：Flask Web 入口与路由
- `templates/index.html`：单页前端（无额外构建步骤）
- `evaluator/`：评估编排、Judge、资源池、进度日志等
- `simulator/`：用户模拟、场景生成、persona 相关逻辑
- `prompts/`：各模块 prompt 模板
- `output/`：运行日志、进度日志、报告输出

## 部署建议（通用）

- 建议使用 **进程守护/反向代理**（systemd / supervisord / gunicorn + nginx）替代开发模式的长连接，以提升稳定性。
- 建议将“提交任务”和“查看进度/结果”分离（提交返回 run_id，前端轮询进度与结果），避免浏览器长时间等待导致的“网络错误”误报。

