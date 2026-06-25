"""
Aegis Web API 服务（完整版）
==========================
FastAPI 后端，覆盖所有 Aegis 功能：

端点一览:
  GET  /                    聊天页面
  POST /api/chat            对话（支持文件上传 + action 参数）
  POST /api/chat/stream     流式对话
  GET  /api/health          健康检查

  # 文件处理
  POST /api/file/parse      仅解析文件（返回内容预览）
  POST /api/file/translate  翻译文件
  POST /api/file/polish     润色文件内容
  POST /api/file/generate-ppt  从内容生成 PPT

  # 研究分析
  POST /api/research        研究分析（支持 SWAT + 多源）

  # 提醒
  GET    /api/reminders         列出提醒
  POST   /api/reminders         创建提醒
  DELETE /api/reminders/{id}    取消提醒
  POST   /api/reminders/followup  检查跟进

  # 任务
  GET  /api/tasks             待办概览
  POST /api/tasks/dispatch    手动触发任务分派

  # 记忆
  POST /api/memory/search     搜索记忆
  POST /api/memory/store      手动存储记忆
  POST /api/memory/summarize  总结对话

  # 简报
  POST /api/briefing          生成早晨简报

启动:
    python -m uvicorn src.api.server:app --host 0.0.0.0 --port 7860 --reload
"""

import os
import sys
import json
import uuid
import hashlib
import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from src.models.schemas import TodoItem, ReminderType

logger = logging.getLogger(__name__)

# Shift sys.path manipulation to only happen in __main__ context
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 保护 _save_upload 文件索引的并发访问
_save_upload_lock = threading.Lock()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.core.llm.provider import create_llm_provider
from src.core.memory.memory_manager import MemoryManager
from src.core.agents.orchestrator import Orchestrator
from src.core.tools.file_tools import parse_file, detect_file_type
from src.core.tools.translation_tools import TranslationTool
from src.workflows.file_processing import run_file_processing_workflow
from src.workflows.research_report import run_research_workflow
from src.workflows.reminder_followup import run_reminder_followup_workflow, run_morning_briefing
from src.api.auth import AuthMiddleware
from src.api.rate_limiter import RateLimitMiddleware

app = FastAPI(title="Aegis API", version="0.1.0")

# 中间件顺序（后添加的先执行）：
# 1. CORS（最内层）
# 2. 认证
# 3. 速率限制（最外层，防止资源浪费）
_cors_origins_env = os.getenv("AEGIS_CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] if _cors_origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=bool(_cors_origins_env and "*" not in _cors_origins_env),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 认证中间件（从环境变量/Config 读取配置）
# 配置方式: AEGIS_AUTH_MODE=or|and|off, AEGIS_IP_WHITELIST=ip1,ip2, AEGIS_API_KEYS=key1,key2
# 开发模式: AEGIS_DEV_MODE=true 可跳过认证（仅限本地开发！）
# 生产部署必须同时设置 AEGIS_IP_WHITELIST 和/或 AEGIS_API_KEYS
_auth_mode = os.getenv("AEGIS_AUTH_MODE", "or")
_auth_whitelist = [e.strip() for e in os.getenv("AEGIS_IP_WHITELIST", "").split(",") if e.strip()]
_auth_keys = set(k.strip() for k in os.getenv("AEGIS_API_KEYS", "").split(",") if k.strip())

if not _auth_whitelist and not _auth_keys:
    if os.getenv("AEGIS_DEV_MODE", "").lower() in ("1", "true", "yes"):
        logger.warning("⚠️ AEGIS_DEV_MODE=true — 所有 API 端点无需认证即可访问！仅限本地开发！")
    else:
        logger.warning(
            "⚠️ 未配置 AEGIS_IP_WHITELIST 或 AEGIS_API_KEYS，认证中间件处于放行模式。"
            "生产部署请配置这些环境变量。开发环境请设置 AEGIS_DEV_MODE=true 以消除此警告。"
        )

app.add_middleware(
    AuthMiddleware,
    whitelist=_auth_whitelist,
    api_keys=_auth_keys,
    public_paths={"/health", "/api/health", "/"},
    public_prefixes=("/static/", "/dashboards/", "/reports/"),
    mode=_auth_mode,
)

# 速率限制中间件
app.add_middleware(
    RateLimitMiddleware,
    default_limit=60,
    window_seconds=60,
    public_paths={"/health", "/api/health"},
    public_prefixes=("/static/", "/dashboards/", "/reports/"),
    path_limits={
        "/api/chat": 30,
        "/api/chat/stream": 15,
    },
)

# Content-Security-Policy 中间件（防御 XSS）
# 设计说明：
# - script-src 包含 'unsafe-inline'：模板使用内联 <script> 块，必需
# - style-src 保留 'unsafe-inline'：模板内联 <style> 块是必要的视觉呈现
# - connect-src 仅允许 Aegis 自身端口，不开放 localhost:* 通配
_aegis_port = os.getenv("AEGIS_PORT", "7860")
_csp_connect = f"http://127.0.0.1:{_aegis_port} http://localhost:{_aegis_port}" if _aegis_port else "'self'"

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com "
            "blob:; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https: blob:; "
            "font-src 'self' data: https://cdn.jsdelivr.net; "
            f"connect-src 'self' {_csp_connect}; "
            "media-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # HSTS：仅在明确开启时启用（生产环境 HTTPS 反向代理后）
        if os.getenv("AEGIS_HSTS", "").lower() in ("1", "true", "yes"):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
    return response

_orchestrator: Optional[Orchestrator] = None
_config: Optional[Config] = None
_init_lock = threading.Lock()


def get_orchestrator() -> Orchestrator:
    global _orchestrator, _config
    if _orchestrator is not None:
        return _orchestrator

    with _init_lock:
        # 双重检查：锁内再次确认未被其他线程初始化
        if _orchestrator is not None:
            return _orchestrator

        print("Initializing Aegis...")
        _config = Config()
        setup_logger("aegis", level=_config.get("system.log_level", "INFO"))

        llm_config = _config.get("llm", {})
        llm = create_llm_provider({
            "provider": llm_config.get("provider", "openai"),
            "model": llm_config.get("model", "deepseek-v4-pro"),
            "api_key": llm_config.get("api_key"),
            "api_base": llm_config.get("api_base"),
            "temperature": llm_config.get("temperature", 0.7),
            "max_tokens": llm_config.get("max_tokens", 4096),
        })

        # 使用完整 MemoryManager（8 个子系统，与 main.py 一致）
        mm = MemoryManager(llm=llm, config=_config.get_all())
        mm.initialize(
            file_store_dir=_config.get("memory.file_store.base_dir", "./data/memory"),
            chroma_dir=_config.get("memory.long_term.persist_dir", "./data/chroma_db"),
            collection_name=_config.get("memory.long_term.collection_name", "aegis_long_term_memory"),
            embedding_model=_config.get("embedding.model", "text-embedding-3-small"),
            short_term_max_tokens=_config.get("memory.short_term.max_tokens", 16000),
            short_term_window=_config.get("memory.short_term.window_size", 20),
        )

        _orchestrator = Orchestrator(llm=llm, memory=mm.retriever, config=_config.get_all())
        _orchestrator.memory_manager = mm  # 启用自动记忆、偏好学习、经验库
        print("Aegis initialized!")
    return _orchestrator


# ===================== Request Models =====================

class ResearchRequest(BaseModel):
    topic: str = Field(..., max_length=500, description="研究主题")
    sources: Optional[List[str]] = ["web", "news"]
    include_swot: bool = True

class ReminderRequest(BaseModel):
    title: str = Field(..., max_length=200, description="提醒标题")
    description: Optional[str] = Field(default="", max_length=2000)
    trigger_time: Optional[str] = Field(default=None, max_length=30)
    trigger_event: Optional[str] = Field(default=None, max_length=200)
    cron_expression: Optional[str] = Field(default=None, max_length=50)

class FileActionRequest(BaseModel):
    filepath: str = Field(..., max_length=500)
    target_lang: Optional[str] = Field(default="zh-CN", max_length=10)
    style: Optional[str] = Field(default="professional", max_length=20)

class MemoryStoreRequest(BaseModel):
    content: str = Field(..., max_length=50000, description="要存储的记忆内容")
    source: Optional[str] = Field(default="web_ui", max_length=50)
    tags: Optional[List[str]] = Field(default=[], max_length=20)

class DispatchRequest(BaseModel):
    todos: List[dict] = Field(..., max_length=100)
    source: Optional[str] = Field(default="web_ui", max_length=50)
    auto_assign: bool = True


# ===================== 页面 =====================

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


# ===================== 核心聊天 =====================

@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
    action: Optional[str] = Form(None),      # translate / polish / generate_ppt / parse
    target_lang: Optional[str] = Form(None), # zh-CN / en / ja ...
    style: Optional[str] = Form(None),       # professional / casual / academic / concise
    session_id: Optional[str] = Form(None),  # 会话隔离 ID
):
    """统一聊天入口 — LLM 自动识别意图并路由到对应 Agent"""
    if len(message) > 50000:
        raise HTTPException(400, "消息过长，最多支持 50000 个字符")
    orch = get_orchestrator()
    filepath = await _save_upload(file)
    session_id = session_id or str(uuid.uuid4())
    if len(session_id) > 64:
        session_id = session_id[:64]

    try:
        # 有文件 + 明确指定操作 → 直接走工作流，不靠 LLM 猜意图
        # audio_transcribe 不走文件处理流 — 有专用的语音转录 pipeline
        if filepath and action and action not in ("parse", "audio_transcribe"):
            wf_result = await asyncio.to_thread(
                run_file_processing_workflow,
                orch, filepath, action, target_lang or "zh-CN", style or "professional"
            )
            return {
                "success": wf_result["status"] != "error",
                "response": _format_file_result(wf_result, action),
                "intent": "file_" + action,
                "session_id": session_id,
            }
        # 否则走 LLM 意图识别 + 路由（推入线程池避免阻塞事件循环）
        result = await asyncio.to_thread(
            orch.process_user_request,
            user_message=message, attached_file=filepath, session_id=session_id,
        )
        return {
            "success": result.get("status") != "error",
            "response": result.get("response", ""),
            "intent": result.get("intent", ""),
            "session_id": session_id,
        }
    except Exception as e:
        logger.exception("Chat processing failed")
        raise HTTPException(status_code=500, detail="处理请求时出错，请稍后重试")


@app.post("/api/chat/stream")
async def chat_stream(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
    action: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """流式聊天 — SSE (Server-Sent Events)，用 asyncio.to_thread 避免事件循环阻塞"""
    if len(message) > 50000:
        raise HTTPException(400, "消息过长，最多支持 50000 个字符")
    orch = get_orchestrator()
    filepath = await _save_upload(file)
    session_id = session_id or str(uuid.uuid4())
    if len(session_id) > 64:
        session_id = session_id[:64]

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_token(token: str):
            loop.call_soon_threadsafe(queue.put_nowait, token)

        def run_sync():
            try:
                if filepath and action and action not in ("parse", "audio_transcribe"):
                    wf_result = run_file_processing_workflow(
                        orch, filepath, action, target_lang or "zh-CN", style or "professional"
                    )
                    text = _format_file_result(wf_result, action)
                    for i in range(0, len(text), 2):
                        on_token(text[i:i+2])
                else:
                    orch.process_user_request(
                        user_message=message, attached_file=filepath,
                        stream_callback=on_token, session_id=session_id,
                    )
            except Exception as e:
                logger.exception("Stream processing failed")
                on_token("\n\n处理请求时出错，请稍后重试")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        try:
            yield f"event: start\ndata: {json.dumps({'session_id': session_id})}\n\n"

            # 推入线程池避免阻塞事件循环 (Issue 2)
            task = asyncio.create_task(asyncio.to_thread(run_sync))

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield f"data: {json.dumps({'token': token})}\n\n"

            await task  # 确保线程完成
            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
        except Exception as e:
            logger.exception("SSE stream failed")
            yield f"event: error\ndata: {json.dumps({'error': '处理请求时出错，请稍后重试'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===================== 文件处理 =====================

@app.post("/api/file/parse")
async def file_parse(file: UploadFile = File(...)):
    """仅解析文件，返回文本预览"""
    filepath = await _save_upload(file)
    if not filepath:
        raise HTTPException(400, "文件保存失败")
    parsed = parse_file(filepath)
    return {
        "success": True,
        "filename": parsed.filename,
        "file_type": parsed.file_type.value,
        "text_preview": parsed.raw_text[:3000],
        "text_length": len(parsed.raw_text),
        "page_count": parsed.page_count,
        "metadata": parsed.metadata,
    }


@app.post("/api/file/translate")
async def file_translate(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    target_lang: str = Form("zh-CN"),
):
    """翻译文件或文本"""
    orch = get_orchestrator()

    if file:
        filepath = await _save_upload(file)
        parsed = parse_file(filepath)
        content = parsed.raw_text
    elif text:
        content = text
    else:
        raise HTTPException(400, "请提供文件或文本")

    translation_tool = TranslationTool(orch.llm)
    result = translation_tool.translate(content, target_lang=target_lang)
    return {
        "success": True,
        "original_length": len(content),
        "translated_text": result.translated_text,
        "target_lang": target_lang,
        "source_lang": result.source_lang_detected,
    }


@app.post("/api/file/polish")
async def file_polish(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    style: str = Form("professional"),
):
    """润色文件或文本"""
    orch = get_orchestrator()

    if file:
        filepath = await _save_upload(file)
        parsed = parse_file(filepath)
        content = parsed.raw_text
    elif text:
        content = text
    else:
        raise HTTPException(400, "请提供文件或文本")

    translation_tool = TranslationTool(orch.llm)
    polished = translation_tool.polish_text(content, style=style)
    return {
        "success": True,
        "original_length": len(content),
        "polished_text": polished,
        "style": style,
    }


@app.post("/api/file/generate-ppt")
async def file_generate_ppt(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    title: str = Form("演示文稿"),
):
    """从文件/文本生成 PPT"""
    orch = get_orchestrator()

    if file:
        filepath = await _save_upload(file)
        parsed = parse_file(filepath)
        content = parsed.raw_text
    elif text:
        content = text
    else:
        raise HTTPException(400, "请提供文件或文本")

    # 用 LLM 分解为幻灯片结构
    prompt = f"""请将以下内容分解为 PPT 幻灯片结构，每页包含一个标题和3-5个要点。
输出 JSON 格式: [{{"title": "页面标题", "bullets": ["要点1", "要点2"]}}, ...]

内容:
---
{content[:5000]}
---"""

    response = orch.llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0.5)
    try:
        raw = response["content"].strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        slides = json.loads(raw)
    except json.JSONDecodeError:
        slides = [{"title": title, "bullets": [line.strip() for line in content.split("\n")[:10] if line.strip()]}]

    safe_title = "".join(c for c in title if c.isalnum() or c in "._- ()（）").strip()[:80] or "演示文稿"
    output_path = f"./output/{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    from src.core.tools.file_tools import generate_pptx
    generate_pptx(title, slides, output_path)

    return {
        "success": True,
        "slides_count": len(slides),
        "output_path": output_path,
        "slides": slides,
    }


# ===================== 语音转录 =====================

@app.post("/api/audio/transcribe")
async def audio_transcribe(file: UploadFile = File(...)):
    """
    上传音频文件，执行说话人分离 + 语音识别 + 标点恢复。

    返回带时间戳和说话人标签的转录结果。
    首次调用会加载模型（约 30-60 秒），后续调用复用。
    """
    if not file.filename:
        raise HTTPException(400, "未提供文件")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
    if ext not in allowed:
        raise HTTPException(400, f"不支持的音频格式: {ext}。支持: {', '.join(sorted(allowed))}")

    # 读取文件头进行魔数校验（防御扩展名伪装）
    header = await file.read(12)
    await file.seek(0)  # 重置以便后续 _save_upload 读取
    if not _verify_audio_magic_bytes(ext, header):
        logger.warning(f"Audio magic byte mismatch: filename={file.filename}, ext={ext}, header={header[:8].hex()}")
        raise HTTPException(400, "文件格式与扩展名不匹配，请上传有效的音频文件")

    filepath = await _save_upload(file)
    logger.info(f"Audio upload saved: {filepath}")

    try:
        from src.core.tools.audio_tools import transcribe_audio, format_transcript_for_display
        result = transcribe_audio(filepath)
        display_text = format_transcript_for_display(result)
    except ImportError as e:
        logger.error(f"Audio pipeline import failed: {e}")
        raise HTTPException(500, "语音转录模块未安装。请确保 speech_pipeline1 路径正确。")
    except Exception as e:
        logger.exception("Audio transcription failed")
        raise HTTPException(500, "语音转录失败，请稍后重试")

    return {
        "success": True,
        "result": result,
        "display_text": display_text,
    }


@app.get("/api/audio/status")
async def audio_status():
    """查询语音模型加载状态"""
    try:
        from src.core.tools.audio_tools import get_audio_model_status
        model_status = get_audio_model_status()
        model_status.pop("success", None)  # 防止覆盖外层 success
        return {"success": True, **model_status}
    except ImportError:
        return {"success": True, "loaded": False, "error": "语音转录模块未安装"}


# ===================== 研究分析 =====================

@app.post("/api/research")
async def research(req: ResearchRequest):
    """研究分析"""
    orch = get_orchestrator()
    result = run_research_workflow(
        orchestrator=orch,
        topic=req.topic,
        sources=req.sources,
        include_swot=req.include_swot,
    )
    return {
        "success": result["status"] == "success",
        "response": result.get("report_markdown", ""),
        "sources_count": result.get("sources_count", 0),
        "previous_research": result.get("previous_research", 0),
    }


# ===================== 提醒 =====================

@app.get("/api/reminders")
async def list_reminders(active_only: bool = False):
    orch = get_orchestrator()
    result = orch.agents["reminder_agent"].execute({"operation": "list", "active_only": active_only})
    reminders = [_serialize_reminder(r) for r in result.get("reminders", [])]
    return {"success": True, "reminders": reminders, "count": len(reminders)}


@app.post("/api/reminders")
async def create_reminder(req: ReminderRequest):
    orch = get_orchestrator()
    result = orch.agents["reminder_agent"].execute({
        "operation": "set",
        "title": req.title,
        "description": req.description,
        "trigger_time": req.trigger_time,
        "trigger_event": req.trigger_event,
        "cron_expression": req.cron_expression,
        "notify_method": ["console"],
    })
    return {"success": True, "message": result.get("message", "已创建"), "reminder_id": result.get("reminder_id")}


@app.delete("/api/reminders/{reminder_id}")
async def cancel_reminder(reminder_id: str):
    orch = get_orchestrator()
    ok = orch.agents["reminder_agent"].calendar.cancel_reminder(reminder_id)
    return {"success": ok, "message": "已取消" if ok else "未找到该提醒"}


@app.post("/api/reminders/{reminder_id}/acknowledge")
async def acknowledge_reminder(reminder_id: str):
    """确认提醒 — 用户点击"知道了"，停止重复提醒，并返回下一个待办任务"""
    orch = get_orchestrator()
    ok = orch.agents["reminder_agent"].calendar.acknowledge_reminder(reminder_id)

    # 查找下一个待办任务并发送桌面通知
    next_task = None
    try:
        active_tasks = orch.task_store.get_active_group_tasks()
        pending = [t for t in active_tasks if t.status == "pending"]
        if pending:
            next_task = pending[0]
            # 发送桌面通知（失败不影响主流程）
            try:
                orch.agents["reminder_agent"].calendar.send_simple_toast(
                    "Aegis — 下一步", f"待办: {next_task.title}"
                )
            except Exception:
                logger.debug("Toast notification failed for next task", exc_info=True)
    except Exception:
        logger.debug("Next-task lookup failed in acknowledge", exc_info=True)

    return {
        "success": ok,
        "message": "已确认" if ok else "未找到该提醒",
        "next_task": {
            "id": next_task.id,
            "title": next_task.title,
            "priority": next_task.priority,
        } if next_task else None,
    }


@app.post("/api/reminders/{reminder_id}/snooze")
async def snooze_reminder(reminder_id: str, minutes: int = 5):
    """延迟提醒 — 用户点击"稍后提醒"，过几分钟再响"""
    minutes = max(1, min(minutes, 1440))  # 限制 1 分钟 ~ 24 小时
    orch = get_orchestrator()
    ok = orch.agents["reminder_agent"].calendar.snooze_reminder(reminder_id, minutes)
    return {"success": ok, "message": f"将在 {minutes} 分钟后再次提醒" if ok else "未找到该提醒"}


@app.post("/api/reminders/followup")
async def check_followup():
    """检查需要跟进的事项"""
    orch = get_orchestrator()
    result = run_reminder_followup_workflow(orchestrator=orch)
    return {
        "success": True,
        "summary": result.get("summary", ""),
        "due_count": result.get("due_reminders_count", 0),
        "followup_count": result.get("followup_count", 0),
    }


@app.post("/api/reminders/clear-all")
async def clear_all_reminders():
    """清除所有已过期/已确认/不活跃的提醒（批量清理测试遗留数据）"""
    orch = get_orchestrator()
    cal = orch.agents["reminder_agent"].calendar
    removed = cal.clear_all_reminders()
    return {"success": True, "removed": removed, "message": f"已清除 {removed} 个提醒"}


# ===================== 任务 =====================

@app.get("/api/tasks")
async def list_tasks():
    orch = get_orchestrator()
    # 获取提醒 + 跟进
    reminders = orch.agents["reminder_agent"].execute({"operation": "list", "active_only": True})
    followup = orch.agents["reminder_agent"].execute({"operation": "followup"})

    return {
        "success": True,
        "active_reminders": [_serialize_reminder(r) for r in reminders.get("reminders", [])],
        "followup_suggestions": followup.get("followup_suggestions", []),
        "summary": orch.execute_task_inquiry("").get("response", ""),
    }


@app.post("/api/tasks/dispatch")
async def dispatch_tasks(req: DispatchRequest):
    """手动分派任务"""
    orch = get_orchestrator()
    result = orch.agents["task_dispatcher"].execute({
        "todos": req.todos,
        "source": req.source,
        "auto_assign": req.auto_assign,
    })
    # 为分派的任务创建提醒
    if result.get("assigned_todos"):
        orch.agents["reminder_agent"].create_reminders_from_todos(result["assigned_todos"])

    return {
        "success": True,
        "assigned": [_serialize(t) for t in result.get("assigned_todos", [])],
        "unassigned": result.get("unassigned", []),
        "summary": result.get("summary", ""),
    }


# ===================== 记忆 =====================

@app.post("/api/memory/search")
async def search_memory(query: str = Form(...)):
    if len(query) > 2000:
        raise HTTPException(400, "查询过长，最多支持 2000 个字符")
    orch = get_orchestrator()
    result = orch.agents["memory_agent"].execute({"operation": "retrieve", "query": query, "top_k": 5})
    memories = []
    for m in result.get("relevant_memories", []):
        memories.append({
            "content": getattr(m, 'content', str(m)),
            "source": getattr(m, 'source', ''),
            "tags": getattr(m, 'tags', []),
        })
    return {"success": True, "memories": memories, "context": result.get("context", "")}


@app.post("/api/memory/store")
async def store_memory(req: MemoryStoreRequest):
    orch = get_orchestrator()
    result = orch.agents["memory_agent"].execute({
        "operation": "store",
        "content": req.content,
        "source": req.source,
        "tags": req.tags,
    })
    return {"success": True, "stored_count": result.get("stored_count", 0)}


@app.post("/api/memory/summarize")
async def summarize_conversation(session_id: Optional[str] = Form(None)):
    """总结当前对话（支持会话隔离）"""
    orch = get_orchestrator()

    # 优先使用会话专属短期记忆
    stm = None
    session_mem = orch.get_existing_session_memory(session_id) if session_id else None
    if session_mem:
        stm = session_mem.short_term
    elif hasattr(orch, 'memory') and orch.memory:
        stm = orch.memory.short_term

    if stm:
        turns = stm.get_context()
        if turns:
            text = "\n".join(f"[{t.role}]: {t.content}" for t in turns)
            result = orch.agents["memory_agent"].execute({"operation": "summarize", "content": text})
            return {"success": True, "summary": result.get("summary", ""), "turns": len(turns)}
    return {"success": False, "message": "暂无对话历史"}


# ===================== 简报 =====================

@app.post("/api/briefing")
async def briefing():
    """生成早晨简报"""
    orch = get_orchestrator()
    result = run_morning_briefing(orchestrator=orch)
    return {"success": True, "briefing": result.get("briefing", ""), "active_reminders": result.get("active_reminders", 0)}


# ===================== 历史会话 =====================

@app.get("/api/history")
async def list_conversation_history(days: int = 30):
    """列出最近 N 天的会话记录（摘要）"""
    import os as _os
    data_dir = _os.environ.get("AEGIS_DATA_DIR", "./data")
    conv_dir = Path(data_dir) / "conversations"
    if not conv_dir.exists():
        return {"success": True, "dates": [], "total_conversations": 0}

    cutoff = datetime.now().date()
    date_list = []
    total = 0

    for f in sorted(conv_dir.glob("*.md"), reverse=True):
        try:
            date_str = f.stem
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if (cutoff - file_date).days > days:
                continue

            text = f.read_text(encoding="utf-8")
            # 计数对话轮数（每个 ## 标题是一轮）
            turn_count = text.count("\n## ")
            # 提取第一行作为标题
            first_line = text.split("\n")[0].replace("# ", "") if text else date_str

            date_list.append({
                "date": date_str,
                "title": first_line,
                "turn_count": turn_count,
                "preview": text[:200] + ("..." if len(text) > 200 else ""),
            })
            total += turn_count
        except (ValueError, OSError):
            continue

    return {
        "success": True,
        "dates": date_list[:days],
        "total_conversations": total,
    }


@app.get("/api/history/{date}")
async def get_conversation_history(date: str):
    """获取指定日期的完整会话内容"""
    # 路径穿越防护
    safe_date = Path(date).name
    if safe_date != date or not safe_date:
        raise HTTPException(400, "无效的日期格式")

    conv_file = Path("./data/conversations") / f"{safe_date}.md"
    if not conv_file.exists():
        raise HTTPException(404, f"未找到 {date} 的会话记录")

    content = conv_file.read_text(encoding="utf-8")
    return {
        "success": True,
        "date": date,
        "content": content,
        "turn_count": content.count("\n## "),
    }


@app.delete("/api/history")
async def clear_conversation_history(before_date: Optional[str] = None):
    """清除会话记录（可指定删除某日期之前的）"""
    conv_dir = Path("./data/conversations")
    if not conv_dir.exists():
        return {"success": True, "deleted": 0}

    deleted = 0
    if before_date:
        try:
            cutoff = datetime.strptime(Path(before_date).name, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "无效的日期格式，请使用 YYYY-MM-DD")
        for f in conv_dir.glob("*.md"):
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d").date()
                if file_date < cutoff:
                    f.unlink()
                    deleted += 1
            except (ValueError, OSError):
                pass
    else:
        for f in conv_dir.glob("*.md"):
            f.unlink()
            deleted += 1

    return {"success": True, "deleted": deleted, "message": f"已删除 {deleted} 个会话文件"}


# ===================== 系统 =====================

@app.get("/api/health")
async def health():
    orch = get_orchestrator()
    agents = list(orch.agents.keys()) if orch else []
    mem_stats = {}
    if orch and hasattr(orch, 'memory_manager') and orch.memory_manager:
        try:
            mem_stats = orch.memory_manager.get_stats()
        except Exception:
            logger.debug("Failed to get memory stats for health check", exc_info=True)
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "agents": agents,
        "memory": {
            "short_term_turns": mem_stats.get("short_term", {}).get("turns", 0),
            "long_term_vectors": mem_stats.get("long_term", {}).get("total_vectors", 0),
        },
    }


@app.get("/api/health/stats")
async def health_stats():
    """详细指标端点：LLM 用量、延迟、错误率、意图分布"""
    try:
        orch = get_orchestrator()
        from src.utils.metrics import MetricsCollector

        stats = MetricsCollector().get_stats()

        # 合并 Agent 和 Memory 状态
        if orch:
            stats["agents"] = list(orch.agents.keys())
            if hasattr(orch, 'memory_manager') and orch.memory_manager:
                try:
                    mem_stats = orch.memory_manager.get_stats()
                    stats["memory"] = {
                        "short_term_turns": mem_stats.get("short_term", {}).get("turns", 0),
                        "long_term_vectors": mem_stats.get("long_term", {}).get("total_vectors", 0),
                        "file_memories": mem_stats.get("file_store", {}).get("total_memories", 0),
                        "experiences": mem_stats.get("experiences", {}).get("total", 0),
                    }
                except Exception:
                    stats["memory"] = {"error": "failed to retrieve memory stats"}

        return stats
    except Exception as e:
        logger.error("Failed to retrieve health stats: %s", e)
        return {
            "status": "error",
            "message": "Failed to retrieve metrics",
            "timestamp": datetime.now().isoformat(),
        }

@app.get("/api/agents/status")
async def agent_status():
    orch = get_orchestrator()
    return {
        "agents": list(orch.agents.keys()),
        "messages_processed": len(orch.message_bus.history),
        "mcp_services": [],
    }


# ===================== 文件验证工具 =====================

def _verify_audio_magic_bytes(ext: str, header: bytes) -> bool:
    """
    校验音频文件魔数（magic bytes），防御扩展名伪装攻击。

    支持的格式及其魔数:
      - .wav:  "RIFF" at 0, "WAVE" at 8
      - .mp3:  0xFF 0xFB/0xF3/0xF2 at 0, or "ID3" at 0
      - .flac: "fLaC" at 0
      - .m4a/.aac: "ftyp" at 4 (ISO base media)
      - .ogg:  "OggS" at 0
      - .wma:  0x30 0x26 0xB2 0x75 (ASF header)
    """
    if len(header) < 4:
        return False

    h = header  # shortcut

    if ext == ".wav":
        return h[:4] == b"RIFF" and len(h) >= 12 and h[8:12] == b"WAVE"
    elif ext == ".mp3":
        if h[:3] == b"ID3":
            return True
        if len(h) >= 2:
            return h[0] == 0xFF and (h[1] & 0xE0) == 0xE0
        return False
    elif ext == ".flac":
        return h[:4] == b"fLaC"
    elif ext in (".m4a", ".aac"):
        return len(h) >= 8 and h[4:8] == b"ftyp"
    elif ext == ".ogg":
        return h[:4] == b"OggS"
    elif ext == ".wma":
        return len(h) >= 4 and h[:4] == b"\x30\x26\xb2\x75"
    else:
        return True  # 未知扩展名已在扩展名检查阶段拒绝


# ===================== Dashboard 看板文件服务 =====================

def _safe_serve_path(base_dir: str, fname: str, allowed_exts: tuple = (".html",)) -> Path:
    """
    安全解析文件服务路径，防止路径穿越攻击。

    仅提取 fname 的基本文件名，拒绝包含路径分隔符或上级引用的输入。
    解析后验证最终路径仍在 base_dir 内。
    防护范围：字面量 ../、URL 编码 (%2e%2e%2f)、双重编码、空字节注入、Unicode 斜线。
    """
    import os as _os
    from urllib.parse import unquote

    # 双重解码以防御 URL 编码绕过攻击
    decoded = fname
    for _ in range(3):  # 最多解码 3 层以防御双重/三重编码
        prev = decoded
        decoded = unquote(decoded)
        if decoded == prev:
            break

    # 检查原始和已解码形式中的路径穿越模式
    for check_str in (fname, decoded):
        check_lower = check_str.lower()
        if ".." in check_str or "/" in check_str or "\\" in check_str:
            raise HTTPException(status_code=400, detail="无效的文件名")
        # 防御空字节注入
        if "\x00" in check_str:
            raise HTTPException(status_code=400, detail="无效的文件名")
        # 防御 Unicode 斜线 (U+2215 ∕, U+FF0F ／, U+2044 ⁄)
        if any(ch in check_str for ch in ("∕", "／", "⁄")):
            raise HTTPException(status_code=400, detail="无效的文件名")

    # 仅提取基本文件名（去掉任何可能残留的路径成分）
    safe_name = _os.path.basename(decoded)
    if not safe_name or safe_name != decoded:
        raise HTTPException(status_code=400, detail="无效的文件名")

    # 检查文件扩展名
    if allowed_exts and not safe_name.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="不支持的文件类型")

    base = Path(base_dir).resolve()
    file_path = (base / safe_name).resolve()

    # 确保解析后的路径仍在 base_dir 内
    if not str(file_path).startswith(str(base)):
        raise HTTPException(status_code=400, detail="无效的文件名")

    return file_path


@app.get("/dashboards/{fname}")
async def serve_dashboard(fname: str):
    """提供生成的 Dashboard HTML 文件"""
    dashboard_path = _safe_serve_path("./data/dashboards", fname)
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))


@app.get("/api/dashboards")
async def list_dashboards():
    """列出所有已生成的 Dashboard"""
    dash_dir = Path("./data/dashboards")
    if not dash_dir.exists():
        return {"dashboards": []}
    files = sorted(dash_dir.glob("dashboard_*.html"), key=lambda f: f.stat().st_mtime, reverse=True)
    return {
        "dashboards": [
            {
                "name": f.name,
                "url": f"/dashboards/{f.name}",
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in files[:20]
        ]
    }


# ===================== 分析报告文件服务 =====================

@app.get("/reports/{fname}")
async def serve_report(fname: str):
    """提供生成的分析报告 HTML 文件"""
    report_path = _safe_serve_path("./data/reports", fname)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return HTMLResponse(report_path.read_text(encoding="utf-8"))


@app.get("/api/workload")
async def check_workload(date: Optional[str] = None):
    """
    负荷感知 — 分析当日任务负载，超载时给出排期建议。

    查询参数:
      - date: 要检查的日期 (YYYY-MM-DD)，默认今天
    """
    try:
        orch = get_orchestrator()
        agent = orch.agents.get("reminder_agent")
        if not agent:
            return {"status": "error", "message": "ReminderAgent 未初始化"}

        task_input = {"operation": "workload"}
        if date:
            task_input["target_date"] = date

        result = agent.execute(task_input)
        return result
    except Exception as e:
        logger.exception("Workload check failed")
        return {"status": "error", "message": "负荷检查失败，请稍后重试"}


@app.get("/api/reports")
async def list_reports():
    """列出所有已生成的分析报告"""
    reports_dir = Path("./data/reports")
    if not reports_dir.exists():
        return {"reports": []}
    files = sorted(reports_dir.glob("analysis_*.html"), key=lambda f: f.stat().st_mtime, reverse=True)
    recent = files[:20]
    return {
        "total": len(files),
        "reports": [
            {
                "name": f.name,
                "url": f"/reports/{f.name}",
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in recent
        ]
    }


# ===================== 辅助 =====================

async def _save_upload(file: Optional[UploadFile], max_size_mb: int = 50) -> Optional[str]:
    """保存上传文件，基于 SHA-256 去重。

    相同内容的文件只保留一份，后续上传直接复用已有文件路径。
    索引文件 .file_index.json 记录 hash → path 映射，O(1) 查找。
    """
    if not file or not file.filename:
        return None

    # 文件类型白名单验证（防恶意文件上传）
    _allowed = os.getenv("AEGIS_ALLOWED_EXTENSIONS", ".csv,.xlsx,.xls,.txt,.pdf,.docx,.pptx,.md,.json,.xml,.yaml,.yml,.py,.js,.ts,.html,.htm,.jpg,.jpeg,.png,.gif,.webp,.svg,.wav,.mp3,.mp4,.webm,.zip,.log,.toml,.cfg,.ini,.env.example")
    allowed_exts = {e.strip().lower() for e in _allowed.split(",") if e.strip()}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"不支持的文件类型: {ext}。允许的类型: {', '.join(sorted(allowed_exts))}")

    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(413, f"文件过大，最大支持 {max_size_mb}MB")
    file_hash = hashlib.sha256(content).hexdigest()

    # 检查索引中是否已有相同 hash 的文件（加锁防止并发竞态）
    index_path = upload_dir / ".file_index.json"

    with _save_upload_lock:
        index = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        if file_hash in index:
            existing_path = Path(index[file_hash]["path"])
            if existing_path.exists():
                logger.info(f"重复文件，复用已有: {existing_path.name}")
                return str(existing_path.absolute())

        # 新文件：用 hash 前缀命名，防碰撞且可读
        safe_name = Path(file.filename).name
        unique_name = f"{file_hash[:16]}_{safe_name}"
        filepath = upload_dir / unique_name
        filepath.write_bytes(content)

        # 更新索引（原子写入：先写临时文件再 replace）
        index[file_hash] = {
            "path": str(filepath.absolute()),
            "original_name": safe_name,
            "size": len(content),
            "saved_at": datetime.now().isoformat(),
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix=".file_index_", dir=str(upload_dir)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(index_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    return str(filepath.absolute())


def _serialize(obj):
    if obj is None: return None
    if isinstance(obj, dict): return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_serialize(v) for v in obj]
    if hasattr(obj, 'model_dump'): return obj.model_dump()
    if hasattr(obj, '__dict__'): return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj)


def _serialize_reminder(r) -> dict:
    def _to_iso(val):
        """安全转换 datetime/字符串 为 ISO 格式字符串"""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, str):
            return val
        return str(val)

    trigger_time = getattr(r, 'trigger_time', None)
    last_triggered = getattr(r, 'last_triggered', None)

    return {
        "id": getattr(r, 'id', ''),
        "title": getattr(r, 'title', str(r)),
        "description": getattr(r, 'description', ''),
        "type": getattr(r, 'type', None),
        "trigger_time": _to_iso(trigger_time),
        "trigger_event": getattr(r, 'trigger_event', None),
        "is_active": getattr(r, 'is_active', True),
        "acknowledged": getattr(r, 'acknowledged', False),
        "snooze_minutes": getattr(r, 'snooze_minutes', 5),
        "fire_count": getattr(r, 'fire_count', 0),
        "max_fires": getattr(r, 'max_fires', 5),
        "last_triggered": _to_iso(last_triggered),
    }


def _format_file_result(wf_result: dict, action: str) -> str:
    """格式化文件处理结果，翻译/润色显示完整文本而非截断预览"""
    full_text = wf_result.get("file_result", {}).get("content_preview", "")
    is_text_action = action in ("translate", "polish")

    parts = [f"✅ 文件处理完成: {wf_result.get('filename', '')}"]
    parts.append(f"操作: {action}")
    parts.append(f"提取待办: {wf_result.get('todos_count', 0)} 个")
    parts.append(f"已分派: {wf_result.get('assigned_count', 0)} 个")
    parts.append(f"已创建提醒: {wf_result.get('reminders_created', 0)} 个")

    if full_text:
        if is_text_action:
            parts.append(f"\n{full_text}")
        elif len(full_text) > 2000:
            parts.append(f"\n内容预览:\n{full_text[:2000]}...")
        else:
            parts.append(f"\n内容预览:\n{full_text}")
    return "\n".join(parts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
