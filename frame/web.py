"""Local browser interface backed by the existing P-Helper services."""

from __future__ import annotations

import os
import secrets
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import Config
from .errors import safe_error
from .indexing import qa_root
from .manifest import load_manifest, utc_now
from .pipeline import run_batch
from .qa import answer_question
from .qa_index import QAIndex
from .sessions import SessionStore


MAX_UPLOAD_FILES = 100
MAX_UPLOAD_BYTES = 1024 ** 3
UPLOAD_CHUNK_SIZE = 1024 ** 2


class PathJobRequest(BaseModel):
    input: str = Field(min_length=1)
    language: str = "zh"
    workers: int = Field(default=1, ge=1, le=16)
    force: bool = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None
    papers: list[str] = Field(default_factory=list)
    language: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


@dataclass
class Job:
    id: str
    source: str
    status: str
    created_at: str
    updated_at: str
    exit_code: int | None = None
    error: str | None = None


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phelper-web")

    def submit(self, source: str, work, cleanup=None) -> Job:
        now = utc_now()
        job = Job(uuid.uuid4().hex[:12], source, "queued", now, now)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id, work, cleanup)
        return job

    def _run(self, job_id: str, work, cleanup):
        self._update(job_id, status="running")
        try:
            code = work()
            self._update(job_id, status="success" if code == 0 else "failed",
                         exit_code=code,
                         error=None if code == 0 else "部分论文处理失败，请查看论文库状态")
        except Exception as exc:
            self._update(job_id, status="failed", exit_code=1,
                         error=safe_error(exc, os.environ.get("P_HELPER_API_KEY", "")))
        finally:
            if cleanup is not None:
                cleanup()

    def _update(self, job_id: str, **values):
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)
            job.updated_at = utc_now()

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [asdict(job) for job in jobs[:20]]


def create_app(output_dir: Path) -> FastAPI:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    static_dir = Path(__file__).with_name("web_static")
    app = FastAPI(title="P-Helper", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.output_dir = output_dir
    app.state.csrf_token = secrets.token_urlsafe(24)
    app.state.jobs = JobManager()

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; "
            "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    def require_token(token: str | None):
        if not token or not secrets.compare_digest(token, app.state.csrf_token):
            raise HTTPException(status_code=403, detail="页面令牌无效，请刷新页面")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/app.css")
    def css():
        return FileResponse(static_dir / "app.css", media_type="text/css")

    @app.get("/app.js")
    def javascript():
        return FileResponse(static_dir / "app.js", media_type="text/javascript")

    @app.get("/lucide.min.js")
    def lucide():
        return FileResponse(static_dir / "lucide.min.js", media_type="text/javascript")

    @app.get("/api/bootstrap")
    def bootstrap():
        return {
            "csrf_token": app.state.csrf_token,
            "configured": _configuration_status(),
            "output_dir": str(output_dir),
        }

    @app.get("/api/library")
    def library():
        return {"papers": _library(output_dir), "jobs": app.state.jobs.list()}

    @app.get("/api/papers/{fingerprint}/summary")
    def paper_summary(fingerprint: str):
        manifest = load_manifest(output_dir / "manifest.json")
        entry = manifest.papers.get(fingerprint)
        if entry is None or entry.status != "success" or not entry.output_path:
            raise HTTPException(status_code=404, detail="摘要不存在")
        path = _artifact_path(output_dir, entry.output_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="摘要文件不存在")
        return {"markdown": path.read_text(encoding="utf-8")}

    @app.get("/artifacts/{artifact_path:path}")
    def artifact(artifact_path: str):
        path = _artifact_path(output_dir, artifact_path, allowed_root="images")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(path)

    @app.post("/api/jobs/path", status_code=202)
    def start_path_job(payload: PathJobRequest,
                       x_p_helper_token: str | None = Header(default=None)):
        require_token(x_p_helper_token)
        config = _model_config(output_dir, payload.language, payload.workers, payload.force)
        input_path = Path(payload.input).expanduser().resolve()
        if not input_path.exists():
            raise HTTPException(status_code=400, detail="输入路径不存在")
        job = app.state.jobs.submit(str(input_path), lambda: run_batch(config, input_path))
        return asdict(job)

    @app.post("/api/jobs/upload", status_code=202)
    async def start_upload_job(
            files: list[UploadFile] = File(...), language: str = Form("zh"),
            workers: int = Form(1), force: bool = Form(False),
            x_p_helper_token: str | None = Header(default=None)):
        require_token(x_p_helper_token)
        if language not in {"zh", "en"} or not 1 <= workers <= 16:
            raise HTTPException(status_code=400, detail="任务参数无效")
        if not files or len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(status_code=400, detail=f"每批最多上传 {MAX_UPLOAD_FILES} 个 PDF")
        config = _model_config(output_dir, language, workers, force)
        upload_dir = output_dir / ".phelper" / "uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True)
        total = 0
        try:
            for index, upload in enumerate(files, 1):
                name = Path(upload.filename or f"paper-{index}.pdf").name
                if Path(name).suffix.lower() != ".pdf":
                    raise HTTPException(status_code=400, detail="只能上传 PDF 文件")
                target = upload_dir / f"{index:03d}-{name}"
                with target.open("wb") as handle:
                    while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                        total += len(chunk)
                        if total > MAX_UPLOAD_BYTES:
                            raise HTTPException(status_code=413, detail="单批上传总大小不能超过 1 GB")
                        handle.write(chunk)
        except Exception:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        finally:
            for upload in files:
                await upload.close()
        job = app.state.jobs.submit(
            f"上传的 {len(files)} 篇论文", lambda: run_batch(config, upload_dir),
            cleanup=lambda: shutil.rmtree(upload_dir, ignore_errors=True),
        )
        return asdict(job)

    @app.post("/api/ask")
    def ask(payload: AskRequest, x_p_helper_token: str | None = Header(default=None)):
        require_token(x_p_helper_token)
        return _ask(output_dir, payload)

    @app.get("/api/sessions")
    def sessions():
        path = qa_root(output_dir) / "sessions.sqlite3"
        if not path.exists():
            return {"sessions": []}
        with SessionStore(path) as store:
            return {"sessions": [asdict(item) for item in store.list()]}

    @app.get("/api/sessions/{session_id}/turns")
    def session_turns(session_id: str):
        path = qa_root(output_dir) / "sessions.sqlite3"
        if not path.exists():
            raise HTTPException(status_code=404, detail="会话不存在")
        with SessionStore(path) as store:
            session = store.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            return {"session": asdict(session), "turns": store.recent_turns(session_id, 1000)}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str,
                       x_p_helper_token: str | None = Header(default=None)):
        require_token(x_p_helper_token)
        path = qa_root(output_dir) / "sessions.sqlite3"
        if not path.exists():
            raise HTTPException(status_code=404, detail="会话不存在")
        with SessionStore(path) as store:
            if not store.delete(session_id):
                raise HTTPException(status_code=404, detail="会话不存在")
        return {"deleted": session_id}

    return app


def _configuration_status() -> dict[str, bool]:
    return {
        "api_key": bool(os.environ.get("P_HELPER_API_KEY")),
        "api_url": bool(os.environ.get("P_HELPER_API_URL")),
        "model": bool(os.environ.get("P_HELPER_MODEL")),
    }


def _model_config(output_dir: Path, language: str, workers: int = 1,
                  force: bool = False) -> Config:
    values = {
        "api_key": os.environ.get("P_HELPER_API_KEY", ""),
        "api_url": os.environ.get("P_HELPER_API_URL", ""),
        "model": os.environ.get("P_HELPER_MODEL", ""),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise HTTPException(status_code=400, detail="模型配置不完整，请检查 .env")
    return Config(**values, output_dir=str(output_dir), language=language,
                  workers=workers, force=force)


def _allowed_fingerprints(output_dir: Path, index: QAIndex) -> set[str]:
    manifest = load_manifest(output_dir / "manifest.json")
    indexed = {paper.fingerprint: paper for paper in index.list_papers()}
    return {
        fingerprint for fingerprint, entry in manifest.papers.items()
        if entry.index_status == "success" and entry.index_signature
        and fingerprint in indexed
        and indexed[fingerprint].signature == entry.index_signature
    }


def _ask(output_dir: Path, payload: AskRequest) -> dict:
    index_path = qa_root(output_dir) / "index.sqlite3"
    if not index_path.exists():
        raise HTTPException(status_code=400, detail="尚未建立问答索引")
    config = _model_config(output_dir, payload.language or "zh")
    sessions_path = qa_root(output_dir) / "sessions.sqlite3"
    created = False
    try:
        with QAIndex(index_path) as index, SessionStore(sessions_path) as sessions:
            allowed = _allowed_fingerprints(output_dir, index)
            if not allowed:
                raise HTTPException(status_code=400, detail="当前没有可问答的论文")
            selected = index.resolve_papers(payload.papers, allowed) if payload.papers else None
            if payload.session_id:
                session = sessions.get(payload.session_id)
                if session is None:
                    raise HTTPException(status_code=404, detail="会话不存在")
                if not set(session.fingerprints).issubset(allowed):
                    raise HTTPException(status_code=409, detail="会话中的论文已不可用")
                if payload.language and payload.language != session.language:
                    raise HTTPException(status_code=409, detail="语言与已有会话冲突")
                if payload.top_k and payload.top_k != session.top_k:
                    raise HTTPException(status_code=409, detail="检索数量与已有会话冲突")
                if selected is not None and set(selected) != set(session.fingerprints):
                    raise HTTPException(status_code=409, detail="论文范围与已有会话冲突")
            else:
                fingerprints = selected if selected is not None else index.resolve_papers([], allowed)
                session = sessions.create(payload.language or "zh", fingerprints,
                                          payload.top_k or 8)
                created = True
            session_config = Config(**{**config.__dict__, "language": session.language})
            answer, evidence = answer_question(
                session_config, payload.question,
                sessions.recent_turns(session.id), index,
                session.fingerprints, session.top_k,
            )
            sessions.add_turn(session.id, payload.question, answer)
            return {
                "session_id": session.id,
                "answer": answer,
                "evidence": [{"id": item.id, "fingerprint": item.fingerprint,
                              "source_path": item.source_path, "page": item.page}
                             for item in evidence],
            }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=safe_error(exc)) from exc
    except Exception as exc:
        if created:
            with SessionStore(sessions_path) as sessions:
                if sessions.turn_count(session.id) == 0:
                    sessions.delete(session.id)
        raise HTTPException(status_code=500, detail="问答处理失败，请稍后重试") from exc


def _library(output_dir: Path) -> list[dict]:
    manifest = load_manifest(output_dir / "manifest.json")
    papers = []
    for entry in sorted(manifest.papers.values(), key=lambda item: item.source_path):
        image = next(iter(sorted((output_dir / "images" / entry.fingerprint).glob("figure-01.*"))), None)
        papers.append({
            "fingerprint": entry.fingerprint,
            "source_path": entry.source_path,
            "aliases": entry.aliases,
            "summary_status": entry.status,
            "index_status": entry.index_status,
            "stage": entry.stage,
            "error": entry.error,
            "index_error": entry.index_error,
            "processed_at": entry.processed_at,
            "image_url": (f"/artifacts/{image.relative_to(output_dir).as_posix()}"
                          if image else None),
        })
    return papers


def _artifact_path(output_dir: Path, value: str, allowed_root: str | None = None) -> Path:
    path = (output_dir / value).resolve()
    try:
        relative = path.relative_to(output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="文件不存在") from exc
    if allowed_root is not None and (not relative.parts or relative.parts[0] != allowed_root):
        raise HTTPException(status_code=404, detail="文件不存在")
    return path


def run_web(output_dir: Path, port: int):
    import uvicorn
    uvicorn.run(create_app(output_dir), host="127.0.0.1", port=port, log_level="info")
