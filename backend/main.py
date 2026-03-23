from __future__ import annotations

import json
import queue
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.schemas import (
    CoverageModel,
    HealthModel,
    IndexStatsModel,
    JumpRequestModel,
    JumpResponseModel,
    ReindexRequestModel,
    ReindexStateModel,
    SearchResponseModel,
    StatusResponseModel,
    TimelineOptionModel,
)
from backend.services.index_store import IndexStore
from backend.services.indexing import IndexingService, ReindexCoordinator
from backend.services.resolve_api import ResolveConnectionError, ResolveFacade
from backend.services.search import SearchService
from backend.services.thumbnails import ThumbnailService


settings = get_settings()
store = IndexStore(settings.index_db_path)
resolve = ResolveFacade()
indexing_service = IndexingService(settings=settings, store=store, resolve=resolve)
reindexer = ReindexCoordinator(indexing_service)
search_service = SearchService(store)
thumbnail_service = ThumbnailService(settings=settings, store=store)

app = FastAPI(title="Resolve Clip Search", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    store.initialize()


@app.get("/api/health", response_model=HealthModel)
def health() -> HealthModel:
    return HealthModel()


@app.get("/api/status", response_model=StatusResponseModel)
def status() -> StatusResponseModel:
    reindex_state = reindexer.snapshot()
    resolve_status = (
        resolve.get_cached_status()
        if reindex_state.running
        else resolve.get_status()
    )
    stats = store.get_stats()
    is_stale = False
    project_coverage = None
    current_timeline_coverage = None
    timeline_options: list[TimelineOptionModel] = []

    if resolve_status.connected and not reindex_state.running:
        try:
            signature = resolve.compute_project_signature()
            if stats["project_name"]:
                is_stale = signature["signature_hash"] != stats["signature_hash"]
            if stats["project_name"] and signature["project_name"] != stats["project_name"]:
                is_stale = True

            indexed_project_uid = (
                stats.get("project_uid")
                if stats.get("project_uid") == signature["project_uid"]
                else None
            )
            indexed_coverage = store.get_indexed_coverage(indexed_project_uid)

            project_total = int(signature["clip_count"])
            project_indexed = int(indexed_coverage["project_indexed"])
            project_missing = max(project_total - project_indexed, 0)
            project_coverage = CoverageModel(
                label="Project",
                indexed=project_indexed,
                total=project_total,
                missing=project_missing,
                complete=project_total > 0 and project_missing == 0,
            )

            current_timeline = next(
                (
                    timeline
                    for timeline in signature["timelines"]
                    if timeline["timeline_uid"] == resolve_status.current_timeline_uid
                ),
                None,
            )
            if not current_timeline and resolve_status.current_timeline_name:
                current_timeline = next(
                    (
                        timeline
                        for timeline in signature["timelines"]
                        if timeline["timeline_name"] == resolve_status.current_timeline_name
                    ),
                    None,
                )
            if current_timeline:
                current_total = int(current_timeline["clip_count"])
                current_indexed = int(
                    indexed_coverage["timeline_counts"].get(current_timeline["timeline_uid"])
                    or indexed_coverage["timeline_name_counts"].get(current_timeline["timeline_name"])
                    or 0
                )
                current_missing = max(current_total - current_indexed, 0)
                current_timeline_coverage = CoverageModel(
                    label=current_timeline["timeline_name"],
                    indexed=current_indexed,
                    total=current_total,
                    missing=current_missing,
                    complete=current_total > 0 and current_missing == 0,
                )

            timeline_options = [
                TimelineOptionModel(
                    timeline_name=timeline["timeline_name"],
                    timeline_uid=timeline["timeline_uid"],
                    total=int(timeline["clip_count"]),
                    indexed=int(
                        indexed_coverage["timeline_counts"].get(timeline["timeline_uid"])
                        or indexed_coverage["timeline_name_counts"].get(timeline["timeline_name"])
                        or 0
                    ),
                    missing=max(
                        int(timeline["clip_count"])
                        - int(
                            indexed_coverage["timeline_counts"].get(timeline["timeline_uid"])
                            or indexed_coverage["timeline_name_counts"].get(timeline["timeline_name"])
                            or 0
                        ),
                        0,
                    ),
                    complete=(
                        int(timeline["clip_count"]) > 0
                        and max(
                            int(timeline["clip_count"])
                            - int(
                                indexed_coverage["timeline_counts"].get(timeline["timeline_uid"])
                                or indexed_coverage["timeline_name_counts"].get(timeline["timeline_name"])
                                or 0
                            ),
                            0,
                        )
                        == 0
                    ),
                    current=(
                        timeline["timeline_uid"] == resolve_status.current_timeline_uid
                        or timeline["timeline_name"] == resolve_status.current_timeline_name
                    ),
                )
                for timeline in signature["timelines"]
            ]
        except Exception:
            is_stale = False

    if not timeline_options and stats.get("project_uid"):
        timeline_options = [
            TimelineOptionModel(
                timeline_name=row["timeline_name"],
                timeline_uid=row.get("timeline_uid"),
                total=int(row["clip_count"]),
                indexed=int(row["clip_count"]),
                missing=0,
                complete=int(row["clip_count"]) > 0,
                current=(
                    row.get("timeline_uid") == resolve_status.current_timeline_uid
                    or row["timeline_name"] == resolve_status.current_timeline_name
                ),
            )
            for row in store.get_indexed_timelines(stats["project_uid"])
        ]

    index_model = IndexStatsModel(
        project_name=stats["project_name"],
        total=stats["total"],
        timelines=stats["timelines"],
        last_indexed=stats["last_indexed"],
        available_types=search_service.build_filter_options(),
        is_stale=is_stale,
        quick_mode=stats["quick_mode"],
        storage_format=stats["storage_format"],
        storage_path=stats["storage_path"],
        loaded_from_disk=stats["loaded_from_disk"],
        project_coverage=project_coverage,
        current_timeline_coverage=current_timeline_coverage,
        timeline_options=timeline_options,
    )

    return StatusResponseModel(
        connected=resolve_status.connected,
        message=resolve_status.message,
        project_name=resolve_status.project_name,
        current_timeline=resolve_status.current_timeline_name,
        current_timeline_uid=resolve_status.current_timeline_uid,
        current_page=resolve_status.current_page,
        index=index_model,
        reindex=ReindexStateModel(**reindex_state.to_dict()),
    )


@app.get("/api/search", response_model=SearchResponseModel)
def search(
    q: str = Query("", alias="q"),
    clip_type: str = Query("All"),
    scope: str = Query("all"),
    limit: int | None = Query(None, ge=1),
) -> SearchResponseModel:
    search_scope = scope.lower()
    if search_scope not in {"all", "current"}:
        raise HTTPException(status_code=400, detail='Search scope must be "all" or "current".')

    timeline_uid = None
    timeline_name = None
    if search_scope == "current":
        resolve_status = resolve.get_status()
        if not resolve_status.connected:
            raise HTTPException(status_code=503, detail=resolve_status.message)
        if not resolve_status.current_timeline_name:
            raise HTTPException(status_code=400, detail="Resolve does not currently have an active timeline.")
        timeline_uid = resolve_status.current_timeline_uid
        timeline_name = resolve_status.current_timeline_name

    payload = search_service.search(
        q,
        clip_type=clip_type,
        scope=search_scope,
        timeline_uid=timeline_uid,
        timeline_name=timeline_name,
        limit=limit,
    )
    return SearchResponseModel(**payload)


@app.post("/api/reindex", response_model=ReindexStateModel)
def reindex(request: ReindexRequestModel) -> ReindexStateModel:
    quick_mode = (
        request.quick_mode
        if request.quick_mode is not None
        else settings.default_quick_mode
    )
    resolve_status = resolve.get_status()
    if not resolve_status.connected:
        raise HTTPException(status_code=503, detail=resolve_status.message)
    try:
        state = reindexer.start(
            timeline_uids=request.timeline_uids,
            timeline_names=request.timeline_names,
            quick_mode=quick_mode,
        )
    except ResolveConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ReindexStateModel(**state.to_dict())


@app.post("/api/reindex/cancel", response_model=ReindexStateModel)
def cancel_reindex() -> ReindexStateModel:
    state = reindexer.cancel()
    return ReindexStateModel(**state.to_dict())


@app.get("/api/reindex/stream", response_model=None)
def reindex_stream() -> StreamingResponse:
    listener = reindexer.subscribe()

    def event_stream():
        try:
            while True:
                try:
                    payload = listener.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            reindexer.unsubscribe(listener)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/jump", response_model=JumpResponseModel)
def jump(request: JumpRequestModel) -> JumpResponseModel:
    clip = store.get_clip(request.clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found in the local index.")

    try:
        if reindexer.snapshot().running:
            result = reindexer.request_jump(clip)
        else:
            result = resolve.jump_to_clip(
                clip_id=clip["clip_id"],
                timeline_uid=clip["timeline_uid"],
                timeline_name=clip["timeline_name"],
                start_timecode=clip["start_timecode"],
                clip_name=clip["clip_name"],
                file_path=clip["file_path"],
                duration_frames=clip["duration_frames"],
                track_index=clip["track"],
            )
    except ResolveConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JumpResponseModel(
        clip_id=request.clip_id,
        timeline_name=result["timeline_name"],
        start_timecode=result["start_timecode"],
    )


@app.get("/api/clips/{clip_id}/thumbnail", response_model=None)
def thumbnail(clip_id: str) -> Response:
    image_bytes = thumbnail_service.get_or_create_thumbnail(clip_id)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Thumbnail unavailable for this clip.")
    return Response(content=image_bytes, media_type="image/jpeg")


frontend_dir = settings.frontend_dist_dir
frontend_index = frontend_dir / "index.html"


if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")


@app.get("/", include_in_schema=False, response_model=None)
def frontend_root() -> FileResponse | HTMLResponse:
    if frontend_index.exists():
        return FileResponse(frontend_index)
    return HTMLResponse(
        """
        <html>
          <body style="font-family: sans-serif; padding: 32px;">
            <h1>Resolve Clip Search</h1>
            <p>Frontend not built yet. Run <code>npm install</code> and <code>npm run build</code> in <code>frontend/</code>, then start the FastAPI server.</p>
          </body>
        </html>
        """
    )
