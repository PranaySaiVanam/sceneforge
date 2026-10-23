"""SceneForge production server (FastAPI).

    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sceneforge import api

app = FastAPI(title="SceneForge AI", version="1.0.0",
              description="Natural-language previsualization built on real motion-capture data.")
app.add_middleware(GZipMiddleware, minimum_size=1024)


class DirectRequest(BaseModel):
    prompt: str
    provider: str = "auto"


@app.get("/api/health")
def health():
    return api.health()


@app.get("/api/providers")
def providers():
    return api.providers()


@app.get("/api/library")
def library():
    return api.library()


@app.post("/api/direct")
def direct(req: DirectRequest):
    status, body = api.direct_scene(req.model_dump())
    if status != 200:
        raise HTTPException(status_code=status, detail=body["error"])
    return JSONResponse(body)


@app.get("/motions/{clip_id}.json")
def motion(clip_id: str):
    path = api.motion_path(clip_id)
    if path is None:
        raise HTTPException(status_code=404, detail="unknown clip")
    return FileResponse(path, media_type="application/json",
                        headers={"cache-control": "public, max-age=86400"})


app.mount("/", StaticFiles(directory=api.FRONTEND, html=True), name="frontend")
