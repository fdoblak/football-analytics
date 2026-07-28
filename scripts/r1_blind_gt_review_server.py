#!/usr/bin/env python3
"""R1-F1 blind GT review server (localhost only).

Canonical storage: source xyxy on 1336x744. Canvas is letterboxed; boxes
round-trip through football_analytics.annotation.coordinates.

Blind mode hides YOLO / old GT / tracker / roles. Predictions only after freeze
via --audit-predictions (off by default).
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from football_analytics.annotation.coordinates import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    make_source_bbox,
    validate_source_bbox_xyxy,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = Path(
    "/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
)
DEFAULT_SESSION = Path("/home/fdoblak/workspace/own_video_analysis/r1_blind_gt/review_session")
DEFAULT_SELECTION = Path(
    "/home/fdoblak/workspace/own_video_analysis/r1_blind_gt/frame_selection.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class ReviewState:
    def __init__(
        self,
        video: Path,
        selection: Path,
        session_dir: Path,
        *,
        blind: bool = True,
        audit_predictions: Path | None = None,
    ) -> None:
        self.video = video
        self.session_dir = session_dir
        self.blind = blind
        self.audit_predictions = audit_predictions
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open video: {video}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_video = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width != SOURCE_WIDTH or self.height != SOURCE_HEIGHT:
            raise SystemExit(
                f"unexpected source size {self.width}x{self.height}; "
                f"expected {SOURCE_WIDTH}x{SOURCE_HEIGHT}"
            )
        sel = json.loads(selection.read_text(encoding="utf-8"))
        self.frames = list(sel["frames"])
        self.index = 0
        self.lock = threading.Lock()
        self.undo: list[dict[str, Any]] = []
        self.redo: list[dict[str, Any]] = []
        self.ann_path = session_dir / "session_annotations.json"
        self.session = self._load_or_init()
        self.pred_by: dict[int, list[list[float]]] = {}
        if audit_predictions and audit_predictions.is_file() and not blind:
            payload = json.loads(audit_predictions.read_text(encoding="utf-8"))
            for fr in payload.get("frames", []):
                self.pred_by[int(fr["frame_idx"])] = [
                    list(map(float, p["bbox_xyxy"])) for p in fr.get("predictions", [])
                ]

    def _load_or_init(self) -> dict[str, Any]:
        if self.ann_path.is_file():
            return json.loads(self.ann_path.read_text(encoding="utf-8"))
        frames = []
        for item in self.frames:
            frames.append(
                {
                    "frame_idx": item["frame_idx"],
                    "t_s": item["t_s"],
                    "split": item["split"],
                    "review_status": "not_reviewed",
                    "completed": False,
                    "humans": [],
                    "source_width": SOURCE_WIDTH,
                    "source_height": SOURCE_HEIGHT,
                }
            )
        session = {
            "schema": "r1_blind_gt_review_session_v1",
            "dataset_id": "own_video_human_blind_gt_v1",
            "provenance": "human_or_agent_review_session",
            "blind_mode": self.blind,
            "coordinate_space": "source_xyxy_px_v1",
            "video": str(self.video),
            "frames": frames,
            "updated_at_utc": utc_now(),
        }
        atomic_write_json(self.ann_path, session)
        return session

    def save(self) -> None:
        self.session["updated_at_utc"] = utc_now()
        atomic_write_json(self.ann_path, self.session)

    def current(self) -> dict[str, Any]:
        return self.session["frames"][self.index]

    def read_frame(self, frame_idx: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_idx}")
        return frame


HTML = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>R1 Blind GT Review</title>
<style>
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#111;color:#eee;display:flex;height:100vh}
#side{width:340px;padding:12px;overflow:auto;background:#1a1a1a}
#main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center}
canvas{background:#000;cursor:crosshair;max-width:96%;max-height:86vh}
button,select{margin:3px 2px;padding:6px 8px}
.row{margin:6px 0}
.badge{display:inline-block;padding:2px 6px;border-radius:4px;background:#333}
.warn{color:#ffd36a}
</style></head><body>
<div id="side">
  <h2>R1 Blind GT</h2>
  <div class="warn">Blind mode: prediction/YOLO/eski GT gizli</div>
  <div class="row">Frame <span id="meta" class="badge"></span></div>
  <div class="row">Source: 1336×744 · xyxy · top-left</div>
  <div class="row">
    <button onclick="nav(-1)">Prev</button>
    <button onclick="nav(1)">Next</button>
    <button onclick="undo()">Undo</button>
    <button onclick="redo()">Redo</button>
  </div>
  <div class="row">
    <label>Mode
      <select id="mode">
        <option value="draw">draw</option>
        <option value="on_pitch">on_pitch</option>
        <option value="ignore">ignore</option>
        <option value="uncertain">uncertain</option>
        <option value="difficult">difficult</option>
      </select>
    </label>
  </div>
  <div class="row">
    <button onclick="delSel()">Delete selected</button>
    <button onclick="markDone()">Mark frame complete</button>
  </div>
  <div class="row"><button onclick="save()">Save</button> <span id="status"></span></div>
  <div class="row">Zoom: wheel · Pan: middle-drag</div>
  <ol id="list"></ol>
</div>
<div id="main"><canvas id="c"></canvas></div>
<script>
let state=null, img=new Image(), boxes=[], sel=-1;
let drag=null, view={scale:1,x:0,y:0};
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const SOURCE_W=1336, SOURCE_H=744;

function letterbox(cw,ch){
  const scale=Math.min(cw/SOURCE_W, ch/SOURCE_H);
  const content_w=SOURCE_W*scale, content_h=SOURCE_H*scale;
  const pad_x=(cw-content_w)/2, pad_y=(ch-content_h)/2;
  return {scale,pad_x,pad_y,content_w,content_h,cw,ch};
}
function srcToCanvas(b,t){
  return [t.pad_x+b[0]*t.scale, t.pad_y+b[1]*t.scale, t.pad_x+b[2]*t.scale, t.pad_y+b[3]*t.scale];
}
function canvasToSrc(b,t){
  return [(b[0]-t.pad_x)/t.scale,(b[1]-t.pad_y)/t.scale,(b[2]-t.pad_x)/t.scale,(b[3]-t.pad_y)/t.scale];
}
function transform(){ return letterbox(canvas.width, canvas.height); }

async function load(){
  const r=await fetch('/api/state'); state=await r.json();
  document.getElementById('meta').textContent=
    `idx=${state.index}/${state.n_frames-1} f=${state.frame.frame_idx} t=${state.frame.t_s}s ${state.frame.split} ${state.frame.review_status}`;
  boxes=state.frame.humans.map(h=>[h.x1,h.y1,h.x2,h.y2,h.ignore||false,h.uncertain||false,h.difficult||false]);
  img.src='/api/frame.jpg?ts='+Date.now();
  img.onload=()=>{fit(); draw(); renderList();};
}
function fit(){
  const maxW=window.innerWidth-380, maxH=window.innerHeight-40;
  const s=Math.min(maxW/SOURCE_W, maxH/SOURCE_H);
  canvas.width=Math.round(SOURCE_W*s); canvas.height=Math.round(SOURCE_H*s);
}
function draw(){
  const t=transform();
  ctx.fillStyle='#000'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,t.pad_x,t.pad_y,t.content_w,t.content_h);
  boxes.forEach((b,i)=>{
    const c=srcToCanvas(b,t);
    ctx.strokeStyle=b[4]?'#ff0':(b[5]?'#fa0':'#0f0');
    ctx.lineWidth=(i===sel)?3:2;
    ctx.strokeRect(c[0],c[1],c[2]-c[0],c[3]-c[1]);
    ctx.fillStyle=ctx.strokeStyle; ctx.font='12px sans-serif';
    ctx.fillText(b[4]?'IGNORE':(b[5]?'UNC':'HUMAN'), c[0], Math.max(12,c[1]-4));
  });
  if(state.predictions && state.predictions.length){
    // audit only
    state.predictions.forEach(p=>{
      const c=srcToCanvas(p,t);
      ctx.strokeStyle='#08f'; ctx.setLineDash([4,3]);
      ctx.strokeRect(c[0],c[1],c[2]-c[0],c[3]-c[1]);
      ctx.setLineDash([]);
    });
  }
}
function renderList(){
  const ol=document.getElementById('list'); ol.innerHTML='';
  boxes.forEach((b,i)=>{
    const li=document.createElement('li');
    li.textContent=`#${i} ${b.map(x=>Math.round(x*10)/10).slice(0,4).join(',')}`;
    li.onclick=()=>{sel=i; draw();};
    ol.appendChild(li);
  });
}
function canvasPos(ev){
  const r=canvas.getBoundingClientRect();
  return [ (ev.clientX-r.left)*(canvas.width/r.width), (ev.clientY-r.top)*(canvas.height/r.height) ];
}
canvas.onmousedown=(ev)=>{
  const [x,y]=canvasPos(ev);
  if(ev.button===1){ drag={pan:true,x,y}; return; }
  if(document.getElementById('mode').value==='draw'){
    drag={x1:x,y1:y,x2:x,y2:y}; return;
  }
  // select hit
  const t=transform();
  sel=-1;
  boxes.forEach((b,i)=>{
    const c=srcToCanvas(b,t);
    if(x>=c[0]&&x<=c[2]&&y>=c[1]&&y<=c[3]) sel=i;
  });
  draw();
};
canvas.onmousemove=(ev)=>{
  if(!drag) return;
  const [x,y]=canvasPos(ev);
  if(drag.pan){ drag=null; return; }
  drag.x2=x; drag.y2=y; draw();
  const t=transform();
  ctx.strokeStyle='#0ff'; ctx.strokeRect(drag.x1,drag.y1,drag.x2-drag.x1,drag.y2-drag.y1);
};
canvas.onmouseup=async (ev)=>{
  if(!drag||drag.pan){ drag=null; return; }
  const t=transform();
  let b=canvasToSrc([Math.min(drag.x1,drag.x2),Math.min(drag.y1,drag.y2),Math.max(drag.x1,drag.x2),Math.max(drag.y1,drag.y2)], t);
  drag=null;
  if((b[2]-b[0])<4||(b[3]-b[1])<4) return;
  const mode=document.getElementById('mode').value;
  const human={x1:b[0],y1:b[1],x2:b[2],y2:b[3],ignore:mode==='ignore',uncertain:mode==='uncertain',difficult:mode==='difficult',on_pitch:mode!=='ignore'};
  await post('/api/add_box', human); await load();
};
async function post(url, body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  const j=await r.json(); document.getElementById('status').textContent=j.status||r.status; return j;
}
async function nav(d){ await post('/api/nav',{delta:d}); await load(); }
async function undo(){ await post('/api/undo',{}); await load(); }
async function redo(){ await post('/api/redo',{}); await load(); }
async function delSel(){ if(sel<0) return; await post('/api/delete_box',{index:sel}); sel=-1; await load(); }
async function markDone(){ await post('/api/complete',{}); await load(); }
async function save(){ await post('/api/save',{}); }
window.onresize=()=>{fit(); draw();};
load();
</script></body></html>
"""


def build_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _bytes(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._bytes(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                with state.lock:
                    fr = state.current()
                    payload = {
                        "index": state.index,
                        "n_frames": len(state.frames),
                        "frame": fr,
                        "blind": state.blind,
                        "source_width": SOURCE_WIDTH,
                        "source_height": SOURCE_HEIGHT,
                        "predictions": (
                            [] if state.blind else state.pred_by.get(int(fr["frame_idx"]), [])
                        ),
                    }
                self._json(200, payload)
                return
            if path == "/api/frame.jpg":
                with state.lock:
                    fr = state.current()
                    frame = state.read_frame(int(fr["frame_idx"]))
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ok:
                    self._json(500, {"status": "encode_fail"})
                    return
                self._bytes(200, buf.tobytes(), "image/jpeg")
                return
            self._json(404, {"status": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"status": "bad_json"})
                return
            path = urlparse(self.path).path
            with state.lock:
                if path == "/api/nav":
                    state.index = int(
                        max(0, min(len(state.frames) - 1, state.index + int(body.get("delta", 0))))
                    )
                    self._json(200, {"status": "ok", "index": state.index})
                    return
                if path == "/api/add_box":
                    fr = state.current()
                    snapshot = json.loads(json.dumps(fr))
                    state.undo.append({"frame_idx": fr["frame_idx"], "frame": snapshot})
                    state.redo.clear()
                    try:
                        box = validate_source_bbox_xyxy(
                            [body["x1"], body["y1"], body["x2"], body["y2"]]
                        )
                        bb = make_source_bbox(
                            frame_index=int(fr["frame_idx"]),
                            fps=state.fps,
                            bbox_xyxy=box,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._json(400, {"status": f"reject:{exc}"})
                        return
                    human = bb.to_dict()
                    human.update(
                        {
                            "ignore": bool(body.get("ignore")),
                            "uncertain": bool(body.get("uncertain")),
                            "difficult": bool(body.get("difficult")),
                            "on_pitch": not bool(body.get("ignore")),
                            "class": "human_on_pitch",
                            "occlusion": "none",
                            "truncated": box[0] <= 2
                            or box[1] <= 2
                            or box[2] >= SOURCE_WIDTH - 2
                            or box[3] >= SOURCE_HEIGHT - 2,
                            "visible_fraction": float(body.get("visible_fraction", 1.0)),
                            "size": "medium",
                        }
                    )
                    fr["humans"].append(human)
                    fr["review_status"] = "in_progress"
                    state.save()
                    self._json(200, {"status": "added"})
                    return
                if path == "/api/delete_box":
                    fr = state.current()
                    idx = int(body.get("index", -1))
                    if 0 <= idx < len(fr["humans"]):
                        state.undo.append(
                            {"frame_idx": fr["frame_idx"], "frame": json.loads(json.dumps(fr))}
                        )
                        fr["humans"].pop(idx)
                        state.save()
                    self._json(200, {"status": "ok"})
                    return
                if path == "/api/complete":
                    fr = state.current()
                    fr["completed"] = True
                    fr["review_status"] = "blind_reviewed"
                    state.save()
                    self._json(200, {"status": "completed"})
                    return
                if path == "/api/undo":
                    if state.undo:
                        item = state.undo.pop()
                        # find frame
                        for i, fr in enumerate(state.session["frames"]):
                            if fr["frame_idx"] == item["frame_idx"]:
                                state.redo.append(
                                    {
                                        "frame_idx": fr["frame_idx"],
                                        "frame": json.loads(json.dumps(fr)),
                                    }
                                )
                                state.session["frames"][i] = item["frame"]
                                state.index = i
                                break
                        state.save()
                    self._json(200, {"status": "ok"})
                    return
                if path == "/api/redo":
                    if state.redo:
                        item = state.redo.pop()
                        for i, fr in enumerate(state.session["frames"]):
                            if fr["frame_idx"] == item["frame_idx"]:
                                state.undo.append(
                                    {
                                        "frame_idx": fr["frame_idx"],
                                        "frame": json.loads(json.dumps(fr)),
                                    }
                                )
                                state.session["frames"][i] = item["frame"]
                                state.index = i
                                break
                        state.save()
                    self._json(200, {"status": "ok"})
                    return
                if path == "/api/save":
                    state.save()
                    self._json(200, {"status": "saved"})
                    return
            self._json(404, {"status": "not_found"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--blind", action="store_true", default=True)
    p.add_argument("--no-blind", action="store_true")
    p.add_argument("--audit-predictions", type=Path, default=None)
    args = p.parse_args(argv)
    blind = not args.no_blind
    if blind and args.audit_predictions:
        raise SystemExit("refusing audit predictions while blind mode is on")
    state = ReviewState(
        args.video,
        args.selection,
        args.session_dir,
        blind=blind,
        audit_predictions=args.audit_predictions,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    print(f"R1 blind GT review: http://{args.host}:{args.port}/  (Ctrl+C stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        state.save()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
