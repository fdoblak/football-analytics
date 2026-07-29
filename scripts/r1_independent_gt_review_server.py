#!/usr/bin/env python3
"""R1-F2-A independent football human GT review server (localhost only).

Train: YOLO11n-hybrid proposals shown as editable non-GT seeds.
Dev/holdout: fully blind — no predictions, no confidence, no auto boxes.

Canonical storage: source xyxy 1336×744. Runtime drafts stay outside Git.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2

from football_analytics.annotation.coordinates import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    CoordinateError,
)
from football_analytics.annotation.independent_gt import (
    DEFAULT_RUNTIME,
    DEFAULT_VIDEO,
    EXPECTED_SOURCE_SHA256,
    IndependentGTError,
    append_audit_line,
    assert_no_prediction_leakage,
    atomic_write_json,
    sha256_file,
    soft_box_warnings,
    utc_now,
    validate_box_geometry,
    validate_freeze_ready,
    validate_metadata,
)

REPO = Path(__file__).resolve().parents[1]


class ReviewApp:
    def __init__(self, runtime: Path, video: Path) -> None:
        self.runtime = runtime
        self.video = video
        digest = sha256_file(video)
        if digest != EXPECTED_SOURCE_SHA256:
            raise SystemExit(f"SOURCE_SHA_MISMATCH:{digest}")
        self.source_sha = digest
        self.draft_path = runtime / "draft_annotations.json"
        self.progress_path = runtime / "progress.json"
        self.session_path = runtime / "session_state.json"
        self.audit_path = runtime / "review_audit.jsonl"
        self.cache_dir = runtime / "frame_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w != SOURCE_WIDTH or h != SOURCE_HEIGHT:
            raise SystemExit(f"bad size {w}x{h}")
        self.lock = threading.Lock()
        self.undo: list[dict[str, Any]] = []
        self.redo: list[dict[str, Any]] = []
        if not self.draft_path.is_file():
            raise SystemExit("missing draft — run prepare_r1_f2a_independent_gt.py first")
        self.draft = json.loads(self.draft_path.read_text(encoding="utf-8"))
        self.index = 0
        if self.session_path.is_file():
            sess = json.loads(self.session_path.read_text(encoding="utf-8"))
            self.index = int(sess.get("index", 0)) % max(1, len(self.draft["frames"]))
        self._audit({"event": "server_start", "index": self.index})

    def _audit(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event["ts"] = utc_now()
        event.setdefault("index", self.index)
        if self.draft["frames"]:
            fr = self.draft["frames"][self.index]
            event.setdefault("frame_idx", fr.get("frame_idx"))
            event.setdefault("split", fr.get("split"))
        append_audit_line(self.audit_path, event)

    def save(self) -> None:
        self.draft["updated_at_utc"] = utc_now()
        self.draft["human_approved"] = False
        self.draft["reviewed_gt"] = False
        self.draft["frozen"] = False
        self.draft["acceptance_eligible"] = False
        atomic_write_json(self.draft_path, self.draft)
        # progress
        by = {
            "train": {"n": 0, "complete": 0},
            "dev": {"n": 0, "complete": 0},
            "holdout": {"n": 0, "complete": 0},
        }
        n_complete = 0
        for fr in self.draft["frames"]:
            sp = fr["split"]
            by[sp]["n"] += 1
            if fr.get("completed"):
                by[sp]["complete"] += 1
                n_complete += 1
        atomic_write_json(
            self.progress_path,
            {
                "schema": "independent_gt_progress_v1",
                "n_frames": len(self.draft["frames"]),
                "n_complete": n_complete,
                "by_split": by,
                "updated_at_utc": utc_now(),
            },
        )
        atomic_write_json(
            self.session_path,
            {
                "schema": "independent_gt_session_state_v1",
                "index": self.index,
                "video": str(self.video),
                "source_sha256": self.source_sha,
                "runtime": str(self.runtime),
                "updated_at_utc": utc_now(),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.draft))

    def push_undo(self) -> None:
        self.undo.append(self.snapshot())
        self.redo.clear()
        if len(self.undo) > 80:
            self.undo = self.undo[-80:]

    def current(self) -> dict[str, Any]:
        return self.draft["frames"][self.index]

    def read_frame_jpeg(self, frame_idx: int) -> bytes:
        cached = self.cache_dir / f"{frame_idx:06d}.jpg"
        if cached.is_file():
            return cached.read_bytes()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"read fail {frame_idx}")
        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok2:
            raise RuntimeError("jpeg encode fail")
        data = buf.tobytes()
        cached.write_bytes(data)
        with contextlib.suppress(OSError):
            cached.chmod(0o600)
        return data

    def public_state(self) -> dict[str, Any]:
        fr = self.current()
        split = fr["split"]
        # Hard guarantee: never send proposals/predictions on dev/holdout
        assert_no_prediction_leakage(fr, split=split)
        proposals = []
        if split == "train":
            proposals = list(fr.get("proposals") or [])
        prog = json.loads(self.progress_path.read_text(encoding="utf-8"))
        return {
            "index": self.index,
            "n_frames": len(self.draft["frames"]),
            "frame_idx": fr["frame_idx"],
            "t_s": fr["t_s"],
            "split": split,
            "categories": fr.get("categories") or [],
            "completed": bool(fr.get("completed")),
            "humans": list(fr.get("humans") or []),
            "proposals": proposals,
            "progress": prog,
            "source_wh": [SOURCE_WIDTH, SOURCE_HEIGHT],
            "coordinate_space": "source_xyxy_px_v1",
            "blind": split in {"dev", "holdout"},
            "policy": {
                "class": "human",
                "train_proposals_are_not_gt": True,
                "dev_holdout_blind": True,
                "freeze_requires_explicit_user_approval": True,
            },
        }


HTML = r"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>R1 Independent Human GT</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#0e0e10;color:#eee;display:flex;height:100vh;overflow:hidden}
#side{width:380px;padding:12px;overflow:auto;background:#16161a;border-right:1px solid #333}
#main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;position:relative;background:#000}
canvas{cursor:crosshair;background:#000;max-width:98%;max-height:90vh}
button,select,input{margin:2px;padding:6px 8px;font-size:13px}
.row{margin:6px 0}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;background:#2a2a32;margin-right:4px}
.warn{color:#ffd36a}.err{color:#ff8a8a}.ok{color:#8dffb0}
.split-train{background:#1e3a5f}.split-dev{background:#3a2a1e}.split-holdout{background:#3a1e2e}
h2{margin:4px 0 8px;font-size:18px}
#list{font-size:12px;max-height:220px;overflow:auto;padding-left:18px}
kbd{background:#333;padding:1px 5px;border-radius:3px;font-size:12px}
</style></head><body>
<div id="side">
  <h2>R1 Bağımsız İnsan GT</h2>
  <div class="warn">Acceptance / freeze / fine-tuning YOK. Yalnız etiketleme.</div>
  <div class="row"><span id="splitBadge" class="badge"></span>
    <span id="meta" class="badge"></span></div>
  <div class="row">İlerleme: <b id="prog"></b></div>
  <div class="row" id="blindNote" class="ok"></div>
  <div class="row">
    <button onclick="nav(-1)">◀ Prev</button>
    <button onclick="nav(1)">Next ▶</button>
    <button onclick="undo()">Undo</button>
    <button onclick="redo()">Redo</button>
  </div>
  <div class="row">
    <label>Role <select id="role">
      <option>player</option><option>goalkeeper</option><option>referee</option>
      <option>staff</option><option>unknown</option>
    </select></label>
    <label>Team <select id="team">
      <option>yellow</option><option>white</option><option>official</option>
      <option>unknown</option><option>not_applicable</option>
    </select></label>
  </div>
  <div class="row">
    <label>Elig <select id="elig">
      <option>on_pitch</option><option>off_pitch</option><option>uncertain</option>
    </select></label>
    <label>Vis <select id="vis">
      <option>clear</option><option>small</option><option>occluded</option>
      <option>truncated</option><option>blurred</option>
    </select></label>
  </div>
  <div class="row">
    <label><input type="checkbox" id="jvis" onchange="togJersey()"/> jersey visible</label>
    <input type="number" id="jnum" min="0" max="99" placeholder="#" disabled style="width:60px"/>
  </div>
  <div class="row">
    <button onclick="applyMeta()">Apply meta → selected</button>
    <button onclick="delSel()">Delete</button>
  </div>
  <div class="row">
    <button onclick="acceptProposal()">Accept proposal → human</button>
    <button onclick="completeNext()" style="background:#2d5a3d;color:#fff">Complete + Next (C)</button>
  </div>
  <div class="row">
    <button onclick="markComplete(false)">Mark complete</button>
    <button onclick="markComplete(true)">Uncomplete</button>
    <button onclick="save()">Save</button>
  </div>
  <div class="row" id="status"></div>
  <div class="row" style="font-size:12px">
    Kısayol: <kbd>Y</kbd><kbd>W</kbd><kbd>R</kbd><kbd>G</kbd><kbd>U</kbd><kbd>O</kbd><kbd>C</kbd>
    · wheel zoom · orta tuş pan · sürükle çiz
  </div>
  <div class="row warn" id="warnBox"></div>
  <ol id="list"></ol>
</div>
<div id="main"><canvas id="c"></canvas></div>
<script>
const SOURCE_W=1336, SOURCE_H=744;
let state=null, img=new Image(), sel=-1, selProp=-1;
let drag=null, pan=null;
let view={scale:1, x:0, y:0};
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');

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
  const x1=(b[0]-t.pad_x)/t.scale, y1=(b[1]-t.pad_y)/t.scale;
  const x2=(b[2]-t.pad_x)/t.scale, y2=(b[3]-t.pad_y)/t.scale;
  return [
    Math.max(0,Math.min(SOURCE_W,x1)), Math.max(0,Math.min(SOURCE_H,y1)),
    Math.max(0,Math.min(SOURCE_W,x2)), Math.max(0,Math.min(SOURCE_H,y2))
  ];
}
function setStatus(m,cls){const e=document.getElementById('status'); e.className=cls||''; e.textContent=m;}
function togJersey(){document.getElementById('jnum').disabled=!document.getElementById('jvis').checked;}

async function api(path, body){
  const r=await fetch(path,{method:body?'POST':'GET',
    headers:body?{'Content-Type':'application/json'}:undefined,
    body:body?JSON.stringify(body):undefined});
  const j=await r.json();
  if(!r.ok) throw new Error(j.error||r.statusText);
  return j;
}

async function load(){
  state=await api('/api/state');
  document.getElementById('splitBadge').textContent=state.split.toUpperCase();
  document.getElementById('splitBadge').className='badge split-'+state.split;
  document.getElementById('meta').textContent=
    `i=${state.index+1}/${state.n_frames} f=${state.frame_idx} t=${state.t_s.toFixed(2)}s`+
    (state.completed?' ✓':'');
  const p=state.progress;
  document.getElementById('prog').textContent=
    `${p.n_complete}/${p.n_frames} · train ${p.by_split.train.complete}/${p.by_split.train.n}`+
    ` · dev ${p.by_split.dev.complete}/${p.by_split.dev.n}`+
    ` · holdout ${p.by_split.holdout.complete}/${p.by_split.holdout.n}`;
  document.getElementById('blindNote').textContent=state.blind
    ? 'DEV/HOLDOUT KÖR: proposal/prediction gizli — sıfırdan çizin'
    : 'TRAIN: mavi kesikli = proposal (GT değil). Kabul/düzelt/sil.';
  sel=-1; selProp=-1;
  img.onload=()=>{fit(); draw();};
  img.src='/api/frame.jpg?frame_idx='+state.frame_idx+'&_='+Date.now();
  renderList();
}

function fit(){
  const maxW=window.innerWidth-400, maxH=window.innerHeight-40;
  const s=Math.min(maxW/SOURCE_W, maxH/SOURCE_H);
  canvas.width=Math.floor(SOURCE_W*s); canvas.height=Math.floor(SOURCE_H*s);
  view={scale:1,x:0,y:0};
}

function transform(){
  // view.scale zooms around center; letterbox on canvas buffer
  const t=letterbox(canvas.width, canvas.height);
  return t;
}

function draw(){
  const t=transform();
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.translate(view.x, view.y);
  ctx.scale(view.scale, view.scale);
  ctx.drawImage(img, t.pad_x, t.pad_y, t.content_w, t.content_h);
  // proposals (train only)
  (state.proposals||[]).forEach((p,i)=>{
    const b=srcToCanvas(p.bbox_xyxy,t);
    ctx.strokeStyle=i===selProp?'#4fc3f7':'#0288d1';
    ctx.setLineDash([6,4]); ctx.lineWidth=2;
    ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
    ctx.setLineDash([]);
  });
  (state.humans||[]).forEach((h,i)=>{
    const b=srcToCanvas(h.bbox_xyxy,t);
    let col='#00e676';
    if(h.eligibility==='off_pitch') col='#9e9e9e';
    if(h.eligibility==='uncertain') col='#ffd54f';
    if(i===sel) col='#ff5252';
    ctx.strokeStyle=col; ctx.lineWidth=i===sel?3:2;
    ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
    ctx.fillStyle=col; ctx.font='12px sans-serif';
    ctx.fillText((h.role||'?')+'/'+(h.team_appearance||'?'), b[0]+2, Math.max(12,b[1]-4));
  });
  if(drag && drag.kind==='draw'){
    ctx.strokeStyle='#fff'; ctx.setLineDash([4,2]);
    const x=Math.min(drag.x0,drag.x1), y=Math.min(drag.y0,drag.y1);
    ctx.strokeRect(x,y,Math.abs(drag.x1-drag.x0),Math.abs(drag.y1-drag.y0));
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function renderList(){
  const ol=document.getElementById('list'); ol.innerHTML='';
  (state.humans||[]).forEach((h,i)=>{
    const li=document.createElement('li');
    li.textContent=`#${i} ${h.role}/${h.team_appearance}/${h.eligibility}/${h.visibility}`;
    li.style.cursor='pointer';
    if(i===sel) li.style.color='#ff8a80';
    li.onclick=()=>{sel=i; selProp=-1; syncMeta(); draw();};
    ol.appendChild(li);
  });
  if(!state.blind){
    (state.proposals||[]).forEach((p,i)=>{
      const li=document.createElement('li');
      li.textContent=`proposal#${i} score=${(p.score||0).toFixed(2)} NOT GT`;
      li.style.color='#4fc3f7'; li.style.cursor='pointer';
      li.onclick=()=>{selProp=i; sel=-1; draw();};
      ol.appendChild(li);
    });
  }
}

function syncMeta(){
  if(sel<0||!state.humans[sel]) return;
  const h=state.humans[sel];
  document.getElementById('role').value=h.role||'unknown';
  document.getElementById('team').value=h.team_appearance||'unknown';
  document.getElementById('elig').value=h.eligibility||'uncertain';
  document.getElementById('vis').value=h.visibility||'clear';
  document.getElementById('jvis').checked=!!h.jersey_number_visible;
  document.getElementById('jnum').value=h.jersey_number==null?'':h.jersey_number;
  togJersey();
}

function pointerSrc(ev){
  const rect=canvas.getBoundingClientRect();
  const sx=(ev.clientX-rect.left)* (canvas.width/rect.width);
  const sy=(ev.clientY-rect.top)* (canvas.height/rect.height);
  const x=(sx-view.x)/view.scale, y=(sy-view.y)/view.scale;
  const t=transform();
  const src=canvasToSrc([x,y,x,y], t);
  return {cx:x, cy:y, sx:src[0], sy:src[1]};
}

canvas.addEventListener('mousedown', ev=>{
  if(ev.button===1 || (ev.button===0 && ev.altKey)){
    pan={x:ev.clientX, y:ev.clientY, vx:view.x, vy:view.y}; ev.preventDefault(); return;
  }
  if(ev.button!==0) return;
  const p=pointerSrc(ev);
  // hit-test humans for move
  const t=transform();
  for(let i=state.humans.length-1;i>=0;i--){
    const b=srcToCanvas(state.humans[i].bbox_xyxy,t);
    if(p.cx>=b[0]&&p.cx<=b[2]&&p.cy>=b[1]&&p.cy<=b[3]){
      sel=i; selProp=-1; syncMeta();
      drag={kind:'move', i, ox:p.sx, oy:p.sy, box:state.humans[i].bbox_xyxy.slice()};
      draw(); return;
    }
  }
  drag={kind:'draw', x0:p.cx, y0:p.cy, x1:p.cx, y1:p.cy};
});
canvas.addEventListener('mousemove', ev=>{
  if(pan){
    view.x=pan.vx+(ev.clientX-pan.x);
    view.y=pan.vy+(ev.clientY-pan.y);
    draw(); return;
  }
  if(!drag) return;
  const p=pointerSrc(ev);
  if(drag.kind==='draw'){ drag.x1=p.cx; drag.y1=p.cy; draw(); }
  if(drag.kind==='move'){
    const dx=p.sx-drag.ox, dy=p.sy-drag.oy;
    const b=drag.box;
    state.humans[drag.i].bbox_xyxy=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy];
    draw();
  }
});
async function endDrag(ev){
  if(pan){ pan=null; return; }
  if(!drag) return;
  const d=drag; drag=null;
  if(d.kind==='draw'){
    const t=transform();
    let b=canvasToSrc([
      Math.min(d.x0,d.x1), Math.min(d.y0,d.y1),
      Math.max(d.x0,d.x1), Math.max(d.y0,d.y1)
    ], t);
    if((b[2]-b[0])<8||(b[3]-b[1])<8){ draw(); return; }
    try{
      const body=metaPayload(); body.bbox_xyxy=b;
      const j=await api('/api/add_box', body);
      state=j.state; sel=state.humans.length-1; renderList(); syncMeta();
      if(j.warnings&&j.warnings.length) document.getElementById('warnBox').textContent=j.warnings.join('; ');
      else document.getElementById('warnBox').textContent='';
      setStatus('box eklendi','ok');
    }catch(e){ setStatus(String(e),'err'); }
  } else if(d.kind==='move'){
    try{
      const j=await api('/api/update_box',{index:d.i, bbox_xyxy:state.humans[d.i].bbox_xyxy});
      state=j.state; renderList();
      setStatus('box taşındı','ok');
    }catch(e){ setStatus(String(e),'err'); await load(); }
  }
  draw();
}
canvas.addEventListener('mouseup', endDrag);
canvas.addEventListener('mouseleave', endDrag);
canvas.addEventListener('wheel', ev=>{
  ev.preventDefault();
  const f=ev.deltaY<0?1.1:0.9;
  view.scale=Math.max(0.5, Math.min(6, view.scale*f));
  draw();
}, {passive:false});
canvas.addEventListener('contextmenu', ev=>ev.preventDefault());

function metaPayload(){
  const jvis=document.getElementById('jvis').checked;
  let jn=null;
  if(jvis){
    const v=document.getElementById('jnum').value;
    if(v!=='') jn=parseInt(v,10);
  }
  return {
    role: document.getElementById('role').value,
    team_appearance: document.getElementById('team').value,
    eligibility: document.getElementById('elig').value,
    visibility: document.getElementById('vis').value,
    jersey_number_visible: jvis,
    jersey_number: jn,
  };
}

async function applyMeta(){
  if(sel<0) return;
  try{
    const j=await api('/api/update_box', Object.assign({index:sel}, metaPayload()));
    state=j.state; renderList(); setStatus('meta güncellendi','ok');
  }catch(e){ setStatus(String(e),'err'); }
}
async function delSel(){
  if(sel<0) return;
  const j=await api('/api/delete_box',{index:sel});
  state=j.state; sel=-1; renderList(); draw(); setStatus('silindi','ok');
}
async function acceptProposal(){
  if(state.blind){ setStatus('dev/holdout kör — proposal yok','err'); return; }
  if(selProp<0){ setStatus('proposal seçin','err'); return; }
  const j=await api('/api/accept_proposal', Object.assign({proposal_index:selProp}, metaPayload()));
  state=j.state; sel=state.humans.length-1; selProp=-1; renderList(); draw();
  setStatus('proposal human-reviewed olarak alındı','ok');
}
async function nav(d){
  const j=await api('/api/nav',{delta:d}); state=j.state; await load();
}
async function markComplete(un){
  const j=await api('/api/complete',{completed: !un});
  state=j.state; await load();
}
async function completeNext(){
  await api('/api/complete',{completed:true});
  await nav(1);
}
async function save(){ await api('/api/save',{}); setStatus('kaydedildi','ok'); }
async function undo(){ const j=await api('/api/undo',{}); state=j.state; await load(); }
async function redo(){ const j=await api('/api/redo',{}); state=j.state; await load(); }

document.addEventListener('keydown', ev=>{
  if(ev.target.tagName==='INPUT'||ev.target.tagName==='SELECT') return;
  const k=ev.key.toLowerCase();
  if(k==='y'){ document.getElementById('role').value='player'; document.getElementById('team').value='yellow'; applyMeta(); }
  if(k==='w'){ document.getElementById('role').value='player'; document.getElementById('team').value='white'; applyMeta(); }
  if(k==='r'){ document.getElementById('role').value='referee'; document.getElementById('team').value='official'; applyMeta(); }
  if(k==='g'){ document.getElementById('role').value='goalkeeper'; document.getElementById('team').value='unknown'; applyMeta(); }
  if(k==='u'){ document.getElementById('role').value='unknown'; document.getElementById('team').value='unknown'; applyMeta(); }
  if(k==='o'){ document.getElementById('elig').value='off_pitch'; applyMeta(); }
  if(k==='c'){ ev.preventDefault(); completeNext(); }
  if(k==='delete'||k==='backspace'){ ev.preventDefault(); delSel(); }
  if(ev.ctrlKey && k==='z'){ ev.preventDefault(); undo(); }
  if(ev.ctrlKey && k==='y'){ ev.preventDefault(); redo(); }
  if(ev.key==='ArrowLeft') nav(-1);
  if(ev.key==='ArrowRight') nav(1);
});
window.addEventListener('resize', ()=>{ fit(); draw(); });
load().catch(e=>setStatus(String(e),'err'));
setInterval(()=>{ api('/api/save',{}).catch(()=>{}); }, 20000);
</script></body></html>
"""


def build_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                # Read-only liveness; no annotation mutation / no predictions / no secrets.
                payload = {
                    "status": "ok",
                    "service": "r1_independent_gt_review",
                    "source_id": "own_video_97b298e4",
                    "source_sha256_ok": app.source_sha == EXPECTED_SOURCE_SHA256,
                    "host": "127.0.0.1",
                }
                self._json(200, payload)
                return
            if path == "/":
                data = HTML.encode("utf-8")
                # Hard fail if CDN-looking strings appear
                if b"cdn." in data or b"http://" in data and b"127.0.0.1" not in data:
                    # allow only relative /api — HTML has no external refs
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/api/state":
                with app.lock:
                    self._json(200, app.public_state())
                return
            if path == "/api/frame.jpg":
                from urllib.parse import parse_qs

                qs = parse_qs(urlparse(self.path).query)
                frame_idx = int(qs.get("frame_idx", ["0"])[0])
                with app.lock:
                    # only allow selected frames
                    allowed = {int(f["frame_idx"]) for f in app.draft["frames"]}
                    if frame_idx not in allowed:
                        self.send_error(403, "frame not in selection")
                        return
                    jpeg = app.read_frame_jpeg(frame_idx)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if path == "/api/freeze_check":
                with app.lock:
                    report = validate_freeze_ready(
                        app.draft,
                        source_sha256=app.source_sha,
                        user_approved=False,
                    )
                self._json(200, report)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            body = self._read_json()
            try:
                with app.lock:
                    if path == "/api/nav":
                        app.index = (app.index + int(body.get("delta", 0))) % len(
                            app.draft["frames"]
                        )
                        app.save()
                        app._audit({"event": "nav", "delta": body.get("delta")})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/save":
                        app.save()
                        app._audit({"event": "autosave"})
                        self._json(200, {"ok": True, "state": app.public_state()})
                        return
                    if path == "/api/complete":
                        fr = app.current()
                        fr["completed"] = bool(body.get("completed", True))
                        fr["review_status"] = "complete" if fr["completed"] else "incomplete"
                        app.save()
                        app._audit({"event": "complete", "completed": fr["completed"]})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/add_box":
                        app.push_undo()
                        fr = app.current()
                        xyxy = body["bbox_xyxy"]
                        validate_box_geometry(xyxy)
                        hum = {
                            "box_id": str(uuid.uuid4()),
                            "bbox_xyxy": [float(x) for x in xyxy],
                            "class_name": "human",
                            "role": body.get("role", "unknown"),
                            "team_appearance": body.get("team_appearance", "unknown"),
                            "eligibility": body.get("eligibility", "uncertain"),
                            "visibility": body.get("visibility", "clear"),
                            "jersey_number_visible": bool(body.get("jersey_number_visible", False)),
                            "jersey_number": body.get("jersey_number", None),
                            "origin": (
                                "manual_blind" if fr["split"] in {"dev", "holdout"} else "manual"
                            ),
                        }
                        errs = validate_metadata(hum)
                        if errs:
                            raise IndependentGTError(";".join(errs))
                        others = [h["bbox_xyxy"] for h in fr.get("humans") or []]
                        warns = soft_box_warnings(xyxy, others)
                        hum["warnings"] = warns
                        fr.setdefault("humans", []).append(hum)
                        if fr["split"] in {"dev", "holdout"}:
                            assert_no_prediction_leakage(fr, split=fr["split"])
                        app.save()
                        app._audit({"event": "add_box", "box": hum})
                        self._json(200, {"state": app.public_state(), "warnings": warns})
                        return
                    if path == "/api/update_box":
                        app.push_undo()
                        fr = app.current()
                        i = int(body["index"])
                        hum = fr["humans"][i]
                        if "bbox_xyxy" in body:
                            validate_box_geometry(body["bbox_xyxy"])
                            hum["bbox_xyxy"] = [float(x) for x in body["bbox_xyxy"]]
                        for k in (
                            "role",
                            "team_appearance",
                            "eligibility",
                            "visibility",
                            "jersey_number_visible",
                            "jersey_number",
                        ):
                            if k in body:
                                hum[k] = body[k]
                        errs = validate_metadata(hum)
                        if errs:
                            raise IndependentGTError(";".join(errs))
                        others = [h["bbox_xyxy"] for j, h in enumerate(fr["humans"]) if j != i]
                        hum["warnings"] = soft_box_warnings(hum["bbox_xyxy"], others)
                        app.save()
                        app._audit({"event": "update_box", "index": i, "box": hum})
                        self._json(
                            200,
                            {"state": app.public_state(), "warnings": hum["warnings"]},
                        )
                        return
                    if path == "/api/delete_box":
                        app.push_undo()
                        fr = app.current()
                        i = int(body["index"])
                        removed = fr["humans"].pop(i)
                        app.save()
                        app._audit({"event": "delete_box", "box": removed})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/accept_proposal":
                        fr = app.current()
                        if fr["split"] != "train":
                            raise IndependentGTError("PROPOSALS_ONLY_ON_TRAIN")
                        app.push_undo()
                        pi = int(body["proposal_index"])
                        prop = fr["proposals"][pi]
                        xyxy = prop["bbox_xyxy"]
                        validate_box_geometry(xyxy)
                        hum = {
                            "box_id": str(uuid.uuid4()),
                            "bbox_xyxy": [float(x) for x in xyxy],
                            "class_name": "human",
                            "role": body.get("role", "unknown"),
                            "team_appearance": body.get("team_appearance", "unknown"),
                            "eligibility": body.get(
                                "eligibility", prop.get("eligibility_hint", "uncertain")
                            ),
                            "visibility": body.get("visibility", "clear"),
                            "jersey_number_visible": bool(body.get("jersey_number_visible", False)),
                            "jersey_number": body.get("jersey_number", None),
                            "origin": "proposal_reviewed",
                            "proposal_id": prop.get("proposal_id"),
                        }
                        errs = validate_metadata(hum)
                        if errs:
                            raise IndependentGTError(";".join(errs))
                        fr.setdefault("humans", []).append(hum)
                        # remove accepted proposal from list so it isn't double-counted
                        fr["proposals"].pop(pi)
                        app.save()
                        app._audit({"event": "accept_proposal", "box": hum})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/undo":
                        if not app.undo:
                            raise IndependentGTError("NOTHING_TO_UNDO")
                        app.redo.append(app.snapshot())
                        app.draft = app.undo.pop()
                        app.save()
                        app._audit({"event": "undo"})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/redo":
                        if not app.redo:
                            raise IndependentGTError("NOTHING_TO_REDO")
                        app.undo.append(app.snapshot())
                        app.draft = app.redo.pop()
                        app.save()
                        app._audit({"event": "redo"})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/freeze":
                        # Always hard-fail in R1-F2-A — freeze not allowed here
                        raise IndependentGTError(
                            "FREEZE_FORBIDDEN_IN_R1_F2_A_REQUIRES_EXPLICIT_LATER_APPROVAL"
                        )
            except (
                IndependentGTError,
                CoordinateError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                self._json(400, {"error": str(exc)})
                return
            self.send_error(404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    args = ap.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("HOST_MUST_BE_LOCALHOST")
    app = ReviewApp(args.runtime, args.video)
    handler = build_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"R1 Independent GT review on http://{args.host}:{args.port}/", flush=True)
    print(f"runtime={args.runtime}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutdown", flush=True)
    finally:
        app.save()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
