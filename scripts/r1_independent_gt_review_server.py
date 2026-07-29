#!/usr/bin/env python3
"""R1-F2-A independent football human GT review server (localhost only).

Train: YOLO11n-hybrid proposals shown as editable non-GT seeds.
Dev/holdout: fully blind — no predictions, no confidence, no auto boxes.

Repair mode (FIX2): navigate only false-complete train frames.
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
from football_analytics.annotation.train_repair import (
    FAILED_TRAIN_FRAME_INDICES,
    PROTECTED_TRAIN_FRAME_INDICES,
    REPAIR_MODE,
    assert_immutable_fingerprints,
    bulk_accept_proposals,
    collect_fingerprints,
    frame_gate_counts,
    reject_pending_proposals,
    set_no_human_confirmed,
    soft_complete_warnings,
    validate_repair_complete,
    validate_train_complete_allowed,
)

REPO = Path(__file__).resolve().parents[1]


class ReviewApp:
    def __init__(
        self,
        runtime: Path,
        video: Path,
        *,
        repair_mode: str | None = None,
        holdout_v2: bool = False,
        active_learning: bool = False,
    ) -> None:
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
        if repair_mode and repair_mode != REPAIR_MODE:
            raise SystemExit(f"UNSUPPORTED_REPAIR_MODE:{repair_mode}")
        if holdout_v2 and repair_mode:
            raise SystemExit("HOLDOUT_V2_AND_REPAIR_MUTUALLY_EXCLUSIVE")
        if active_learning and repair_mode:
            raise SystemExit("ACTIVE_LEARNING_AND_REPAIR_MUTUALLY_EXCLUSIVE")
        self.repair_mode = repair_mode
        self.active_learning = bool(active_learning or self.draft.get("active_learning"))
        # Pure holdout_v2 mode vs dual AL+holdout draft
        self.holdout_v2 = bool(
            holdout_v2 or (self.draft.get("holdout_v2") and not self.active_learning)
        )
        if self.active_learning:
            for fr in self.draft["frames"]:
                if fr.get("section") == "holdout_v2":
                    fr["proposals"] = []
                    fr["rejected_proposals"] = []
                    fr["split"] = "holdout"
            self.draft["active_learning"] = True
            self.draft["holdout_v2"] = True
        elif self.holdout_v2:
            # Blind-only draft: no proposals, no model leakage, annotate from scratch.
            for fr in self.draft["frames"]:
                fr["proposals"] = []
                fr["rejected_proposals"] = []
                fr["split"] = "holdout"
            self.draft["holdout_v2"] = True
            self.draft["blind"] = True
        self.immutable_fp = collect_fingerprints(self.draft)
        self.nav_indices = self._build_nav_indices()
        self.index = 0
        if self.session_path.is_file():
            sess = json.loads(self.session_path.read_text(encoding="utf-8"))
            if self.repair_mode:
                raw = sess.get("repair_nav_pos")
                if raw is None:
                    raw = sess.get("index", 0)
                saved = int(raw if raw is not None else 0)
                self.index = saved % max(1, len(self.nav_indices))
            else:
                self.index = int(sess.get("index", 0) or 0) % max(1, len(self.nav_indices))
        self._audit(
            {
                "event": "server_start",
                "index": self.index,
                "repair_mode": self.repair_mode,
                "holdout_v2": self.holdout_v2,
                "active_learning": self.active_learning,
                "nav_n": len(self.nav_indices),
            }
        )

    def _build_nav_indices(self) -> list[int]:
        frames = self.draft["frames"]
        if not self.repair_mode:
            return list(range(len(frames)))
        failed = set(FAILED_TRAIN_FRAME_INDICES)
        protected = set(PROTECTED_TRAIN_FRAME_INDICES)
        out: list[int] = []
        for i, fr in enumerate(frames):
            if fr.get("split") != "train":
                continue
            idx = int(fr["frame_idx"])
            if idx in protected:
                continue
            if idx in failed:
                out.append(i)
        if len(out) != len(FAILED_TRAIN_FRAME_INDICES):
            raise SystemExit(f"REPAIR_NAV_MISMATCH:{len(out)}")
        return out

    def _audit(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event["ts"] = utc_now()
        event.setdefault("index", self.index)
        event.setdefault("repair_mode", self.repair_mode)
        if self.nav_indices:
            fr = self.current()
            event.setdefault("frame_idx", fr.get("frame_idx"))
            event.setdefault("split", fr.get("split"))
        append_audit_line(self.audit_path, event)

    def _assert_mutable_current(self) -> None:
        fr = self.current()
        if not self.repair_mode:
            return
        if fr.get("split") != "train":
            raise IndependentGTError("REPAIR_MODE_TRAIN_ONLY")
        idx = int(fr["frame_idx"])
        if idx in PROTECTED_TRAIN_FRAME_INDICES:
            raise IndependentGTError(f"PROTECTED_TRAIN_IMMUTABLE:{idx}")
        if idx not in FAILED_TRAIN_FRAME_INDICES:
            raise IndependentGTError(f"NOT_IN_REPAIR_SET:{idx}")

    def save(self) -> None:
        after = collect_fingerprints(self.draft)
        if self.holdout_v2 or self.active_learning:
            # Protect frozen-era fingerprints / source; AL+holdout drafts are isolated runtimes.
            if after.get("source_sha256") != self.immutable_fp.get("source_sha256"):
                raise IndependentGTError("AL_OR_HOLDOUT_SOURCE_SHA_CHANGED")
            if after.get("dev") != self.immutable_fp.get("dev"):
                raise IndependentGTError("AL_OR_HOLDOUT_MUST_NOT_TOUCH_DEV")
            if not self.active_learning and after.get("protected_train") != self.immutable_fp.get(
                "protected_train"
            ):
                raise IndependentGTError("HOLDOUT_V2_MUST_NOT_TOUCH_TRAIN")
        else:
            assert_immutable_fingerprints(self.immutable_fp, after)
        self.draft["updated_at_utc"] = utc_now()
        self.draft["human_approved"] = False
        self.draft["reviewed_gt"] = False
        self.draft["frozen"] = False
        self.draft["acceptance_eligible"] = False
        if self.repair_mode:
            self.draft["repair_mode"] = self.repair_mode
        if self.holdout_v2:
            self.draft["holdout_v2"] = True
            self.draft["blind"] = True
        if self.active_learning:
            self.draft["active_learning"] = True
            self.draft["holdout_v2"] = True
        atomic_write_json(self.draft_path, self.draft)
        by = {
            "train": {"n": 0, "complete": 0},
            "dev": {"n": 0, "complete": 0},
            "holdout": {"n": 0, "complete": 0},
        }
        n_complete = 0
        repair_complete = 0
        for fr in self.draft["frames"]:
            sp = fr["split"]
            by[sp]["n"] += 1
            if fr.get("completed"):
                by[sp]["complete"] += 1
                n_complete += 1
            if (
                fr.get("split") == "train"
                and int(fr["frame_idx"]) in FAILED_TRAIN_FRAME_INDICES
                and fr.get("completed")
            ):
                repair_complete += 1
        al_n = al_c = ho_n = ho_c = 0
        for fr in self.draft["frames"]:
            sec = fr.get("section")
            if sec == "active_learning" or (self.active_learning and fr.get("split") == "train"):
                al_n += 1
                if fr.get("completed"):
                    al_c += 1
            if sec == "holdout_v2" or (self.active_learning and fr.get("split") == "holdout"):
                ho_n += 1
                if fr.get("completed"):
                    ho_c += 1
        prog_payload = {
            "schema": (
                "active_learning_progress_v1"
                if self.active_learning
                else "independent_gt_progress_v1"
            ),
            "n_frames": len(self.draft["frames"]),
            "n_complete": n_complete,
            "by_split": by,
            "repair": {
                "mode": self.repair_mode,
                "target_n": len(FAILED_TRAIN_FRAME_INDICES),
                "complete_n": repair_complete,
                "protected_train": list(PROTECTED_TRAIN_FRAME_INDICES),
            },
            "updated_at_utc": utc_now(),
        }
        if self.active_learning:
            prog_payload["active_learning"] = {"n": al_n, "complete": al_c}
            prog_payload["holdout_v2"] = {"n": ho_n, "complete": ho_c}
        atomic_write_json(self.progress_path, prog_payload)
        atomic_write_json(
            self.session_path,
            {
                "schema": "independent_gt_session_state_v1",
                "index": self.index if not self.repair_mode else self.nav_indices[self.index],
                "repair_nav_pos": self.index if self.repair_mode else None,
                "repair_mode": self.repair_mode,
                "active_learning": self.active_learning,
                "holdout_v2": self.holdout_v2 or self.active_learning,
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
        return self.draft["frames"][self.nav_indices[self.index]]

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
        assert_no_prediction_leakage(fr, split=split)
        section = str(fr.get("section") or "")
        is_holdout_section = section == "holdout_v2" or (
            self.holdout_v2 and not self.active_learning
        )
        is_al_section = section == "active_learning" or (self.active_learning and split == "train")
        proposals = []
        if (is_al_section and not is_holdout_section) or (
            split == "train" and not self.holdout_v2 and not self.active_learning
        ):
            proposals = list(fr.get("proposals") or [])
        prog = json.loads(self.progress_path.read_text(encoding="utf-8"))
        counts = frame_gate_counts(fr)
        gate_errs = (
            validate_train_complete_allowed(fr)
            if (split == "train" or is_al_section or is_holdout_section)
            else []
        )
        blind = (
            is_holdout_section
            or split in {"dev", "holdout"}
            or (self.holdout_v2 and not self.active_learning)
        )
        return {
            "index": self.index,
            "n_frames": len(self.nav_indices),
            "frame_idx": fr["frame_idx"],
            "t_s": fr["t_s"],
            "split": split,
            "section": section or None,
            "categories": fr.get("categories") or [],
            "completed": bool(fr.get("completed")),
            "humans": list(fr.get("humans") or []),
            "proposals": proposals,
            "rejected_proposals": ([] if blind else list(fr.get("rejected_proposals") or [])),
            "no_human_confirmed": bool(fr.get("no_human_confirmed")),
            "repair_required": bool(fr.get("repair_required")),
            "repair_reason": fr.get("repair_reason"),
            "repair_mode": self.repair_mode,
            "holdout_v2": self.holdout_v2 or is_holdout_section,
            "active_learning": self.active_learning,
            "gate_counts": counts,
            "complete_hard_errors": gate_errs,
            "complete_soft_warnings": soft_complete_warnings(fr),
            "progress": prog,
            "source_wh": [SOURCE_WIDTH, SOURCE_HEIGHT],
            "coordinate_space": "source_xyxy_px_v1",
            "blind": blind,
            "policy": {
                "class": "human",
                "train_proposals_are_not_gt": True,
                "dev_holdout_blind": True,
                "holdout_v2_blind": True,
                "active_learning_dual_section": self.active_learning,
                "freeze_requires_explicit_user_approval": True,
                "false_complete_blocked": True,
                "model_predictions_api_forbidden": True,
            },
        }


HTML = r"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>R1 Independent Human GT</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#0e0e10;color:#eee;display:flex;height:100vh;overflow:hidden}
#side{width:400px;padding:12px;overflow:auto;background:#16161a;border-right:1px solid #333}
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
#repairBanner{display:none;background:#3a2a12;border:1px solid #c47a20;padding:8px;margin-bottom:8px;font-size:13px}
.bulk{background:#c47a20;color:#111;font-weight:600}
</style></head><body>
<div id="side">
  <div id="repairBanner"></div>
  <h2 id="title">R1 Bağımsız İnsan GT</h2>
  <div class="warn" id="topNote">Acceptance / freeze / fine-tuning YOK. Yalnız etiketleme.</div>
  <div class="row"><span id="splitBadge" class="badge"></span>
    <span id="meta" class="badge"></span></div>
  <div class="row">İlerleme: <b id="prog"></b></div>
  <div class="row" id="repairProg" class="ok"></div>
  <div class="row" id="blindNote" class="ok"></div>
  <div class="row" id="gateCounts" class="warn"></div>
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
  <div class="row" id="repairActions" style="display:none">
    <button class="bulk" onclick="bulkAccept()">Karedeki tüm önerileri insan olarak kabul et</button>
    <button onclick="rejectAllProps()">Tüm önerileri reddet</button>
  </div>
  <div class="row">
    <button onclick="acceptProposal()">Accept proposal → human</button>
    <button onclick="completeNext()" style="background:#2d5a3d;color:#fff">Complete + Next (C)</button>
  </div>
  <div class="row">
    <label><input type="checkbox" id="noHuman" onchange="togNoHuman()"/>
      Bu karede görünür insan olmadığını elle doğruladım</label>
  </div>
  <div class="row">
    <button onclick="markComplete(false)">Mark complete</button>
    <button onclick="markComplete(true)">Uncomplete</button>
    <button onclick="save()">Save</button>
  </div>
  <div class="row" id="status"></div>
  <div class="row" style="font-size:12px">
    Kısayol: <kbd>Y</kbd><kbd>W</kbd><kbd>R</kbd><kbd>G</kbd><kbd>U</kbd><kbd>O</kbd><kbd>C</kbd>
    · Shift+click çoklu seçim · wheel zoom · orta tuş pan
  </div>
  <div class="row warn" id="warnBox"></div>
  <ol id="list"></ol>
</div>
<div id="main"><canvas id="c"></canvas></div>
<script>
const SOURCE_W=1336, SOURCE_H=744;
let state=null, img=new Image(), sel=-1, selProp=-1;
let selected=new Set();
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
  const repair=!!state.repair_mode;
  const al=!!state.active_learning;
  const banner=document.getElementById('repairBanner');
  const actions=document.getElementById('repairActions');
  if(repair){
    banner.style.display='block';
    actions.style.display='block';
    banner.innerHTML='<b>TRAIN REPAIR — 37 KARE</b><br/>Turuncu = model önerisi, henüz GT değil<br/>Camgöbeği = sizin kabul ettiğiniz insan kutusu';
    document.getElementById('title').textContent='TRAIN REPAIR';
    document.getElementById('topNote').textContent='Dev/holdout korunur. Model önerisi otomatik GT olmaz.';
  } else if(al){
    banner.style.display='block';
    actions.style.display=(state.section==='active_learning')?'block':'none';
    const sec=state.section||'';
    banner.innerHTML=sec==='holdout_v2'
      ? '<b>BÖLÜM B — BLIND HOLDOUT V2</b><br/>Proposal/prediction YOK — sıfırdan kör annotation'
      : '<b>BÖLÜM A — ACTIVE LEARNING</b><br/>Turuncu = proposal (GT değil). Accept/düzelt/sil/çiz.';
    document.getElementById('title').textContent='ACTIVE LEARNING + HOLDOUT V2';
    document.getElementById('topNote').textContent='Freeze/eğitim/R2 YOK. Frozen GT v1 değişmez.';
  } else {
    banner.style.display='none';
    actions.style.display='none';
  }
  document.getElementById('splitBadge').textContent=(state.section||state.split).toUpperCase();
  document.getElementById('splitBadge').className='badge split-'+state.split;
  document.getElementById('meta').textContent=
    `i=${state.index+1}/${state.n_frames} f=${state.frame_idx} t=${state.t_s.toFixed(2)}s`+
    (state.completed?' ✓':'');
  const p=state.progress;
  if(al && p.active_learning && p.holdout_v2){
    document.getElementById('prog').textContent=
      `active learning ${p.active_learning.complete}/${p.active_learning.n}`+
      ` · blind holdout ${p.holdout_v2.complete}/${p.holdout_v2.n}`+
      ` · total ${p.n_complete}/${p.n_frames}`;
  } else {
    document.getElementById('prog').textContent=
      `${p.n_complete}/${p.n_frames} · train ${p.by_split.train.complete}/${p.by_split.train.n}`+
      ` · dev ${p.by_split.dev.complete}/${p.by_split.dev.n}`+
      ` · holdout ${p.by_split.holdout.complete}/${p.by_split.holdout.n}`;
  }
  if(repair && p.repair){
    document.getElementById('repairProg').textContent=
      `Repair: ${p.repair.complete_n||0}/${p.repair.target_n||37} (ana 80-kare sayacından ayrı)`;
  } else if(al){
    document.getElementById('repairProg').textContent=
      (state.section==='holdout_v2')
        ? 'Blind holdout: model API kapalı'
        : 'Pending proposal varken Complete çalışmaz';
  } else {
    document.getElementById('repairProg').textContent='';
  }
  document.getElementById('blindNote').textContent=state.blind
    ? 'DEV/HOLDOUT KÖR: proposal/prediction gizli — sıfırdan çizin'
    : (repair || (al && state.section==='active_learning')
      ? 'Turuncu kesikli = pending proposal (GT değil). Camgöbeği = accepted human.'
      : 'TRAIN: mavi kesikli = proposal (GT değil). Kabul/düzelt/sil.');
  const gc=state.gate_counts||{};
  document.getElementById('gateCounts').textContent=
    `Accepted humans: ${gc.accepted_humans||0} · Pending proposals: ${gc.pending_proposals||0}`+
    ` · Rejected proposals: ${gc.rejected_proposals||0}`;
  document.getElementById('noHuman').checked=!!state.no_human_confirmed;
  const soft=state.complete_soft_warnings||[];
  const hard=state.complete_hard_errors||[];
  document.getElementById('warnBox').textContent=
    (hard.length?('HARD: '+hard.join('; ')+' | '):'')+(soft.length?('UYARI: '+soft.join('; ')):'' );
  sel=-1; selProp=-1; selected=new Set();
  img.onload=()=>{fit(); draw();};
  img.src='/api/frame.jpg?frame_idx='+state.frame_idx+'&_='+Date.now();
  renderList();
}

function fit(){
  const maxW=window.innerWidth-420, maxH=window.innerHeight-40;
  const s=Math.min(maxW/SOURCE_W, maxH/SOURCE_H);
  canvas.width=Math.floor(SOURCE_W*s); canvas.height=Math.floor(SOURCE_H*s);
  view={scale:1,x:0,y:0};
}

function transform(){
  return letterbox(canvas.width, canvas.height);
}

function draw(){
  const t=transform();
  const repair=!!state.repair_mode;
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.translate(view.x, view.y);
  ctx.scale(view.scale, view.scale);
  ctx.drawImage(img, t.pad_x, t.pad_y, t.content_w, t.content_h);
  (state.proposals||[]).forEach((p,i)=>{
    const b=srcToCanvas(p.bbox_xyxy,t);
    ctx.strokeStyle=i===selProp?'#ffb74d':(repair?'#ff9800':'#0288d1');
    ctx.setLineDash([6,4]); ctx.lineWidth=2;
    ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
    ctx.setLineDash([]);
  });
  (state.humans||[]).forEach((h,i)=>{
    const b=srcToCanvas(h.bbox_xyxy,t);
    let col=repair?'#00bcd4':'#00e676';
    if(h.eligibility==='off_pitch') col='#9e9e9e';
    if(h.eligibility==='uncertain' && !repair) col='#ffd54f';
    if(i===sel || selected.has(i)) col='#ff5252';
    ctx.strokeStyle=col; ctx.lineWidth=(i===sel||selected.has(i))?3:2;
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
    li.textContent=`#${i} ${h.role}/${h.team_appearance}/${h.eligibility} [${h.origin||'?'}]`;
    li.style.cursor='pointer';
    if(i===sel||selected.has(i)) li.style.color='#ff8a80';
    li.onclick=(ev)=>{
      if(ev.shiftKey){
        if(selected.has(i)) selected.delete(i); else selected.add(i);
        sel=i; selProp=-1;
      } else {
        selected=new Set([i]); sel=i; selProp=-1; syncMeta();
      }
      draw(); renderList();
    };
    ol.appendChild(li);
  });
  if(!state.blind){
    (state.proposals||[]).forEach((p,i)=>{
      const li=document.createElement('li');
      li.textContent=`proposal#${i} score=${(p.score||0).toFixed(2)} NOT GT`;
      li.style.color=state.repair_mode?'#ff9800':'#4fc3f7'; li.style.cursor='pointer';
      li.onclick=()=>{selProp=i; sel=-1; selected=new Set(); draw();};
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
  const t=transform();
  for(let i=state.humans.length-1;i>=0;i--){
    const b=srcToCanvas(state.humans[i].bbox_xyxy,t);
    if(p.cx>=b[0]&&p.cx<=b[2]&&p.cy>=b[1]&&p.cy<=b[3]){
      if(ev.shiftKey){
        if(selected.has(i)) selected.delete(i); else selected.add(i);
        sel=i; selProp=-1;
      } else {
        selected=new Set([i]); sel=i; selProp=-1; syncMeta();
        drag={kind:'move', i, ox:p.sx, oy:p.sy, box:state.humans[i].bbox_xyxy.slice()};
      }
      draw(); renderList(); return;
    }
  }
  if(!ev.shiftKey) selected=new Set();
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
      state=j.state; sel=state.humans.length-1; selected=new Set([sel]); renderList(); syncMeta();
      if(j.warnings&&j.warnings.length) document.getElementById('warnBox').textContent=j.warnings.join('; ');
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
  const idxs=selected.size?Array.from(selected):(sel>=0?[sel]:[]);
  if(!idxs.length) return;
  try{
    for(const i of idxs){
      const j=await api('/api/update_box', Object.assign({index:i}, metaPayload()));
      state=j.state;
    }
    renderList(); draw(); setStatus('meta güncellendi','ok');
  }catch(e){ setStatus(String(e),'err'); }
}
async function delSel(){
  if(sel<0) return;
  const j=await api('/api/delete_box',{index:sel});
  state=j.state; sel=-1; selected=new Set(); renderList(); draw(); setStatus('silindi','ok');
}
async function acceptProposal(){
  if(state.blind){ setStatus('dev/holdout kör — proposal yok','err'); return; }
  if(selProp<0){ setStatus('proposal seçin','err'); return; }
  const j=await api('/api/accept_proposal', Object.assign({proposal_index:selProp}, metaPayload()));
  state=j.state; sel=state.humans.length-1; selProp=-1; selected=new Set([sel]); renderList(); draw();
  setStatus('proposal human-reviewed olarak alındı','ok');
}
async function bulkAccept(){
  try{
    const j=await api('/api/bulk_accept_proposals',{});
    state=j.state; sel=-1; selProp=-1; selected=new Set(); renderList(); draw();
    setStatus('Toplu kabul: '+((j.added_n)||0)+' — Complete otomatik değil','ok');
  }catch(e){ setStatus(String(e),'err'); }
}
async function rejectAllProps(){
  try{
    const j=await api('/api/reject_proposals',{});
    state=j.state; selProp=-1; renderList(); draw();
    setStatus('Öneriler reddedildi','ok');
  }catch(e){ setStatus(String(e),'err'); }
}
async function togNoHuman(){
  const v=document.getElementById('noHuman').checked;
  try{
    const j=await api('/api/no_human_confirm',{confirmed:v});
    state=j.state; await load();
  }catch(e){
    document.getElementById('noHuman').checked=!!state.no_human_confirmed;
    setStatus(String(e),'err');
  }
}
async function nav(d){
  const j=await api('/api/nav',{delta:d}); state=j.state; await load();
}
async function markComplete(un){
  try{
    const j=await api('/api/complete',{completed: !un});
    state=j.state; await load();
  }catch(e){ setStatus(String(e),'err'); }
}
async function completeNext(){
  try{
    await api('/api/complete',{completed:true});
    await nav(1);
  }catch(e){ setStatus(String(e),'err'); }
}
async function save(){ await api('/api/save',{}); setStatus('kaydedildi','ok'); }
async function undo(){ const j=await api('/api/undo',{}); state=j.state; await load(); }
async function redo(){ const j=await api('/api/redo',{}); state=j.state; await load(); }

document.addEventListener('keydown', ev=>{
  if(ev.target.tagName==='INPUT'||ev.target.tagName==='SELECT') return;
  const k=ev.key.toLowerCase();
  if(k==='y'){ document.getElementById('role').value='player'; document.getElementById('team').value='yellow'; document.getElementById('elig').value='on_pitch'; applyMeta(); }
  if(k==='w'){ document.getElementById('role').value='player'; document.getElementById('team').value='white'; document.getElementById('elig').value='on_pitch'; applyMeta(); }
  if(k==='r'){ document.getElementById('role').value='referee'; document.getElementById('team').value='official'; document.getElementById('elig').value='on_pitch'; applyMeta(); }
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
                payload = {
                    "status": "ok",
                    "service": "r1_independent_gt_review",
                    "source_id": "own_video_97b298e4",
                    "source_sha256_ok": app.source_sha == EXPECTED_SOURCE_SHA256,
                    "host": "127.0.0.1",
                    "repair_mode": app.repair_mode,
                    "holdout_v2": bool(app.holdout_v2),
                    "active_learning": bool(app.active_learning),
                }
                self._json(200, payload)
                return
            if path == "/":
                data = HTML.encode("utf-8")
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
                    if app.repair_mode:
                        allowed = {
                            int(app.draft["frames"][i]["frame_idx"]) for i in app.nav_indices
                        }
                    else:
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
            if path == "/api/repair_validate":
                with app.lock:
                    self._json(200, validate_repair_complete(app.draft))
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            body = self._read_json()
            try:
                with app.lock:
                    if path == "/api/nav":
                        app.index = (app.index + int(body.get("delta", 0))) % len(app.nav_indices)
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
                        app._assert_mutable_current()
                        fr = app.current()
                        want = bool(body.get("completed", True))
                        section = str(fr.get("section") or "")
                        needs_gate = fr.get("split") == "train" or section in {
                            "active_learning",
                            "holdout_v2",
                        }
                        if want and needs_gate:
                            errs = validate_train_complete_allowed(fr)
                            if errs:
                                raise IndependentGTError(";".join(errs))
                        fr["completed"] = want
                        fr["review_status"] = (
                            "complete"
                            if want
                            else ("repair_required" if fr.get("repair_required") else "incomplete")
                        )
                        app.save()
                        app._audit(
                            {
                                "event": "complete",
                                "completed": fr["completed"],
                                "gate_counts": frame_gate_counts(fr),
                            }
                        )
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/add_box":
                        app._assert_mutable_current()
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
                        app._assert_mutable_current()
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
                        app._assert_mutable_current()
                        app.push_undo()
                        fr = app.current()
                        i = int(body["index"])
                        removed = fr["humans"].pop(i)
                        app.save()
                        app._audit({"event": "delete_box", "box": removed})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/accept_proposal":
                        if app.holdout_v2 or (
                            app.active_learning and app.current().get("section") == "holdout_v2"
                        ):
                            raise IndependentGTError("HOLDOUT_V2_PROPOSALS_FORBIDDEN")
                        app._assert_mutable_current()
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
                        fr["proposals"].pop(pi)
                        app.save()
                        app._audit({"event": "accept_proposal", "box": hum})
                        self._json(200, {"state": app.public_state()})
                        return
                    if path == "/api/bulk_accept_proposals":
                        if app.holdout_v2 or (
                            app.active_learning and app.current().get("section") == "holdout_v2"
                        ):
                            raise IndependentGTError("HOLDOUT_V2_PROPOSALS_FORBIDDEN")
                        app._assert_mutable_current()
                        fr = app.current()
                        app.push_undo()
                        added = bulk_accept_proposals(fr, box_id_factory=uuid.uuid4)
                        app.save()
                        app._audit(
                            {
                                "event": "bulk_accept_proposals",
                                "added_n": len(added),
                                "provenance": "proposal_reviewed_bulk",
                                "reviewer_action": "bulk_accept_visible_proposals",
                            }
                        )
                        self._json(
                            200,
                            {"state": app.public_state(), "added_n": len(added)},
                        )
                        return
                    if path == "/api/reject_proposals":
                        if app.holdout_v2 or (
                            app.active_learning and app.current().get("section") == "holdout_v2"
                        ):
                            raise IndependentGTError("HOLDOUT_V2_PROPOSALS_FORBIDDEN")
                        app._assert_mutable_current()
                        fr = app.current()
                        app.push_undo()
                        n = reject_pending_proposals(fr)
                        app.save()
                        app._audit({"event": "reject_proposals", "rejected_n": n})
                        self._json(200, {"state": app.public_state(), "rejected_n": n})
                        return
                    if path == "/api/no_human_confirm":
                        app._assert_mutable_current()
                        fr = app.current()
                        app.push_undo()
                        confirmed = bool(body.get("confirmed", False))
                        set_no_human_confirmed(fr, confirmed)
                        app.save()
                        app._audit(
                            {
                                "event": "no_human_confirm",
                                "confirmed": confirmed,
                            }
                        )
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
                        raise IndependentGTError(
                            "FREEZE_FORBIDDEN_IN_R1_F2_A_REQUIRES_EXPLICIT_LATER_APPROVAL"
                        )
                    if path in {"/api/predict", "/api/infer", "/api/model"}:
                        raise IndependentGTError("MODEL_INFERENCE_API_FORBIDDEN")
                    if (
                        app.active_learning
                        and path.startswith("/api/")
                        and (
                            app.current().get("section") == "holdout_v2"
                            and path
                            in {
                                "/api/accept_proposal",
                                "/api/bulk_accept_proposals",
                                "/api/reject_proposals",
                            }
                        )
                    ):
                        raise IndependentGTError("HOLDOUT_V2_PROPOSALS_FORBIDDEN")
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
    ap.add_argument(
        "--repair",
        default=None,
        help=f"Repair mode, e.g. {REPAIR_MODE}",
    )
    ap.add_argument(
        "--holdout-v2",
        action="store_true",
        help="Blind holdout_v2 review (no proposals / no model leakage)",
    )
    ap.add_argument(
        "--active-learning",
        action="store_true",
        help="Dual AL + blind holdout_v2 review",
    )
    args = ap.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("HOST_MUST_BE_LOCALHOST")
    app = ReviewApp(
        args.runtime,
        args.video,
        repair_mode=args.repair,
        holdout_v2=bool(args.holdout_v2),
        active_learning=bool(args.active_learning),
    )
    handler = build_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    mode = ""
    if args.repair:
        mode += f" repair={args.repair}"
    if args.holdout_v2:
        mode += " holdout_v2"
    if args.active_learning:
        mode += " active_learning"
    print(f"R1 Independent GT review on http://{args.host}:{args.port}/{mode}", flush=True)
    print(f"runtime={args.runtime}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutdown", flush=True)
    finally:
        app.cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
