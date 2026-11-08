"""
Subtitle generator v2 (Upload & Path) - video/audio -> speech-to-text (Groq Whisper) -> .vtt, and optionally Bangla .bn.vtt
"""
import bisect
import glob
import json
import math
import os
import queue
import re
import shutil
import struct
import subprocess
import tempfile
import textwrap
import threading
import time
import uuid

import requests
from flask import Flask, Response, abort, jsonify, request, send_file

app = Flask(__name__)

# ---------------- config ----------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
IS_ANDROID = "ANDROID_ROOT" in os.environ or os.path.isdir("/storage/emulated/0")


def default_output_dir():
    if os.environ.get("OUTPUT_DIR"):
        return os.environ["OUTPUT_DIR"]
    if IS_ANDROID:
        for cand in ("/storage/emulated/0/Download", "/sdcard/Download"):
            if os.path.isdir(cand) and os.access(cand, os.W_OK):
                return os.path.join(cand, "subtitles_output")
    return os.path.join(BASE_DIR, "subtitles_output")


OUTPUT_DIR = default_output_dir()
WORK_DIR = os.path.join(tempfile.gettempdir(), "subgen_work")
API_BASE = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
STT_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]
CHAT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
CHUNK_SECS = int(os.environ.get("CHUNK_SECS", "480"))
OVERLAP_SECS = float(os.environ.get("OVERLAP_SECS", "4"))
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
TRANSLATE_CHUNK_CHARS = int(os.environ.get("TRANSLATE_CHUNK_CHARS", "9000"))
TPM_LIMIT = int(os.environ.get("GROQ_TPM", "10000"))
MAX_WAIT = 900
MAX_CHARS = 84
NL = "<br>"
LINE_RE = re.compile(r"^\s*(\d+)\s*\|(.*)$")
MEDIA_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".ts", ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".3gp"}
ISO_EXTS = {".mp4", ".m4v", ".mov", ".m4a", ".3gp"}

os.makedirs(OUTPUT_DIR, exist_ok=True) if os.access(os.path.dirname(OUTPUT_DIR) or ".", os.W_OK) else None
shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(WORK_DIR, exist_ok=True)

jobs, order = {}, []
job_q = queue.Queue()

TRANSLATE_PROMPT = (
    "Translate numbered subtitle lines into natural, spoken Bangla (Bengali script). "
    "Input: one subtitle per line as `id|text`. Output: the SAME ids, in the SAME order, one per "
    "line, as `id|bangla text`. Keep any <br> markers as they are. Keep people's names and brand names consistent "
    "across all lines. Stay faithful to meaning and tone; do not censor, summarize or add notes or extra lines."
)

LANG_CODES = {"english": "en", "bengali": "bn", "bangla": "bn", "hindi": "hi", "urdu": "ur", "spanish": "es", "french": "fr",
              "german": "de", "portuguese": "pt", "russian": "ru", "arabic": "ar", "turkish": "tr", "italian": "it",
              "japanese": "ja", "korean": "ko", "chinese": "zh", "tamil": "ta", "telugu": "te", "marathi": "mr",
              "punjabi": "pa", "gujarati": "gu", "nepali": "ne", "indonesian": "id", "thai": "th", "vietnamese": "vi",
              "dutch": "nl", "polish": "pl", "ukrainian": "uk", "persian": "fa"}


class AuthError(Exception): pass
class LimitError(Exception): pass
class TooLarge(Exception): pass
class NeedFfmpeg(RuntimeError): pass
class RateLimited(Exception):
    def __init__(self, wait):
        super().__init__(f"rate limited {wait}s")
        self.wait = wait
class ModelMissing(Exception): pass
class ApiError(Exception): pass
class ServerError(ApiError): pass


def setp(job, pct=None, msg=None, status=None):
    if pct is not None: job["pct"] = max(0, min(100, int(pct)))
    if msg is not None: job["msg"] = msg
    if status is not None: job["status"] = status


def wait_msg(job, seconds, label):
    keep = job["msg"]
    end = time.time() + seconds
    while time.time() < end:
        job["msg"] = f"{label} — {int(end - time.time())} সেকেন্ড অপেক্ষা..."
        time.sleep(min(1.0, max(0.0, end - time.time())))
    job["msg"] = keep


def safe_name(name):
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", name or "").strip(" .")
    return re.sub(r"\s+", " ", name)[:100] or "video"


def clean_key(key):
    return (key or "").strip().strip('"').strip("'").strip()


def fmt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def has_cjk(text):
    return len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0e00-\u0e7f]", text)) > len(text) * 0.3


def wrap_text(t):
    if len(t) <= 42: return t
    lines = textwrap.wrap(t, width=math.ceil(len(t) / 2) + 6, break_long_words=False)
    if len(lines) > 2: lines = [lines[0], " ".join(lines[1:])]
    return "\n".join(lines)


def write_vtt(cues, path, field="text"):
    out = ["WEBVTT", ""]
    for i, c in enumerate(cues, 1):
        out += [str(i), f"{fmt_time(c['start'])} --> {fmt_time(c['end'])}", wrap_text(c[field]), ""]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f: f.write("\n".join(out))
    os.replace(tmp, path)


def retry_after(resp):
    h = resp.headers.get("retry-after")
    if h:
        try: return float(h)
        except ValueError: pass
    m = re.search(r"try again in\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?", resp.text or "")
    if m and (m.group(1) or m.group(2)):
        return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return 20.0


def check_response(resp):
    c = resp.status_code
    if c < 300: return
    if c == 429: raise RateLimited(retry_after(resp))
    if c == 401: raise AuthError("API key ভুল বা মেয়াদ শেষ")
    if c == 404: raise ModelMissing()
    if c == 413: raise TooLarge()
    msg = (resp.text or "")[:200]
    try: msg = resp.json()["error"]["message"][:200]
    except Exception: pass
    raise (ServerError if c >= 500 else ApiError)(f"HTTP {c}: {msg}")


def groq_request(job, call, models):
    rate_waits = errors = 0
    while True:
        try: return call(models[0])
        except RateLimited as e:
            if e.wait > MAX_WAIT: raise LimitError(f"Groq-এর লিমিট শেষ — প্রায় {max(1, int(e.wait // 60))} মিনিট পরে আবার চেষ্টা করুন")
            rate_waits += 1
            if rate_waits > 12: raise LimitError("বারবার রেট লিমিট — কিছুক্ষণ পরে আবার চেষ্টা করুন")
            wait_msg(job, e.wait + 1, "রেট লিমিট")
        except ModelMissing:
            if len(models) > 1:
                models.pop(0)
                continue
            raise RuntimeError("Groq মডেল পাওয়া যায়নি")
        except (requests.ConnectionError, requests.Timeout, ServerError) as e:
            errors += 1
            if errors >= 3: raise RuntimeError(f"নেটওয়ার্ক/API সমস্যা: {str(e)[:100]}")
            time.sleep(3 * errors)


MIME = {".flac": "audio/flac", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".ogg": "audio/ogg", ".webm": "audio/webm"}


def transcribe_file(job, key, path, models, language, prompt, want_words=True):
    ctype = MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
    def call(model):
        data = [("model", model), ("response_format", "verbose_json"), ("temperature", "0"),
                ("timestamp_granularities[]", "segment")]
        if want_words: data.append(("timestamp_granularities[]", "word"))
        if language: data.append(("language", language))
        if prompt: data.append(("prompt", prompt))
        with open(path, "rb") as fh:
            r = requests.post(f"{API_BASE}/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
                              data=data, files={"file": (os.path.basename(path), fh, ctype)}, timeout=(15, 300))
        check_response(r)
        return r.json()
    try: return groq_request(job, call, models)
    except ApiError as e:
        if want_words and re.search(r"timestamp|granular", str(e), re.I):
            return transcribe_file(job, key, path, models, language, prompt, want_words=False)
        raise


_ffmpeg_cache = {}
def find_ffmpeg():
    if "exe" in _ffmpeg_cache: return _ffmpeg_cache["exe"]
    exe = None
    if not os.environ.get("FORCE_NO_FFMPEG"):
        cands = [shutil.which("ffmpeg")]
        try:
            import imageio_ffmpeg
            cands.append(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception: pass
        for c in cands:
            if c and os.path.isfile(c):
                try:
                    if subprocess.run([c, "-version"], capture_output=True, timeout=15).returncode == 0:
                        exe = c
                        break
                except Exception: pass
    _ffmpeg_cache["exe"] = exe
    return exe


def media_duration_ffmpeg(ffmpeg, src):
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", src], capture_output=True, text=True, errors="replace")
    if "Audio:" not in r.stderr: raise RuntimeError("এই ফাইলে কোনো অডিও ট্র্যাক নেই")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if m: return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", src, "-vn", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    t = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not t: raise RuntimeError("ফাইলের দৈর্ঘ্য বোঝা যায়নি")
    h, mi, s = t[-1]
    return int(h) * 3600 + int(mi) * 60 + float(s)


def plan_chunks(total, chunk=None, overlap=None):
    chunk = chunk or CHUNK_SECS
    overlap = OVERLAP_SECS if overlap is None else overlap
    starts = [i * chunk for i in range(max(1, math.ceil(total / chunk)))]
    if len(starts) > 1 and total - starts[-1] < 5: starts.pop()
    plan = []
    for i, s in enumerate(starts):
        last = i == len(starts) - 1
        d = (total - s) if last else min(chunk + overlap, total - s)
        plan.append((float(s), float(d)))
    return plan


def extract_chunks_ffmpeg(ffmpeg, src, workdir, clean, job):
    total = media_duration_ffmpeg(ffmpeg, src)
    plan = plan_chunks(total)
    af = "highpass=f=80,lowpass=f=7600,loudnorm=I=-18:TP=-2:LRA=11" if clean else None
    out = []
    for i, (s, d) in enumerate(plan):
        setp(job, pct=1 + 7 * i / len(plan), msg=f"অডিও আলাদা করা হচ্ছে: অংশ {i + 1}/{len(plan)}")
        path = os.path.join(workdir, f"chunk_{i:03d}.flac")
        cmd = [ffmpeg, "-y", "-v", "error", "-ss", f"{s:.3f}", "-t", f"{d:.3f}", "-i", src, "-map", "0:a:0", "-vn",
               "-ac", "1", "-ar", "16000"] + (["-af", af] if af else []) + ["-c:a", "flac", path]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0 or not os.path.exists(path):
            raise RuntimeError("ffmpeg অডিও আলাদা করতে পারেনি: " + (r.stderr or "")[-200:])
        if os.path.getsize(path) > MAX_UPLOAD_BYTES:
            path2 = os.path.join(workdir, f"chunk_{i:03d}.mp3")
            subprocess.run([ffmpeg, "-y", "-v", "error", "-i", path, "-c:a", "libmp3lame", "-b:a", "96k", path2],
                           capture_output=True)
            if os.path.exists(path2): path = path2
        out.append({"path": path, "start": s, "dur": d})
    return out


def box(typ, payload=b""):
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def full_box(typ, version, flags, payload):
    return box(typ, struct.pack(">I", (version << 24) | flags) + payload)


def children(buf, start, end):
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(buf[pos:pos + 4], "big")
        typ = bytes(buf[pos + 4:pos + 8])
        hs = 8
        if size == 1:
            size, hs = int.from_bytes(buf[pos + 8:pos + 16], "big"), 16
        elif size == 0:
            size = end - pos
        if size < hs or pos + size > end: break
        yield typ, pos + hs, pos + size, pos
        pos += size


class Mp4Audio:
    def __init__(self, path):
        self.path = path
        size = os.path.getsize(path)
        moov = None
        with open(path, "rb") as f:
            pos = 0
            while pos + 8 <= size:
                f.seek(pos)
                hdr = f.read(8)
                if len(hdr) < 8: break
                bsize, typ, hs = int.from_bytes(hdr[:4], "big"), hdr[4:8], 8
                if bsize == 1: bsize, hs = int.from_bytes(f.read(8), "big"), 16
                elif bsize == 0: bsize = size - pos
                if bsize < hs: break
                if typ == b"moof": raise NeedFfmpeg("fragmented MP4")
                if typ == b"moov":
                    if bsize > 400 * 1024 * 1024: raise NeedFfmpeg("moov too large")
                    f.seek(pos + hs)
                    moov = f.read(bsize - hs)
                pos += bsize
        if moov is None: raise NeedFfmpeg("no moov box (not an MP4)")
        self._parse(moov)

    def _parse(self, buf):
        for typ, b0, b1, _ in children(buf, 0, len(buf)):
            if typ == b"mvex": raise NeedFfmpeg("fragmented MP4")
        for typ, b0, b1, _ in children(buf, 0, len(buf)):
            if typ != b"trak": continue
            kind, tables, timescale = None, {}, None
            for t2, c0, c1, _ in children(buf, b0, b1):
                if t2 != b"mdia": continue
                for t3, d0, d1, _ in children(buf, c0, c1):
                    if t3 == b"hdlr": kind = bytes(buf[d0 + 8:d0 + 12])
                    elif t3 == b"mdhd":
                        ver = buf[d0]
                        timescale = int.from_bytes(buf[d0 + (20 if ver == 1 else 12):d0 + (24 if ver == 1 else 16)], "big")
                    elif t3 == b"minf":
                        for t4, e0, e1, _ in children(buf, d0, d1):
                            if t4 == b"stbl":
                                for t5, g0, g1, gs in children(buf, e0, e1): tables[t5] = (g0, g1, gs)
            if kind != b"soun": continue
            self.timescale = timescale or 44100
            for need in (b"stsd", b"stts", b"stsc", b"stsz"):
                if need not in tables: raise NeedFfmpeg("incomplete audio track")
            if b"stco" not in tables and b"co64" not in tables: raise NeedFfmpeg("incomplete audio track")
            g0, g1, gs = tables[b"stsd"]
            self.stsd_raw = bytes(buf[gs:g1])
            codec = bytes(buf[g0 + 12:g0 + 16])
            if codec != b"mp4a": raise NeedFfmpeg(f"audio codec {codec.decode('latin1')}")
            self._tables(buf, tables)
            return
        raise RuntimeError("এই ফাইলে কোনো অডিও ট্র্যাক নেই")

    def _tables(self, buf, tb):
        g0, g1, _ = tb[b"stsz"]
        fixed, count = struct.unpack(">II", buf[g0 + 4:g0 + 12])
        sizes = [fixed] * count if fixed else list(struct.unpack(f">{count}I", buf[g0 + 12:g0 + 12 + 4 * count]))
        g0, g1, _ = tb[b"stts"]
        n = struct.unpack(">I", buf[g0 + 4:g0 + 8])[0]
        durs = []
        for k in range(n):
            c, d = struct.unpack(">II", buf[g0 + 8 + 8 * k:g0 + 16 + 8 * k])
            durs.extend([d] * c)
        if b"stco" in tb:
            g0, g1, _ = tb[b"stco"]
            n = struct.unpack(">I", buf[g0 + 4:g0 + 8])[0]
            chunk_off = list(struct.unpack(f">{n}I", buf[g0 + 8:g0 + 8 + 4 * n]))
        else:
            g0, g1, _ = tb[b"co64"]
            n = struct.unpack(">I", buf[g0 + 4:g0 + 8])[0]
            chunk_off = list(struct.unpack(f">{n}Q", buf[g0 + 8:g0 + 8 + 8 * n]))
        g0, g1, _ = tb[b"stsc"]
        n = struct.unpack(">I", buf[g0 + 4:g0 + 8])[0]
        ents = [struct.unpack(">III", buf[g0 + 8 + 12 * k:g0 + 20 + 12 * k]) for k in range(n)]
        offsets, si = [], 0
        for ci, off in enumerate(chunk_off, 1):
            spc = 0
            for first, per, _ in ents:
                if first <= ci: spc = per
                else: break
            pos = off
            for _ in range(spc):
                if si >= len(sizes): break
                offsets.append(pos)
                pos += sizes[si]
                si += 1
        n_ok = min(len(offsets), len(sizes), len(durs))
        self.offsets, self.sizes, self.durs = offsets[:n_ok], sizes[:n_ok], durs[:n_ok]
        self.starts, acc = [], 0
        for d in self.durs:
            self.starts.append(acc)
            acc += d
        self.total_ticks = acc
        self.total = acc / self.timescale
        if not n_ok: raise RuntimeError("অডিও ট্র্যাক খালি")

    def write_m4a(self, t0, t1, out_path):
        ts = self.timescale
        i0 = bisect.bisect_left(self.starts, int(t0 * ts))
        i1 = bisect.bisect_left(self.starts, int(t1 * ts))
        i1 = max(i1, i0 + 1)
        i1 = min(i1, len(self.sizes))
        sizes, durs = self.sizes[i0:i1], self.durs[i0:i1]
        n = len(sizes)
        data = bytearray()
        with open(self.path, "rb") as f:
            k = i0
            while k < i1:
                run_end = k
                while run_end + 1 < i1 and self.offsets[run_end] + self.sizes[run_end] == self.offsets[run_end + 1]:
                    run_end += 1
                f.seek(self.offsets[k])
                data += f.read(self.offsets[run_end] + self.sizes[run_end] - self.offsets[k])
                k = run_end + 1
        dur_total = sum(durs)
        runs = []
        for d in durs:
            if runs and runs[-1][1] == d: runs[-1][0] += 1
            else: runs.append([1, d])
        matrix = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)

        def moov_for(chunk_offset):
            mvhd = full_box(b"mvhd", 0, 0, struct.pack(">IIII", 0, 0, ts, dur_total) + struct.pack(">IHH", 0x00010000, 0x0100, 0)
                            + b"\0" * 8 + matrix + b"\0" * 24 + struct.pack(">I", 2))
            tkhd = full_box(b"tkhd", 0, 7, struct.pack(">IIIII", 0, 0, 1, 0, dur_total) + b"\0" * 8 + struct.pack(">HHHH", 0, 0, 0x0100, 0)
                            + matrix + struct.pack(">II", 0, 0))
            mdhd = full_box(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, ts, dur_total, 0x55C4, 0))
            hdlr = full_box(b"hdlr", 0, 0, struct.pack(">I", 0) + b"soun" + b"\0" * 12 + b"SoundHandler\0")
            smhd = full_box(b"smhd", 0, 0, struct.pack(">HH", 0, 0))
            dinf = box(b"dinf", full_box(b"dref", 0, 0, struct.pack(">I", 1) + full_box(b"url ", 0, 1, b"")))
            stts = full_box(b"stts", 0, 0, struct.pack(">I", len(runs)) + b"".join(struct.pack(">II", c, d) for c, d in runs))
            stsc = full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, n, 1))
            stsz = full_box(b"stsz", 0, 0, struct.pack(">II", 0, n) + struct.pack(f">{n}I", *sizes))
            stco = full_box(b"stco", 0, 0, struct.pack(">II", 1, chunk_offset))
            stbl = box(b"stbl", self.stsd_raw + stts + stsc + stsz + stco)
            minf = box(b"minf", smhd + dinf + stbl)
            mdia = box(b"mdia", mdhd + hdlr + minf)
            return box(b"moov", mvhd + box(b"trak", tkhd + mdia))

        ftyp = box(b"ftyp", b"M4A " + struct.pack(">I", 0) + b"M4A mp42isom")
        moov = moov_for(0)
        moov = moov_for(len(ftyp) + len(moov) + 8)
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(ftyp); f.write(moov); f.write(struct.pack(">I", 8 + len(data)) + b"mdat"); f.write(data)
        os.replace(tmp, out_path)
        return self.starts[i0] / ts


def extract_chunks_builtin(src, workdir, job):
    a = Mp4Audio(src)
    bitrate = max(1.0, sum(a.sizes) / max(a.total, 1.0))
    chunk = int(min(CHUNK_SECS, max(60, MAX_UPLOAD_BYTES * 0.75 / bitrate - OVERLAP_SECS)))
    plan = plan_chunks(a.total, chunk)
    out = []
    for i, (s, d) in enumerate(plan):
        setp(job, pct=1 + 7 * i / len(plan), msg=f"অডিও আলাদা করা হচ্ছে: অংশ {i + 1}/{len(plan)}")
        path = os.path.join(workdir, f"chunk_{i:03d}.m4a")
        real_start = a.write_m4a(s, s + d, path)
        out.append({"path": path, "start": real_start, "dur": d})
    return out


def prepare_audio(src, workdir, clean, job):
    ext = os.path.splitext(src)[1].lower()
    ffmpeg = find_ffmpeg()
    if ffmpeg: return extract_chunks_ffmpeg(ffmpeg, src, workdir, clean, job), "ffmpeg"
    if ext in ISO_EXTS:
        try: return extract_chunks_builtin(src, workdir, job), "builtin"
        except NeedFfmpeg as e: raise RuntimeError(f"এই ফাইল ({e}) থেকে অডিও বের করতে ffmpeg লাগবে। পিসিতে চালান, বা MP4 ফাইল দিন")
    if ext in (".mp3", ".wav", ".flac", ".ogg", ".webm") and os.path.getsize(src) <= MAX_UPLOAD_BYTES:
        return [{"path": src, "start": 0.0, "dur": None}], "direct"
    raise RuntimeError(f"{ext or 'এই ধরনের'} ফাইল থেকে অডিও বের করতে ffmpeg লাগবে")


def g(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else default


def lang_code(name):
    name = (name or "").strip().lower()
    if not name: return None
    return name if len(name) == 2 else LANG_CODES.get(name)


def build_prompt(context, tail):
    ctx = re.sub(r"\s+", " ", context or "").strip()[:200]
    tl = re.sub(r"\s+", " ", tail or "").strip()[-110:]
    return (ctx + " " + tl).strip() or None


def transcribe_all(job, key, chunks, cfg, pct_from, pct_to, sink):
    n = len(chunks)
    models = ([cfg["model"]] if cfg["model"] else []) + [m for m in STT_MODELS if m != cfg["model"]]
    lang, tail = cfg["language"], ""
    half = OVERLAP_SECS / 2.0
    for i, ch in enumerate(chunks):
        setp(job, pct=pct_from + (pct_to - pct_from) * i / n, msg=f"কথা শনাক্ত হচ্ছে: অংশ {i + 1}/{n}")
        resp = transcribe_file(job, key, ch["path"], models, lang, build_prompt(cfg["context"], tail if cfg["use_tail"] else ""))
        if not lang:
            lang = lang_code(g(resp, "language"))
            sink["language"] = lang or g(resp, "language")
        else: sink["language"] = sink.get("language") or lang
        off = ch["start"]
        lo = off + (half if i > 0 else 0.0)
        hi = (chunks[i + 1]["start"] + half) if i + 1 < n else float("inf")
        bad = []
        for s in g(resp, "segments", []) or []:
            nsp, lp, cr = g(s, "no_speech_prob", 0) or 0, g(s, "avg_logprob", 0) or 0, g(s, "compression_ratio", 0) or 0
            a, b = off + float(g(s, "start", 0) or 0), off + float(g(s, "end", 0) or 0)
            text = (g(s, "text") or "").strip()
            if not text or (nsp > 0.6 and lp < -1.0) or cr > 2.6:
                bad.append((a, b))
                continue
            if lo <= (a + b) / 2 < hi:
                sink["segs"].append({"start": a, "end": b, "text": text})
                sink["lp"].append(lp)
                if lp < -0.9 or cr > 2.2: sink["review"].append((lp, a, text))
        for w in g(resp, "words", []) or []:
            word = (g(w, "word") or "").strip()
            a, b = off + float(g(w, "start", 0) or 0), off + float(g(w, "end", 0) or 0)
            if not word or not (lo <= a < hi) or any(x <= a <= y for x, y in bad): continue
            sink["words"].append({"start": a, "end": b, "text": word})
        tail = g(resp, "text", "") or ""
    setp(job, pct=pct_to)


def finalize_cues(cues):
    cues.sort(key=lambda c: c["start"])
    for i, c in enumerate(cues):
        limit = cues[i + 1]["start"] - 0.02 if i + 1 < len(cues) else None
        end = min(c["end"], c["start"] + max(2.5, len(c["text"]) / 10))
        if end - c["start"] < 0.9: end = c["start"] + 0.9
        if limit is not None and end > limit: end = max(c["start"] + 0.1, limit)
        c["end"] = end
    return cues


def collapse_repeats(cues):
    out, run = [], 0
    for c in cues:
        key = re.sub(r"[\W_]+", "", c["text"].lower())
        if out and key and key == re.sub(r"[\W_]+", "", out[-1]["text"].lower()):
            run += 1
            if run >= 2: continue
        else: run = 0
        out.append(c)
    return out


def build_cues_from_words(words):
    if not words: return []
    words = sorted(words, key=lambda w: w["start"])
    sep = "" if has_cjk("".join(w["text"] for w in words[:60])) else " "
    cues, cur = [], []
    def flush():
        if not cur: return
        last = cur[-1]
        cap = last["start"] + max(1.0, 0.09 * len(last["text"]) + 0.4)
        end = max(min(last["end"], cap), cur[0]["start"] + 0.3)
        cues.append({"start": cur[0]["start"], "end": end, "text": sep.join(w["text"] for w in cur).strip()})
        cur.clear()
    for w in words:
        if cur:
            text_len = len(sep.join(x["text"] for x in cur)) + 1 + len(w["text"])
            gap = w["start"] - cur[-1]["end"]
            last = cur[-1]["text"]
            if (gap > 0.8 or text_len > MAX_CHARS or w["end"] - cur[0]["start"] > 6.5
                    or (last[-1:] in ".?!…。！？।" and text_len > 28) or (last[-1:] in ",;:" and text_len > 55)):
                flush()
        cur.append(w)
    flush()
    return finalize_cues(collapse_repeats(cues))


def split_long(s):
    out, cur = [], ""
    for w in s.split(" "):
        if cur and len(cur) + 1 + len(w) > MAX_CHARS:
            out.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    if cur: out.append(cur)
    return out


def split_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    pieces, cur = [], ""
    for s in re.split(r"(?<=[.!?…।])\s+", text):
        if len(s) > MAX_CHARS:
            if cur: pieces.append(cur); cur = ""
            pieces.extend(split_long(s))
        elif cur and len(cur) + 1 + len(s) > MAX_CHARS:
            pieces.append(cur); cur = s
        else: cur = (cur + " " + s).strip()
    if cur: pieces.append(cur)
    return pieces


def build_cues_from_segments(segs):
    cues = []
    for sg in sorted(segs, key=lambda x: x["start"]):
        pieces = split_text(sg["text"])
        if not pieces: continue
        dur, total, t = max(0.3, sg["end"] - sg["start"]), sum(len(p) for p in pieces) or 1, sg["start"]
        for p in pieces:
            d = dur * len(p) / total
            cues.append({"start": t, "end": t + d, "text": p})
            t += d
    return finalize_cues(collapse_repeats(cues))


def make_chunks(items, max_chars):
    chunks, cur, size = [], [], 0
    for it in items:
        n = len(it[1]) + 8
        if cur and size + n > max_chars:
            chunks.append(cur); cur, size = [], 0
        cur.append(it); size += n
    if cur: chunks.append(cur)
    return chunks


def translate_items(job, key, items, models, context, depth=0):
    payload = "\n".join(f"{i}|{t.replace(chr(10), NL)}" for i, t in items)
    est_out = min(30000, int(len(payload) * 1.5) + 300)
    system = TRANSLATE_PROMPT + (f" Context about this video (for names/terms): {context[:300]}" if context else "")
    def call(model):
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        body = {"model": model, "temperature": 0.2, "max_tokens": est_out + (2000 if extra else 0),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": payload}], **extra}
        t0 = time.time()
        r = requests.post(f"{API_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=(15, 300))
        check_response(r)
        data = r.json()
        used = (data.get("usage") or {}).get("total_tokens", 0) or 0
        gap = used * 60.0 / TPM_LIMIT - (time.time() - t0)
        if gap > 0: wait_msg(job, gap, "লিমিট বাঁচাতে")
        return data
    try: data = groq_request(job, call, models)
    except TooLarge:
        if len(items) > 1 and depth < 4:
            mid = len(items) // 2
            res = translate_items(job, key, items[:mid], models, context, depth + 1)
            res.update(translate_items(job, key, items[mid:], models, context, depth + 1))
            return res
        return {}
    out = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    wanted = {i for i, _ in items}
    res = {}
    for ln in out.splitlines():
        m = LINE_RE.match(ln)
        if m and int(m.group(1)) in wanted: res[int(m.group(1))] = m.group(2).strip().replace(NL, "\n")
    return res


def translate_cues(job, key, cues, context, pct_from, pct_to):
    models = list(CHAT_MODELS)
    uniq = {}
    for c in cues:
        if c["text"] not in uniq: uniq[c["text"]] = len(uniq) + 1
    items = [(i, t) for t, i in uniq.items()]
    translated = {}
    chunks = make_chunks(items, TRANSLATE_CHUNK_CHARS)
    for k, chunk in enumerate(chunks):
        setp(job, pct=pct_from + (pct_to - pct_from) * k / len(chunks), msg=f"বাংলা অনুবাদ হচ্ছে: অংশ {k + 1}/{len(chunks)}")
        translated.update(translate_items(job, key, chunk, models, context))
    missing = [(i, t) for i, t in items if i not in translated]
    for chunk in make_chunks(missing, TRANSLATE_CHUNK_CHARS):
        translated.update(translate_items(job, key, chunk, models, context))
    for c in cues: c["bn"] = translated.get(uniq[c["text"]], c["text"])
    return sum(1 for i, _ in items if i in translated), len(items)


def preview_of(cues, field):
    return "\n".join(f"{fmt_time(c['start'])[:8]}  {c[field].replace(chr(10), ' ')}" for c in cues[:12])


def review_lines(sink):
    rv = sorted(sink["review"])[:12]
    return [f"{fmt_time(a)[:8]}  {t[:70]}" for lp, a, t in rv]


def run_job(job_id, key, cfg):
    job = jobs[job_id]
    out_dir, base = job["out_dir"], job["base"]
    sink = {"words": [], "segs": [], "lp": [], "review": [], "language": None}
    cues = []
    try:
        translate = cfg["translate"]
        setp(job, status="running", pct=1, msg="অডিও আলাদা করা হচ্ছে...")
        chunks, method = prepare_audio(job["src"], job["workdir"], cfg["clean"], job)
        if job["owns_src"]:
            try: os.remove(job["src"])
            except OSError: pass
        transcribe_all(job, key, chunks, cfg, 8, 70 if translate else 96, sink)
        cues = build_cues_from_words(sink["words"]) if sink["words"] else build_cues_from_segments(sink["segs"])
        if not cues: raise RuntimeError("কোনো কথা শনাক্ত হয়নি")
        os.makedirs(out_dir, exist_ok=True)
        vtt_path = os.path.join(out_dir, base + ".vtt")
        write_vtt(cues, vtt_path)
        job["files"]["vtt"], job["paths"]["vtt"] = True, vtt_path
        job["preview"] = preview_of(cues, "text")
        job["review"] = review_lines(sink)
        conf = ""
        if sink["lp"]:
            avg = sum(sink["lp"]) / len(sink["lp"])
            conf = f" · নিশ্চয়তা {'ভালো' if avg > -0.4 else 'মাঝারি' if avg > -0.7 else 'কম'}"
        if translate and sink["language"] == "bn": translate = False
        if translate:
            done, total = translate_cues(job, key, cues, cfg["context"], 70, 97)
            bn_path = os.path.join(out_dir, base + ".bn.vtt")
            write_vtt(cues, bn_path, "bn")
            job["files"]["bn"], job["paths"]["bn"] = True, bn_path
            job["preview"] = preview_of(cues, "bn")
            setp(job, status="done", pct=100, msg=f"সম্পন্ন — {len(cues)}টা লাইন{conf}")
        else:
            setp(job, status="done", pct=100, msg=f"সম্পন্ন — {len(cues)}টা লাইন{conf}")
    except Exception as e:
        reason = str(e)[:240] or type(e).__name__
        if job["files"]["vtt"]: setp(job, status="partial", msg="আংশিক সম্পন্ন — " + reason)
        else: setp(job, status="error", msg=reason)
    finally: shutil.rmtree(job["workdir"], ignore_errors=True)


def worker():
    while True:
        job_id, key, cfg = job_q.get()
        try: run_job(job_id, key, cfg)
        except Exception as e: setp(jobs[job_id], status="error", msg=str(e)[:200])
        finally: job_q.task_done()


threading.Thread(target=worker, daemon=True).start()


def make_job(display_name, src, owns_src, next_to, key, cfg):
    job_id = uuid.uuid4().hex[:10]
    stem = os.path.splitext(os.path.basename(display_name))[0]
    if next_to: out_dir, base = os.path.dirname(src), os.path.splitext(os.path.basename(src))[0]
    else:
        base = safe_name(stem)
        out_dir = os.path.join(OUTPUT_DIR, base)
        if os.path.exists(out_dir): out_dir = f"{out_dir}_{job_id[:4]}"
    workdir = os.path.join(WORK_DIR, job_id)
    os.makedirs(workdir, exist_ok=True)
    jobs[job_id] = {
        "id": job_id, "name": display_name, "base": base, "out_dir": out_dir, "workdir": workdir, "src": src,
        "owns_src": owns_src, "status": "queued", "msg": "লাইনে আছে...", "pct": 0,
        "files": {"vtt": False, "bn": False}, "paths": {}, "preview": "", "review": [], "created": time.time(),
    }
    order.insert(0, job_id)
    job_q.put((job_id, key, cfg))
    return job_id


def read_cfg(d):
    return {"language": (d.get("language") or "").strip(), "model": (d.get("model") or "").strip(),
            "translate": str(d.get("translate")) in ("1", "true", "True"), "context": (d.get("context") or "").strip()[:300],
            "clean": str(d.get("clean", "1")) in ("1", "true", "True"), "use_tail": True}


def scan_media(root, overwrite):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith("$") and d != "System Volume Information"]
        for f in sorted(filenames):
            if f.startswith("._") or os.path.splitext(f)[1].lower() not in MEDIA_EXTS: continue
            full = os.path.join(dirpath, f)
            if not overwrite and os.path.exists(os.path.splitext(full)[0] + ".vtt"): continue
            found.append(full)
    return found


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/api/test_key", methods=["POST"])
def api_test_key():
    key = clean_key((request.get_json(force=True) or {}).get("api_key"))
    if not key: return jsonify(ok=False, msg="key লিখুন"), 400
    try:
        r = requests.get(f"{API_BASE}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        if r.status_code == 401: return jsonify(ok=False, msg="key ভুল বা মেয়াদ শেষ")
        r.raise_for_status()
        return jsonify(ok=True, msg="key ঠিক আছে ✓")
    except Exception as e: return jsonify(ok=False, msg=f"পরীক্ষা ব্যর্থ: {str(e)[:120]}")


@app.route("/api/info")
def api_info():
    return jsonify(ffmpeg=bool(find_ffmpeg()), android=IS_ANDROID, output_dir=OUTPUT_DIR)


@app.route("/api/submit", methods=["POST"])
def api_submit():
    key = clean_key(request.form.get("api_key"))
    f = request.files.get("file")
    if not key: return jsonify(error="API key লিখুন"), 400
    if not f or not f.filename: return jsonify(error="কোনো ভিডিও বাছা হয়নি"), 400
    ext = os.path.splitext(f.filename)[1].lower()
    tmp_dir = os.path.join(WORK_DIR, "uploads_" + uuid.uuid4().hex[:6])
    os.makedirs(tmp_dir, exist_ok=True)
    src = os.path.join(tmp_dir, "input" + (ext[:8] or ".bin"))
    f.save(src)
    cfg = read_cfg(request.form)
    jid = make_job(f.filename, src, True, False, key, cfg)
    jobs[jid]["workdir_extra"] = tmp_dir
    return jsonify(ok=True, id=jid)


@app.route("/api/submit_paths", methods=["POST"])
def api_submit_paths():
    d = request.get_json(force=True) or {}
    key = clean_key(d.get("api_key"))
    if not key: return jsonify(error="API key লিখুন"), 400
    lines = [p.strip().strip('"').strip("'") for p in (d.get("paths") or "").splitlines() if p.strip()]
    if not lines: return jsonify(error="কোনো ফাইল/ফোল্ডারের পাথ দেওয়া হয়নি"), 400
    overwrite = str(d.get("overwrite")) in ("1", "true", "True")
    next_to = str(d.get("next_to", "1")) in ("1", "true", "True")
    files, missing = [], []
    for p in lines:
        if os.path.isdir(p): files += scan_media(p, overwrite)
        elif os.path.isfile(p):
            if not overwrite and os.path.exists(os.path.splitext(p)[0] + ".vtt") and next_to: continue
            files.append(p)
        else: missing.append(p)
    if not files: return jsonify(error="অনুবাদের জন্য কোনো নতুন ফাইল পাওয়া যায়নি"), 400
    cfg = read_cfg(d)
    ids = [make_job(os.path.basename(p), p, False, next_to, key, cfg) for p in files[:2000]]
    return jsonify(ok=True, count=len(ids))


@app.route("/api/jobs")
def api_jobs():
    hide = {"src", "workdir", "workdir_extra", "out_dir", "base", "created", "owns_src", "paths"}
    return jsonify([{k: v for k, v in jobs[i].items() if k not in hide} for i in order])


@app.route("/download/<job_id>/<kind>")
def download(job_id, kind):
    job = jobs.get(job_id)
    if not job or kind not in ("vtt", "bn") or not job["files"].get(kind): abort(404)
    path = job["paths"].get(kind)
    if not path or not os.path.isfile(path): abort(404)
    suffix = ".vtt" if kind == "vtt" else ".bn.vtt"
    return send_file(path, mimetype="text/vtt", as_attachment=True, download_name=job["base"] + suffix)


HTML = r"""<!DOCTYPE html>
<html lang="bn"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>সাবটাইটেল জেনারেটর</title>
<style>
 *{box-sizing:border-box}
 body{font-family:Arial,sans-serif;background:#121212;color:#fff;margin:0;padding:14px}
 h2{color:#ff4757;text-align:center;margin:6px 0 14px}
 .box{background:#1e1e1e;border-radius:10px;padding:14px;max-width:820px;margin:0 auto 14px}
 label{display:block;margin:12px 0 6px;font-size:14px;color:#dfe4ea}
 .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 input[type=text],input[type=password],select,textarea{flex:1;min-width:140px;width:100%;padding:12px;border:1px solid #333;background:#2f3542;color:#fff;border-radius:6px;outline:none;font-size:16px;font-family:inherit}
 textarea{min-height:70px;resize:vertical}
 button{background:#ff4757;color:#fff;border:none;padding:12px 16px;border-radius:6px;cursor:pointer;font-weight:bold;font-size:15px}
 button:hover{background:#ff6b81} button:disabled{background:#555;cursor:default}
 button.sec{background:#3d4556} button.sec:hover{background:#4b556b}
 .chk{display:flex;align-items:center;gap:8px;font-size:13px;color:#a4b0be;margin:8px 0}
 .chk input{width:20px;height:20px}
 .msg{font-size:13px;color:#a4b0be;min-height:1.2em;margin-top:6px;word-break:break-word}
 .ok{color:#2ed573}.bad{color:#ff6b81}
 #drop{border:2px dashed #444;border-radius:10px;padding:22px;text-align:center;color:#a4b0be;cursor:pointer}
 #drop.over{border-color:#ff4757;color:#fff}
 .bar{height:10px;background:#2f3542;border-radius:5px;overflow:hidden;margin:8px 0}
 .bar>div{height:100%;width:0;background:#2ed573;transition:width .4s}
 .bar.up>div{background:#1e90ff}
 .job{background:#171717;border:1px solid #2a2a2a;border-radius:8px;padding:12px;margin-top:10px}
 .job h4{margin:0 0 4px;font-size:14px;word-break:break-all}
 .job a{display:inline-block;margin:8px 8px 0 0;background:#2ed573;color:#000;text-decoration:none;padding:9px 12px;border-radius:6px;font-size:13px;font-weight:bold}
 .job a.bn{background:#ffa502}
 details{margin-top:8px;font-size:12px;color:#a4b0be}
 pre{white-space:pre-wrap;word-break:break-word;background:#0d0d0d;border-radius:6px;padding:8px;margin:6px 0 0;max-height:200px;overflow:auto}
 .tabs{display:flex;gap:8px;margin-bottom:10px}.tabs button{flex:1}.tabs button.on{background:#ff4757}.tabs button:not(.on){background:#3d4556}
 .info{font-size:12px;color:#7f8c8d;margin-top:6px}
</style></head><body>
<h2>🎬 ভিডিও → সাবটাইটেল জেনারেটর</h2>

<div class="box">
  <label>Groq API Key</label>
  <div class="row">
    <input id="key" type="password" placeholder="gsk_..." autocomplete="off">
    <button class="sec" id="showkey" type="button">দেখান</button>
    <button class="sec" id="testkey" type="button">Key পরীক্ষা</button>
  </div>
  <label class="chk"><input type="checkbox" id="remember"> এই ব্রাউজারে key মনে রাখো</label>
  <div class="msg" id="keyMsg"></div>
</div>

<div class="box">
  <div class="tabs"><button id="tabPath" class="on" type="button">📂 ফাইলের পাথ</button><button id="tabUp" type="button">⬆ ফাইল আপলোড</button></div>

  <div id="paneUp" style="display:none">
    <div id="drop">📁 ভিডিও বাছুন (ক্লিক করুন বা টেনে আনুন)</div>
    <input id="file" type="file" accept="video/*,audio/*,.mkv,.ts" multiple style="display:none">
    <div class="msg" id="picked"></div>
  </div>

  <div id="panePath">
    <label>ভিডিওর পাথ — প্রতি লাইনে একটা</label>
    <textarea id="paths" placeholder="/path/to/video.mp4"></textarea>
    <label class="chk"><input type="checkbox" id="nextto" checked> সাবটাইটেল ভিডিওর পাশেই সেভ করো</label>
    <label class="chk"><input type="checkbox" id="overwrite"> আগের .vtt থাকলেও নতুন করে বানাও</label>
  </div>
</div>

<div class="box">
  <b>সেটিংস</b>
  <div class="row" style="margin-top:6px">
    <div style="flex:1;min-width:150px">
      <label>ভিডিওর কথার ভাষা</label>
      <select id="lang">
        <option value="">অটো-ডিটেক্ট</option><option value="en">English</option><option value="hi">Hindi</option><option value="bn">Bengali</option>
      </select>
    </div>
    <div style="flex:1;min-width:150px">
      <label>মডেল</label>
      <select id="model">
        <option value="whisper-large-v3">whisper-large-v3 (সেরা)</option>
        <option value="whisper-large-v3-turbo">whisper-large-v3-turbo (দ্রুত)</option>
      </select>
    </div>
  </div>
  <label>প্রসঙ্গ / নাম (ঐচ্ছিক)</label>
  <textarea id="context" maxlength="300" placeholder="যেমন: Names: Rahul"></textarea>
  <label class="chk"><input type="checkbox" id="clean" checked> অডিও পরিষ্কার করো</label>
  <label class="chk"><input type="checkbox" id="tr" checked> বাংলায় অনুবাদও করো (.bn.vtt)</label>
  
  <button id="go" type="button" style="width:100%;margin-top:14px;background:#2ed573;color:#000;font-size:18px;padding:16px;">🚀 প্রসেস শুরু করুন (Start Process)</button>
  
  <div class="bar up"><div id="upbar"></div></div>
  <div class="msg" id="uptxt"></div>
</div>

<div class="box"><b>কাজের তালিকা</b><div id="jobs"><div class="msg">এখনও কোনো কাজ নেই</div></div></div>

<script>
const $ = (id) => document.getElementById(id);
let mode = "path";
try { const k = localStorage.getItem("groq_key"); if (k) { $("key").value = k; $("remember").checked = true; }
      const p = localStorage.getItem("sub_paths"); if (p) $("paths").value = p; } catch (e) {}
function remember() {
  try { if ($("remember").checked) localStorage.setItem("groq_key", $("key").value.trim()); else localStorage.removeItem("groq_key");
        localStorage.setItem("sub_paths", $("paths").value); } catch (e) {}
}
async function post(url, body) {
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
  return r.json();
}
function setMode(m) {
  mode = m; $("panePath").style.display = m === "path" ? "block" : "none"; $("paneUp").style.display = m === "up" ? "block" : "none";
  $("tabPath").className = m === "path" ? "on" : ""; $("tabUp").className = m === "up" ? "on" : "";
}
$("tabPath").onclick = () => setMode("path"); $("tabUp").onclick = () => setMode("up");

$("key").oninput = () => {
  const n = $("key").value.trim().length; $("keyMsg").className = "msg";
  $("keyMsg").textContent = n ? "key-এর দৈর্ঘ্য: " + n + " অক্ষর" : "";
};
$("showkey").onclick = () => { const t = $("key").type === "password"; $("key").type = t ? "text" : "password"; $("showkey").textContent = t ? "লুকান" : "দেখান"; };
$("testkey").onclick = async () => {
  $("keyMsg").className = "msg"; $("keyMsg").textContent = "পরীক্ষা হচ্ছে...";
  const d = await post("/api/test_key", {api_key: $("key").value.trim()});
  $("keyMsg").textContent = d.msg; $("keyMsg").className = "msg " + (d.ok ? "ok" : "bad"); if (d.ok) remember();
};

const drop = $("drop");
drop.onclick = () => $("file").click();
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { $("file").files = e.dataTransfer.files; showPicked(); } });
$("file").onchange = showPicked;
function showPicked() {
  const fs = Array.from($("file").files);
  $("picked").textContent = fs.length ? fs.length + "টা ফাইল: " + fs.map((f) => f.name + " (" + (f.size / 1048576).toFixed(0) + " MB)").join(", ") : "";
}
function opts() {
  return {language: $("lang").value, model: $("model").value, translate: $("tr").checked ? "1" : "0",
          context: $("context").value.trim(), clean: $("clean").checked ? "1" : "0"};
}
function uploadOne(file, idx, total) {
  return new Promise((resolve) => {
    const fd = new FormData(); const o = opts();
    fd.append("api_key", $("key").value.trim()); Object.keys(o).forEach((k) => fd.append(k, o[k])); fd.append("file", file);
    const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/submit");
    xhr.upload.onprogress = (e) => { if (!e.lengthComputable) return; const p = Math.round(e.loaded * 100 / e.total);
      $("upbar").style.width = p + "%"; $("uptxt").className = "msg"; $("uptxt").textContent = "আপলোড (" + idx + "/" + total + "): " + file.name + " — " + p + "%"; };
    xhr.onload = () => { let d = {}; try { d = JSON.parse(xhr.responseText); } catch (e) {}
      if (xhr.status !== 200 || d.error) { $("uptxt").className = "msg bad"; $("uptxt").textContent = "Error: " + (d.error || xhr.status); resolve(false); } else resolve(true); };
    xhr.onerror = () => { $("uptxt").className = "msg bad"; $("uptxt").textContent = "আপলোড ব্যর্থ"; resolve(false); };
    xhr.send(fd);
  });
}
$("go").onclick = async () => {
  if (!$("key").value.trim()) { $("keyMsg").className = "msg bad"; $("keyMsg").textContent = "আগে API key লিখুন"; return; }
  remember(); $("go").disabled = true;
  try {
    if (mode === "path") {
      const o = opts(); o.api_key = $("key").value.trim(); o.paths = $("paths").value; o.next_to = $("nextto").checked ? "1" : "0"; o.overwrite = $("overwrite").checked ? "1" : "0";
      const d = await post("/api/submit_paths", o);
      $("uptxt").className = "msg " + (d.error ? "bad" : "ok");
      $("uptxt").textContent = d.error ? "Error: " + d.error : d.count + "টা কাজ কিউতে যোগ হয়েছে!";
    } else {
      const fs = Array.from($("file").files);
      if (!fs.length) { $("uptxt").className = "msg bad"; $("uptxt").textContent = "আগে ভিডিও বাছুন"; }
      else {
        let ok = 0;
        for (let i = 0; i < fs.length; i++) { if (await uploadOne(fs[i], i + 1, fs.length)) ok++; else break; refresh(); }
        if (ok === fs.length) { $("uptxt").className = "msg ok"; $("uptxt").textContent = ok + "টা ফাইল কিউতে যোগ হয়েছে!"; $("file").value = ""; $("picked").textContent = ""; }
      }
    }
  } catch (e) { $("uptxt").className = "msg bad"; $("uptxt").textContent = "Error: " + e; }
  $("go").disabled = false; refresh();
};

const ST = {queued: "লাইনে আছে", running: "চলছে", done: "সম্পন্ন ✓", partial: "আংশিক ⚠", error: "ব্যর্থ ✗"};
function renderJobs(list) {
  const box = $("jobs"); box.textContent = "";
  if (!list.length) { const m = document.createElement("div"); m.className = "msg"; m.textContent = "এখনও কোনো কাজ নেই"; box.appendChild(m); return; }
  list.forEach((j) => {
    const c = document.createElement("div"); c.className = "job";
    const h = document.createElement("h4"); h.textContent = j.name + "  —  " + (ST[j.status] || j.status); c.appendChild(h);
    if (j.status === "running" || j.status === "queued") { const b = document.createElement("div"); b.className = "bar"; const i = document.createElement("div"); i.style.width = j.pct + "%"; b.appendChild(i); c.appendChild(b); }
    const m = document.createElement("div"); m.className = "msg " + (j.status === "error" ? "bad" : j.status === "partial" ? "warn" : j.status === "done" ? "ok" : "");
    m.textContent = j.msg; c.appendChild(m);
    if (j.files.vtt) { const a = document.createElement("a"); a.href = "/download/" + j.id + "/vtt"; a.textContent = "⬇ মূল (.vtt)"; c.appendChild(a); }
    if (j.files.bn) { const a = document.createElement("a"); a.className = "bn"; a.href = "/download/" + j.id + "/bn"; a.textContent = "⬇ বাংলা (.bn.vtt)"; c.appendChild(a); }
    if (j.preview) {
      const d = document.createElement("details"); const s = document.createElement("summary"); s.textContent = "প্রিভিউ";
      const p = document.createElement("pre"); p.textContent = j.preview; d.append(s, p); c.appendChild(d);
    }
    box.appendChild(c);
  });
}
async function refresh() {
  try { const list = await (await fetch("/api/jobs")).json(); renderJobs(list); } catch (e) {}
}
setInterval(refresh, 1500); refresh();
</script></body></html>
"""

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)