import argparse
import csv
import dataclasses
import datetime as dt
import json
import logging
import pathlib
import queue
import re
import shutil
import sqlite3
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


# ============================== CONSTANTES ===============================

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".m4v", ".3gp", ".flv"}
DB_NAME = "estado_descargas.sqlite"
CSV_NAME = "resultados.csv"
TXT_LOG = "consola.txt"
TMP_DIRNAME = "_tmp"
CLIENTS_ROTATION = ["web", "ios", "android", "tv", "mweb"]

# patrón robusto para archivos srt EN INGLÉS (excluye live chat)
RE_srt_EN = re.compile(r"""
    \.srt$                          # extensión .srt
""", re.IGNORECASE | re.VERBOSE)
RE_srt_LIVECHAT = re.compile(r"\blive[_\-]?chat\b", re.IGNORECASE)


# ================================ UTILS ==================================

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def extraer_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    return url.strip()

def detectar_ffmpeg(ffmpeg_location: Optional[str]) -> Optional[str]:
    if ffmpeg_location:
        p = pathlib.Path(ffmpeg_location)
        if p.is_file():
            return str(p)
        found = shutil.which("ffmpeg", path=str(p))
        return found
    return shutil.which("ffmpeg")

def leer_ids(path_ids: pathlib.Path) -> List[str]:
    urls = []
    with path_ids.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            if "youtube.com" in t or "youtu.be" in t:
                urls.append(t)
            else:
                urls.append(f"https://www.youtube.com/watch?v={t}")
    seen, dedup = set(), []
    for u in urls:
        vid = extraer_id(u)
        if vid and vid not in seen:
            seen.add(vid)
            dedup.append(u)
    return dedup


# =============================== PERSISTENCIA ============================

class DB:
    def __init__(self, path_db: pathlib.Path):
        self.path_db = path_db
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path_db, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self):
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY, value TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS items(
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            status TEXT NOT NULL,     -- 'pending' | 'done' | 'failed'
            has_video INTEGER NOT NULL DEFAULT 0,
            has_subs INTEGER NOT NULL DEFAULT 0,   -- SOLO EN (inglés)
            error TEXT,
            tries INTEGER NOT NULL DEFAULT 0,
            last_update REAL NOT NULL
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_status ON items(status)")
        self._conn.commit()

    def set_total(self, n: int):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('total',?)", (str(n),))
            self._conn.commit()

    def get_total(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT value FROM meta WHERE key='total'")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def upsert_pending_ids(self, pairs: List[Tuple[str, str]]):
        ts = time.time()
        with self._lock:
            for vid, url in pairs:
                self._conn.execute("""
                    INSERT INTO items(id,url,status,last_update)
                    VALUES(?,?, 'pending', ?)
                    ON CONFLICT(id) DO NOTHING
                """, (vid, url, ts))
            self._conn.commit()

    def next_pending(self, limit: int) -> List[Tuple[str, str]]:
        with self._lock:
            cur = self._conn.execute("""
                SELECT id,url FROM items
                WHERE status='pending'
                LIMIT ?
            """, (limit,))
            return list(cur.fetchall())

    def mark(self, vid: str, status: str, has_video: bool, has_subs_en: bool, error: Optional[str], tries_inc: int = 1):
        ts = time.time()
        with self._lock:
            self._conn.execute("""
                UPDATE items
                SET status=?, has_video=?, has_subs=?, error=?, tries=tries+?, last_update=?
                WHERE id=?
            """, (status, int(has_video), int(has_subs_en), error, tries_inc, ts, vid))
            self._conn.commit()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            d = self._conn.execute("SELECT COUNT(*) FROM items WHERE status='done'").fetchone()[0]
            f = self._conn.execute("SELECT COUNT(*) FROM items WHERE status='failed'").fetchone()[0]
            p = self._conn.execute("SELECT COUNT(*) FROM items WHERE status='pending'").fetchone()[0]
        return {"done": d, "failed": f, "pending": p, "total": self.get_total()}

    def close(self):
        with self._lock:
            self._conn.close()


class CSVLogger:
    def __init__(self, path_csv: pathlib.Path):
        self.path_csv = path_csv
        self._lock = threading.Lock()
        ensure_dir(self.path_csv.parent)
        if not self.path_csv.exists():
            with self.path_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "status", "has_video", "has_subs_en", "error", "timestamp"])

    def write_row(self, vid: str, status: str, has_video: bool, has_subs_en: bool, error: str):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock, self.path_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([vid, status, int(has_video), int(has_subs_en), error or "", ts])


def configure_text_logger(path_txt: pathlib.Path) -> logging.Logger:
    ensure_dir(path_txt.parent)
    logger = logging.getLogger("yt-en")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(path_txt, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ========================== VERIFICACIÓN EN DISCO ========================

def has_video(dir_root: pathlib.Path, vid: str) -> bool:
    d = dir_root / vid
    if not d.exists():
        return False
    for p in d.glob(f"{vid}.*"):
        if p.suffix.lower() in VIDEO_EXTS and not p.name.endswith(".part"):
            return True
    return False

def has_english_subs(dir_root: pathlib.Path, vid: str) -> bool:
    """
    True si existe al menos un .srt EN inglés en la carpeta FINAL del video.
    Se excluyen srt de live chat.
    """
    d = dir_root / vid
    if not d.exists():
        return False
    for p in d.glob(f"{vid}*.srt"):
        name = p.name
        if RE_srt_LIVECHAT.search(name):
            continue
        if RE_srt_EN.search(name):
            return True
    return False


# =========================== SELECCIÓN DE FORMATOS ========================

def _is_prog(f: dict) -> bool:
    return (f.get("vcodec") not in (None, "none")) and (f.get("acodec") not in (None, "none"))

def _qkey(f: dict) -> tuple:
    return (f.get("height") or 0, f.get("fps") or 0, f.get("tbr") or 0)

def _best_prog(formats: List[dict]) -> Optional[dict]:
    progs = [f for f in formats if _is_prog(f)]
    if not progs:
        return None
    mp4 = sorted([f for f in progs if f.get("ext") == "mp4"], key=_qkey, reverse=True)
    if mp4: return mp4[0]
    webm = sorted([f for f in progs if f.get("ext") == "webm"], key=_qkey, reverse=True)
    if webm: return webm[0]
    return sorted(progs, key=_qkey, reverse=True)[0]

def _best_v_only(formats: List[dict]) -> Optional[dict]:
    vs = [f for f in formats if (f.get("vcodec") not in (None, "none")) and (f.get("acodec") in (None, "none"))]
    if not vs: return None
    return sorted(vs, key=_qkey, reverse=True)[0]

def _best_a_only(formats: List[dict]) -> Optional[dict]:
    a = [f for f in formats if (f.get("acodec") not in (None, "none")) and (f.get("vcodec") in (None, "none"))]
    if not a: return None
    return sorted(a, key=lambda f: (f.get("tbr") or 0), reverse=True)[0]

def select_format_string(info: dict, ffmpeg_bin: Optional[str]) -> Tuple[Optional[str], str]:
    fmts = info.get("formats") or []
    if not fmts:
        return (None, "sin_formats")
    prog = _best_prog(fmts)
    if prog:
        return (prog.get("format_id"), "progresivo")
    v = _best_v_only(fmts)
    if not v:
        return (None, "sin_video")
    if ffmpeg_bin:
        a = _best_a_only(fmts)
        if a:
            return (f"{v.get('format_id')}+{a.get('format_id')}", "video_plus_audio")
    return (v.get("format_id"), "solo_video")


# ========================= METADATA ROBUSTA (ROTACIÓN) ====================

def get_info_rotating(url: str,
                      cookies: Optional[pathlib.Path],
                      proxy: Optional[str],
                      max_tries: int = 5) -> Tuple[Optional[dict], Optional[str]]:
    last_exc = None
    for i, client in enumerate(CLIENTS_ROTATION[:max_tries]):
        opts = {
            "quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": False, "noplaylist": True,
            "force_ipv4": True, "geo_bypass": True, "geo_bypass_country": "US",
            "extractor_retries": 3,
            "retry_sleep_functions": {
                "http": {"function": "exponential", "initial": 1, "max": 6, "jitter": "random"},
                "extractor": {"function": "exponential", "initial": 1, "max": 6, "jitter": "random"},
            },
            "proxy": proxy or None,
            "cookies": str(cookies) if cookies else None,
            "extractor_args": {"youtube": {"player_client": [client], "fetch_pot": ["auto"]}},
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    return info, client
        except Exception as e:
            last_exc = e
            time.sleep(0.35 * (i + 1))
            continue
    return None, None


# ========================= DESCARGA / SUBS EN (srt) ======================

def build_ydl_opts(tmp_base: pathlib.Path, vid: str, client: str, ffmpeg_bin: Optional[str],
                   fmt: str, retries: int, cookies: Optional[pathlib.Path], proxy: Optional[str]) -> dict:
    outtmpl = str((tmp_base / vid / "%(id)s.%(ext)s").as_posix())
    subs_en = ["en", "en.*", "en-", "ase", "asl"]  # *solo inglés*
    opts = {
        "outtmpl": outtmpl,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": subs_en,
        "subtitlesformat": "srt",       # sin conversión -> no depende de ffmpeg
        "ignoreerrors": True,
        "ignore_no_formats_error": True,
        "retries": retries,
        "fragment_retries": retries,
        "noprogress": False,
        "nopart": False,
        "continuedl": True,
        "concurrent_fragment_downloads": 4,
        "force_ipv4": True,
        "geo_bypass": True,
        "geo_bypass_country": "US",
        "proxy": proxy or None,
        "cookies": str(cookies) if cookies else None,
        "http_headers": {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
        "extractor_args": {"youtube": {"player_client": [client], "fetch_pot": ["auto"]}},
        "writeinfojson": False,
        "writedescription": False,
        "skip_download": False,
        # Anti “Requested format…”
        "format": fmt,
        "check_formats": "selected",
        "format_sort": ["res", "fps", "br", "size", "ext"],
        "hls_prefer_native": False,
        "quiet": True,
        "no_warnings": True,
    }
    if ffmpeg_bin:
        opts["ffmpeg_location"] = ffmpeg_bin
        opts["merge_output_format"] = "mp4"
    return opts

def descargar_solo_subs_en(url: str, destino_dir: pathlib.Path,
                           cookies: Optional[pathlib.Path], proxy: Optional[str]) -> bool:
    """
    Segundo pase: descarga SOLO subtítulos EN (manual o auto) en srt directo al destino.
    """
    vid = extraer_id(url)
    outtmpl = str((destino_dir / "%(id)s.%(ext)s").as_posix())
    subs_en = ["en", "en.*", "en-", "ase", "asl"]
    for client in CLIENTS_ROTATION:
        opts = {
            "outtmpl": outtmpl,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": subs_en,
            "subtitlesformat": "srt",
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "force_ipv4": True,
            "geo_bypass": True,
            "geo_bypass_country": "US",
            "proxy": proxy or None,
            "cookies": str(cookies) if cookies else None,
            "extractor_args": {"youtube": {"player_client": [client], "fetch_pot": ["auto"]}},
        }
        try:
            with YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={vid}"])
        except Exception:
            pass
        # re-chequeo en DESTINO
        if has_english_subs(destino_dir.parent, destino_dir.name):  # destino_dir = salida_root/VID
            return True
        # fallback: chequeo directo de archivos
        for p in destino_dir.glob(f"{vid}*.srt"):
            if not RE_srt_LIVECHAT.search(p.name) and RE_srt_EN.search(p.name):
                return True
    return False


# ========================= DESCARGA ATÓMICA POR VIDEO =====================

def limpiar_tmp(tmp_root: pathlib.Path, logger: logging.Logger):
    if tmp_root.exists():
        for p in tmp_root.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                logger.info(f"Limpieza temporal: {p.name}")

def mover_a_final(tmp_dir: pathlib.Path, final_dir: pathlib.Path):
    ensure_dir(final_dir.parent)
    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    ensure_dir(final_dir)
    for p in tmp_dir.iterdir():
        shutil.move(str(p), str(final_dir / p.name))
    shutil.rmtree(tmp_dir, ignore_errors=True)

def descargar_uno(url: str, salida_root: pathlib.Path, tmp_root: pathlib.Path,
                  cookies: Optional[pathlib.Path], proxy: Optional[str],
                  ffmpeg_bin: Optional[str], retries: int) -> Tuple[bool, bool, Optional[str]]:
    vid = extraer_id(url)
    tmp_dir = tmp_root / vid
    final_dir = salida_root / vid

    # 1) Metadata (rotación + cookies/proxy)
    info, client_used = get_info_rotating(url, cookies, proxy)
    if not info:
        return (False, False, "metadata_failed")

    # 2) Selección de formato existente (acepta TODO)
    fmt, _ = select_format_string(info, ffmpeg_bin)
    if not fmt:
        return (False, False, "no_formats_available")

    # 3) Descargar en TMP (video + intento de subs EN)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    ensure_dir(tmp_dir)
    ydl_opts = build_ydl_opts(tmp_root, vid, client_used or "web",
                              ffmpeg_bin, fmt, retries, cookies, proxy)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception:
        # seguimos; validación por archivos
        pass

    # 4) Validación en TMP
    hv_tmp = any((p.suffix.lower() in VIDEO_EXTS and not p.name.endswith(".part"))
                 for p in tmp_dir.glob(f"{vid}.*"))
    # mover a FINAL solo si hay video utilizable
    if not hv_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return (False, False, "no_usable_video_after_download")

    mover_a_final(tmp_dir, final_dir)

    # 5) Re-chequeo FINAL EN (clave del fix)
    has_subs_en = False
    for p in final_dir.glob(f"{vid}*.srt"):
        if not RE_srt_LIVECHAT.search(p.name) and RE_srt_EN.search(p.name):
            has_subs_en = True
            break

    # 6) Si aún no hay .srt EN, segundo pase SOLO SUBS EN directo al destino
    if not has_subs_en:
        got = descargar_solo_subs_en(url, final_dir, cookies, proxy)
        # re-chequeo definitivo en destino
        has_subs_en = False
        for p in final_dir.glob(f"{vid}*.srt"):
            if not RE_srt_LIVECHAT.search(p.name) and RE_srt_EN.search(p.name):
                has_subs_en = True
                break

    return (True, has_subs_en, None)


# =============================== WORKER POOL ==============================

@dataclasses.dataclass
class Job:
    vid: str
    url: str

def worker(db: DB, csvlog: CSVLogger, q: "queue.Queue[Optional[Job]]", stop_event: threading.Event,
           salida_root: pathlib.Path, tmp_root: pathlib.Path,
           cookies: Optional[pathlib.Path], proxy: Optional[str],
           ffmpeg_bin: Optional[str], retries: int, logger: logging.Logger):
    while not stop_event.is_set():
        try:
            job = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if job is None:
            break

        vid, url = job.vid, job.url

        # Si ya existen video y srt EN en la CARPETA FINAL, omitir
        if has_video(salida_root, vid) and has_english_subs(salida_root, vid):
            db.mark(vid, "done", True, True, None, tries_inc=0)
            csvlog.write_row(vid, "done", True, True, "")
            logger.info(f"[OMITIDO] {vid} ya presente con subtítulos EN")
            continue

        hv, hs_en, err = descargar_uno(url, salida_root, tmp_root, cookies, proxy, ffmpeg_bin, retries)
        # Re-chequeo final en disco ANTES de registrar (para evitar falsos negativos)
        hv_final = has_video(salida_root, vid)
        hs_final = has_english_subs(salida_root, vid)
        hv = hv and hv_final
        hs_en = hs_en or hs_final

        if err:
            db.mark(vid, "failed", hv, hs_en, err)
            csvlog.write_row(vid, "failed", hv, hs_en, err)
            logger.warning(f"[FALLO] {vid}: {err}")
        else:
            db.mark(vid, "done", hv, hs_en, None)
            csvlog.write_row(vid, "done", hv, hs_en, "")
            logger.info(f"[OK] {vid} {'CON' if hs_en else 'SIN'} subtítulos en inglés")


# ================================== UI ===================================

def ui_thread(db: DB, stop_event: threading.Event):
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return
    root = tk.Tk()
    root.title("Descargas YT")
    root.geometry("380x170")

    lbl = tk.Label(root, text="Progreso:", font=("Arial", 12))
    lbl.pack(pady=6)
    pb = ttk.Progressbar(root, orient="horizontal", length=320, mode="determinate")
    pb.pack(pady=4)
    pct_lbl = tk.Label(root, text="0.0%", font=("Arial", 11))
    pct_lbl.pack(pady=2)
    stats_lbl = tk.Label(root, text="", font=("Arial", 10))
    stats_lbl.pack(pady=6)

    def update():
        if stop_event.is_set():
            return
        try:
            s = db.stats()
        except Exception:
            return
        total = max(1, s["total"])
        done = s["done"] + s["failed"]
        pct = 100.0 * done / total
        pb["maximum"] = total
        pb["value"] = done
        pct_lbl.config(text=f"{pct:.1f}%")
        stats_lbl.config(text=f"Total={s['total']}  Pend={s['pending']}  OK={s['done']}  Fail={s['failed']}")
        root.after(700, update)

    def stop():
        stop_event.set()
        pct_lbl.config(text="Deteniendo…")

    btn = tk.Button(root, text="Detener", width=14, height=2, command=stop)
    btn.pack(pady=6)
    root.after(200, update)
    root.protocol("WM_DELETE_WINDOW", stop)
    root.mainloop()


# ================================= MAIN ==================================

def main():
    ap = argparse.ArgumentParser(description="Descarga YT (VIDEO + SUBTÍTULOS EN INGLÉS) robusta y reanudable")
    ap.add_argument("--ids", required=True, type=pathlib.Path, help="Archivo con IDs/URLs (uno por línea)")
    ap.add_argument("--salida", default=pathlib.Path("./data"), type=pathlib.Path, help="Carpeta destino raíz")
    ap.add_argument("--reintentos", type=int, default=6)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--ffmpeg-location", default=None, help="Ruta a FFmpeg (binario o carpeta)")
    ap.add_argument("--cookies", type=pathlib.Path, default=None, help="Ruta a cookies.txt (Netscape)")
    ap.add_argument("--proxy", type=str, default=None, help="Proxy (ej. http://127.0.0.1:8888)")
    ap.add_argument("--ui", action="store_true", help="Mostrar UI con porcentaje persistente")
    args = ap.parse_args()

    salida_root = args.salida.resolve()
    tmp_root = (salida_root / TMP_DIRNAME).resolve()
    ensure_dir(salida_root)
    ensure_dir(tmp_root)

    logger = configure_text_logger(salida_root / TXT_LOG)
    logger.info("== Inicio ==")

    ffmpeg_bin = detectar_ffmpeg(args.ffmpeg_location)
    if ffmpeg_bin:
        logger.info(f"FFmpeg: {ffmpeg_bin}")
    else:
        logger.warning("FFmpeg NO detectado: se aceptará progresivo o VIDEO-ONLY si es lo único disponible.")

    # limpiar temporales de ejecuciones previas
    limpiar_tmp(tmp_root, logger)

    urls = leer_ids(args.ids)
    total = len(urls)
    logger.info(f"Entradas únicas: {total}")

    db = DB(salida_root / DB_NAME)
    db.set_total(total)
    pairs = [(extraer_id(u), u) for u in urls]
    db.upsert_pending_ids(pairs)
    csvlog = CSVLogger(salida_root / CSV_NAME)

    # Marcar como done lo ya presente (video+EN)
    for vid, _ in db.next_pending(10_000_000):
        if has_video(salida_root, vid) and has_english_subs(salida_root, vid):
            db.mark(vid, "done", True, True, None, tries_inc=0)
            csvlog.write_row(vid, "done", True, True, "")

    stop_event = threading.Event()
    if args.ui:
        t_ui = threading.Thread(target=ui_thread, args=(db, stop_event), daemon=True)
        t_ui.start()
        logger.info("UI iniciada")

    # Cola y workers
    q: "queue.Queue[Optional[Job]]" = queue.Queue(maxsize=args.max_concurrent * 4)

    def productor():
        pend = db.next_pending(10_000_000)
        for vid, url in pend:
            if stop_event.is_set(): break
            q.put(Job(vid, url))
        for _ in range(args.max_concurrent):
            q.put(None)

    tp = threading.Thread(target=productor, daemon=True)
    tp.start()

    workers = []
    for _ in range(max(1, args.max_concurrent)):
        t = threading.Thread(
            target=worker,
            args=(db, csvlog, q, stop_event, salida_root, tmp_root,
                  args.cookies, args.proxy, ffmpeg_bin, args.reintentos, logger),
            daemon=True
        )
        t.start()
        workers.append(t)

    try:
        for t in workers:
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()

    tp.join(timeout=2.0)
    stop_event.set()
    time.sleep(0.2)

    s = db.stats()
    logger.info(f"== Fin ==  Total={s['total']}  OK={s['done']}  Fail={s['failed']}  Pend={s['pending']}")
    db.close()


if __name__ == "__main__":
    main()