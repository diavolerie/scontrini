# -*- coding: utf-8 -*-
"""Motore: riconoscimento crocini, allineamento, taglio e OCR dei numeri."""
import math, os, re
from pathlib import Path
import numpy as np
import cv2

# ---------------------------------------------------------------- parametri
# Geometria del ritaglio, misurata sui tuoi ritagli manuali (3.png, 93.png ...)
# alla scala in cui la distanza tra i due crocini e' 1310 px.
DEFAULT_GEOM = dict(ref_d=1310.0, half_w=953.0, top=160.0, bottom=250.0)
# Rettangolo nero/foto stampato, misurato sui tuoi ritagli (alla stessa scala):
# centrato (dx = scarto dal centro del ritaglio), largo 1780, alto 982,
# con il bordo alto 160 px sotto il crocino alto.
RECT = dict(dx=-3.0, w=1780.0, dy=160.0, h=982.0)
# Zona dove cercare il numero, relativa alla distanza D tra i crocini
# (origine = centro del crocino basso, y verso il basso)
OCR_ROI = dict(x=0.14, y0=0.035, y1=0.15)


# ------------------------------------------------------------------ I/O
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img, params=None):
    ok, buf = cv2.imencode(Path(path).suffix, img, params or [])
    if not ok:
        raise RuntimeError("Codifica immagine fallita")
    buf.tofile(str(path))


def imwrite_atomic(path, img, params=None):
    """Come imwrite_unicode ma scrive su un file temporaneo e lo sostituisce solo a scrittura riuscita:
    un errore a meta' (disco pieno, crash) non lascia un file troncato e non distrugge l'esistente."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix, img, params or [])
    if not ok:
        raise RuntimeError("Codifica immagine fallita")
    tmp = path.with_name(path.name + ".tmp")
    try:
        buf.tofile(str(tmp))
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ----------------------------------------------------- template del crocino
def synthetic_mark(size=74):
    img = np.full((size, size), 255, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), 32, 0, 2, cv2.LINE_AA)
    cv2.line(img, (2, c), (size - 3, c), 0, 1, cv2.LINE_AA)
    cv2.line(img, (c, 2), (c, size - 3), 0, 1, cv2.LINE_AA)
    cv2.circle(img, (c, c), 14, 0, -1, cv2.LINE_AA)
    return img


def extract_mark_template(template_path=None):
    """Ritaglia il crocino dal template (il primo in alto); se non ci riesce
    usa un crocino sintetico."""
    if template_path and Path(template_path).is_file():
        img = imread_unicode(template_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, b = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            n, _, st, _ = cv2.connectedComponentsWithStats(b)
            cands = []
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if 30 <= w <= 200 and 0.8 <= w / h <= 1.25 and a > 150:
                    cands.append((y, x, w, h))
            if cands:
                y, x, w, h = min(cands)
                pad = 3
                y0, x0 = max(0, y - pad), max(0, x - pad)
                return img[y0:y + h + pad, x0:x + w + pad].copy()
    return synthetic_mark()


# --------------------------------------------------------------- crocini
def detect_marks(gray, tpl, thr=0.6):
    """Ritorna lista di dict(x, y, score, d) con coordinate a piena risoluzione."""
    H, W = gray.shape
    k = min(1.0, 2600.0 / max(H, W))
    sm = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
    hs, ws = sm.shape
    dmin, dmax = 0.011 * W * k, 0.028 * W * k
    ds = sorted({max(10, int(round(d))) for d in np.geomspace(dmin, dmax, 9)})
    best = np.full((hs, ws), -1, np.float32)
    bestd = np.zeros((hs, ws), np.int16)
    for d in ds:
        tt = cv2.resize(tpl, (d, d), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(sm, tt, cv2.TM_CCOEFF_NORMED)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        h_, w_ = r.shape
        o = d // 2
        sub = best[o:o + h_, o:o + w_]
        bd = bestd[o:o + h_, o:o + w_]
        m = r > sub
        sub[m] = r[m]
        bd[m] = d
    rad = max(8, int(ds[0]))
    mx = cv2.dilate(best, np.ones((2 * rad + 1, 2 * rad + 1), np.uint8))
    ys, xs = np.where((best == mx) & (best >= thr))
    order = np.argsort(-best[ys, xs])[:800]
    marks = []
    for i in order:
        cx, cy = int(xs[i]), int(ys[i])
        d = int(bestd[cy, cx])
        marks.append(dict(x=cx / k, y=cy / k, score=float(best[cy, cx]), d=d / k))
    # raffinamento a piena risoluzione
    for m in marks:
        d = int(round(m["d"]))
        tt = cv2.resize(tpl, (d, d), interpolation=cv2.INTER_CUBIC)
        hw = d // 2 + 10
        x0, y0 = int(round(m["x"])) - hw, int(round(m["y"])) - hw
        x1, y1 = x0 + 2 * hw + 1, y0 + 2 * hw + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        r = cv2.matchTemplate(gray[y0:y1, x0:x1], tt, cv2.TM_CCOEFF_NORMED)
        r = np.nan_to_num(r)
        _, mv, _, ml = cv2.minMaxLoc(r)
        # sub-pixel (parabola)
        fx, fy = float(ml[0]), float(ml[1])
        if 0 < ml[0] < r.shape[1] - 1:
            a, b, c = r[ml[1], ml[0] - 1], r[ml[1], ml[0]], r[ml[1], ml[0] + 1]
            den = a - 2 * b + c
            if den != 0:
                fx += 0.5 * (a - c) / den
        if 0 < ml[1] < r.shape[0] - 1:
            a, b, c = r[ml[1] - 1, ml[0]], r[ml[1], ml[0]], r[ml[1] + 1, ml[0]]
            den = a - 2 * b + c
            if den != 0:
                fy += 0.5 * (a - c) / den
        m["x"] = x0 + fx + d / 2.0 - 0.5
        m["y"] = y0 + fy + d / 2.0 - 0.5
        m["score"] = max(m["score"], float(mv))
    return marks


def estimate_D(marks, W):
    dists = []
    for i, a in enumerate(marks):
        for b in marks[i + 1:]:
            if abs(a["x"] - b["x"]) < 0.02 * W:
                dy = abs(a["y"] - b["y"])
                if 0.1 * W < dy < 0.3 * W:
                    dists.append(dy)
    if len(dists) < 1:
        return None
    dists = np.array(dists)
    lo, hi = dists.min(), dists.max()
    bins = np.arange(lo - 6, hi + 12, 6)
    h, edges = np.histogram(dists, bins=bins)
    hs = np.convolve(h, np.ones(5), "same")
    peaks = [i for i in range(len(hs))
             if hs[i] > 0 and hs[i] >= hs[max(0, i - 1)] and hs[i] >= hs[min(len(hs) - 1, i + 1)]
             and hs[i] >= 0.5 * hs.max()]
    i = peaks[0]
    c = 0.5 * (edges[i] + edges[i + 1])
    sel = dists[np.abs(dists - c) < 20]
    return float(np.median(sel)) if len(sel) else float(c)


def pair_marks(marks, D, tol=0.04):
    cands = []
    for i, a in enumerate(marks):
        for j, b in enumerate(marks):
            if b["y"] <= a["y"]:
                continue
            vx, vy = b["x"] - a["x"], b["y"] - a["y"]
            dist = math.hypot(vx, vy)
            if abs(dist - D) < tol * D and abs(vx) < 0.18 * D:
                cost = -(a["score"] + b["score"]) + 5 * abs(dist - D) / D
                cands.append((cost, i, j))
    cands.sort()
    used, pairs = set(), []
    for _, i, j in cands:
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append(dict(top=(marks[i]["x"], marks[i]["y"]),
                          bot=(marks[j]["x"], marks[j]["y"]),
                          score=min(marks[i]["score"], marks[j]["score"])))
    return pairs


# ------------------------------------------------------------ colonne / run
def group_runs(pairs, D):
    """Assegna a ogni coppia: col, run, idx (posizione nella sequenza contigua)."""
    if not pairs:
        return
    ps = sorted(pairs, key=lambda p: p["top"][0])
    cols, cur = [], [ps[0]]
    for p in ps[1:]:
        if p["top"][0] - cur[-1]["top"][0] > 0.5 * D and p["top"][0] - np.mean([q["top"][0] for q in cur]) > 0.5 * D:
            cols.append(cur)
            cur = [p]
        else:
            cur.append(p)
    cols.append(cur)
    run_id = 0
    for ci, col in enumerate(cols):
        col.sort(key=lambda p: p["top"][1])
        diffs = [col[i + 1]["top"][1] - col[i]["top"][1] for i in range(len(col) - 1)]
        near = [d for d in diffs if 1.15 * D < d < 1.6 * D]
        P = float(np.median(near)) if near else None
        idx = 0
        for i, p in enumerate(col):
            p["col"] = ci
            if i == 0:
                run_id += 1
                idx = 0
            else:
                step = None
                if P:
                    ratio = diffs[i - 1] / P
                    if abs(ratio - round(ratio)) < 0.12 and 1 <= round(ratio) <= 3:
                        step = int(round(ratio))
                if step is None:
                    run_id += 1
                    idx = 0
                else:
                    idx += step
            p["run"] = run_id
            p["idx"] = idx


# ------------------------------------------------------------------ crop
def affine_for_pair(pair, D, geom, scale_mult=1.0):
    """Matrice dst->src (per warpAffine con WARP_INVERSE_MAP) e dimensione output.
    Parte dai crocini; se c'e' pair['corr'] (rifinitura sul rettangolo) la applica."""
    tx, ty = pair["top"]
    bx, by = pair["bot"]
    d = np.array([bx - tx, by - ty]) / math.hypot(bx - tx, by - ty)  # "giu'"
    r = np.array([d[1], -d[0]])                                     # "destra"
    s = geom["ref_d"] / D
    cx, tm = geom["half_w"], geom["top"]
    ox = tx - r[0] * cx / s - d[0] * tm / s
    oy = ty - r[1] * cx / s - d[1] * tm / s
    A3 = np.array([[r[0] / s, d[0] / s, ox], [r[1] / s, d[1] / s, oy], [0, 0, 1]], np.float64)
    if pair.get("corr") is not None:
        A3 = A3 @ pair["corr"]
    A = A3[:2].copy()
    A[:, :2] /= scale_mult
    w = int(round(2 * geom["half_w"] * scale_mult))
    h = int(round((geom["top"] + geom["ref_d"] + geom["bottom"]) * scale_mult))
    return A, (w, h)


def crop_frame(img, pair, D, geom, scale_mult=1.0, interp=cv2.INTER_LANCZOS4):
    """interp: Lanczos per il file finale (massima nitidezza), LINEAR per anteprime."""
    A, size = affine_for_pair(pair, D, geom, scale_mult)
    return cv2.warpAffine(img, A, size, flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def refine_rect(img, pair, D, geom, q=0.25, max_angle=2.5, search=0.09):
    """Rifinisce posizione e angolo usando il rettangolo stampato (foto).
    Prende il ritaglio fatto con i crocini, cerca la rotazione/traslazione che
    fa coincidere la zona 'scura' con un rettangolo ideale. Imposta pair['corr']
    e pair['rect_score']. Ritorna lo score (0..1)."""
    pair["corr"] = None
    small = crop_frame(img, pair, D, geom, q, cv2.INTER_LINEAR)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    dens = cv2.blur((g < 140).astype(np.float32), (7, 7))
    Hs, Ws = dens.shape
    rx = (geom["half_w"] + RECT["dx"] - RECT["w"] / 2) * q
    ry = (geom["top"] + RECT["dy"]) * q
    rw, rh = int(round(RECT["w"] * q)), int(round(RECT["h"] * q))
    mg = 14
    tmpl = np.zeros((rh + 2 * mg, rw + 2 * mg), np.float32)
    tmpl[mg:mg + rh, mg:mg + rw] = 1.0
    pad = int(search * D * q) + 2
    c = (Ws / 2.0, Hs / 2.0)
    best = (-1, 0, 0, 0)
    def evaluate(angle):
        M = cv2.getRotationMatrix2D(c, angle, 1.0)
        rot = cv2.warpAffine(dens, M, (Ws, Hs), flags=cv2.INTER_LINEAR, borderValue=0)
        big = cv2.copyMakeBorder(rot, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        res = np.nan_to_num(cv2.matchTemplate(big, tmpl, cv2.TM_CCOEFF_NORMED))
        # solo spostamenti entro +-pad dalla posizione ideale
        x0, y0 = int(round(rx)) - mg, int(round(ry)) - mg
        win = res[max(0, y0): y0 + 2 * pad + 1, max(0, x0): x0 + 2 * pad + 1]
        if win.size == 0:
            return None
        _, mv, _, ml = cv2.minMaxLoc(win)
        return mv, ml[0] + max(0, x0) - pad - x0, ml[1] + max(0, y0) - pad - y0
    for a in np.arange(-max_angle, max_angle + 1e-6, 0.5):
        r_ = evaluate(a)
        if r_ and r_[0] > best[0]:
            best = (r_[0], a, r_[1], r_[2])
    for a in np.arange(best[1] - 0.4, best[1] + 0.41, 0.1):
        r_ = evaluate(a)
        if r_ and r_[0] > best[0]:
            best = (r_[0], a, r_[1], r_[2])
    score, ang, ox, oy = best
    pair["rect_score"] = float(score)
    if score < 0.45:
        return float(score)
    # ox, oy: dove sta il rettangolo (piccolo, ruotato) rispetto alla posizione ideale -> t = -(ox,oy)
    M = np.vstack([cv2.getRotationMatrix2D(c, ang, 1.0), [0, 0, 1]])
    Tneg = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)  # p + (ox,oy) = posizione osservata
    S = np.diag([q, q, 1.0])
    pair["corr"] = np.linalg.inv(S) @ np.linalg.inv(M) @ Tneg @ S
    return float(score)


# -------------------------------------------------------------------- OCR
_tess = {"ok": None}


def setup_tesseract(path=None):
    try:
        import pytesseract
    except ImportError:
        _tess["ok"] = False
        return False
    cands = [path] if path else []
    cands += [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for c in cands:
        if c and os.path.isfile(c):
            pytesseract.pytesseract.tesseract_cmd = c
            break
    try:
        pytesseract.get_tesseract_version()
        _tess["ok"] = True
    except Exception:
        _tess["ok"] = False
    return _tess["ok"]


def _variants(roi_gray):
    out = []
    g = roi_gray
    _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(o)
    # toglie il retino/testo sottile: la cifra e' in grassetto
    inv = 255 - o
    op = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out.append(255 - op)
    gb = cv2.GaussianBlur(g, (0, 0), 1.5)
    _, o2 = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(o2)
    return out


def ocr_number(gray, bot, D):
    """Legge il numero sotto il crocino basso. Ritorna (testo|None, confidenza, roi)."""
    x, y = bot
    H, W = gray.shape
    x0, x1 = int(x - OCR_ROI["x"] * D), int(x + OCR_ROI["x"] * D)
    y0, y1 = int(y + OCR_ROI["y0"] * D), int(y + OCR_ROI["y1"] * D)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0, None
    roi = gray[y0:y1, x0:x1]
    if not _tess["ok"]:
        return None, 0.0, roi
    import pytesseract
    # porta l'altezza delle cifre a ~45 px
    f = 45.0 / max(1.0, 0.034 * D)
    votes = {}
    for v in _variants(roi):
        v = cv2.resize(v, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        v = cv2.copyMakeBorder(v, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        try:
            d = pytesseract.image_to_data(
                v, config="--psm 7 -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        txt, confs = "", []
        for t, c in zip(d["text"], d["conf"]):
            t = t.strip()
            if t and float(c) >= 0:
                txt += t
                confs.append(float(c))
        txt = re.sub(r"\D", "", txt)
        if 1 <= len(txt) <= 4 and confs:
            votes[txt] = votes.get(txt, 0.0) + float(np.mean(confs))
    if not votes:
        return None, 0.0, roi
    best = max(votes, key=votes.get)
    return best, votes[best], roi


# ---------------------------------------------------- numerazione / sequenza
def infer_numbers(frames, direction=0):
    """frames: lista di dict con run, idx, ocr (int|None), conf.
    direction: -1 numeri decrescenti scendendo, +1 crescenti, 0 auto.
    Imposta f['num'], f['status'], f['base']. Ritorna la direzione usata."""
    runs = {}
    for f in frames:
        runs.setdefault(f["run"], []).append(f)

    def votes_for(sign):
        res = {}
        for rid, fs in runs.items():
            c = {}
            for f in fs:
                if f.get("ocr") is not None and f.get("conf", 0) >= 40:
                    b = f["ocr"] - sign * f["idx"]
                    c[b] = c.get(b, 0) + 1
            res[rid] = c
        return res

    if direction == 0:
        tot = {s: sum(max(c.values()) for c in votes_for(s).values() if c and max(c.values()) >= 2)
               for s in (-1, 1)}
        direction = -1 if tot[-1] >= tot[1] else 1
    vts = votes_for(direction)
    for rid, fs in runs.items():
        c = vts[rid]
        base, sup = (None, 0)
        if c:
            base = max(c, key=c.get)
            sup = c[base]
        for f in fs:
            if f.get("locked"):
                continue
            f["base"] = base if (sup >= 2 or (len(fs) == 1 and sup == 1)) else None
            f["dir"] = direction
            if f["base"] is not None:
                f["num"] = f["base"] + direction * f["idx"]
                if f.get("ocr") is None:
                    f["status"] = "dedotto"
                elif f["ocr"] != f["num"]:
                    f["status"] = "OCR diverso: verifica"
                else:
                    f["status"] = "ok"
            else:
                if f.get("ocr") is not None and f.get("conf", 0) >= 120:
                    f["num"], f["status"] = f["ocr"], "verifica"
                else:
                    f["num"] = None
                    f["status"] = "numero mancante" if f.get("ocr") is None else "OCR incerto: inserisci"
    return direction


# ------------------------------------------------------------ analisi scan
def disc_darkness(gray, m, d):
    R = max(3, int(0.22 * d))
    x, y = int(round(m["x"])), int(round(m["y"]))
    H, W = gray.shape
    if x - R < 0 or y - R < 0 or x + R > W or y + R > H:
        return 255.0
    w = gray[y - R:y + R, x - R:x + R]
    yy, xx = np.mgrid[-R:R, -R:R]
    mask = np.hypot(xx + .5, yy + .5) < 0.22 * d
    return float(w[mask].mean())


def refine_mark(gray, x, y, D, d, tpl):
    """Cerca il centro esatto del crocino vicino a (x, y) (usato per i clic manuali)."""
    H, W = gray.shape
    d = int(round(d))
    tt = cv2.resize(tpl, (d, d), interpolation=cv2.INTER_CUBIC)
    hw = int(0.08 * D) + d // 2
    x0, y0 = max(0, int(x) - hw), max(0, int(y) - hw)
    x1, y1 = min(W, int(x) + hw), min(H, int(y) + hw)
    if x1 - x0 <= d or y1 - y0 <= d:
        return x, y
    r = np.nan_to_num(cv2.matchTemplate(gray[y0:y1, x0:x1], tt, cv2.TM_CCOEFF_NORMED))
    _, mv, _, ml = cv2.minMaxLoc(r)
    if mv < 0.3:
        return x, y
    return x0 + ml[0] + d / 2.0 - 0.5, y0 + ml[1] + d / 2.0 - 0.5


def make_frame(img, gray, pair, D, geom, path, run, idx, col, thumb_w=360):
    refine_rect(img, pair, D, geom)
    txt, conf, roi = ocr_number(gray, pair["bot"], D)
    th = crop_frame(img, pair, D, geom, thumb_w / (2 * geom["half_w"]), cv2.INTER_LINEAR)
    return dict(scan=str(path), pair=pair, D=D, run=run, idx=idx, col=col,
                ocr=int(txt) if txt else None, conf=conf, thumb=th, roi=roi,
                include=True, rect_score=pair.get("rect_score", 0.0))


def analyze_scan(path, tpl, thr=0.5, manual_D=0.0, geom=None, log=print):
    geom = geom or DEFAULT_GEOM
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossibile leggere {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    marks = detect_marks(gray, tpl, thr)
    if not marks:
        raise RuntimeError("Nessun crocino trovato")
    dm = float(np.median([m["d"] for m in marks[:40]]))
    marks = [m for m in marks if disc_darkness(gray, m, dm) < 170]
    strong = [m for m in marks if m["score"] >= 0.6] or marks
    D = manual_D if manual_D > 0 else estimate_D(strong, W)
    if not D:
        raise RuntimeError("Distanza tra i crocini non stimabile: imposta D a mano")
    pairs = pair_marks(marks, D)
    if len(pairs) >= 4:
        # le coppie vere hanno tutte la stessa inclinazione: scarta quelle "storte" (falsi crocini nel testo)
        ang = np.array([math.atan2(p["bot"][0] - p["top"][0], p["bot"][1] - p["top"][1]) for p in pairs])
        med = float(np.median(ang))
        keep = [p for p, a in zip(pairs, ang) if abs(a - med) < math.radians(3)]
        if len(keep) < len(pairs):
            log(f"{Path(path).name}: scartate {len(pairs) - len(keep)} coppie di crocini con inclinazione anomala")
        pairs = keep
    group_runs(pairs, D)
    frames = []
    for p in sorted(pairs, key=lambda q: (q["col"], q["top"][1])):
        frames.append(make_frame(img, gray, p, D, geom, path, (str(path), p["run"]), p["idx"], p["col"]))
    log(f"{Path(path).name}: D={D:.0f}px, {len(frames)} frame trovati")
    return frames, dict(D=D, d=dm)


# ------------------------------------------------- overlay per l'editor
def footprint_poly(pair, D, geom, k):
    """Angoli del ritaglio (in coordinate del display = scansione * k)."""
    A, (w, h) = affine_for_pair(pair, D, geom)
    pts = []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        pts.append(((A[0, 0] * u + A[0, 1] * v + A[0, 2]) * k,
                    (A[1, 0] * u + A[1, 1] * v + A[1, 2]) * k))
    return pts


def seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def compose_overlay(base_rgba, shapes):
    """shapes: lista di dict(poly, top, bot, color=(r,g,b)). Disegna i riquadri
    semitrasparenti del taglio + la linea tra i crocini sopra la pagina."""
    from PIL import Image, ImageDraw
    ov = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for s in shapes:
        r, g, b = s["color"]
        d.polygon([tuple(p) for p in s["poly"]], fill=(r, g, b, 62), outline=(r, g, b, 255))
        d.line([tuple(s["top"]), tuple(s["bot"])], fill=(11, 91, 67, 255), width=3)
        for (x, y) in (s["top"], s["bot"]):
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(11, 91, 67, 255), width=2)
    return Image.alpha_composite(base_rgba, ov)


# =========================================================== INTERFACCIA
# Stile "Teenage Engineering": corpo crema, tasti rialzati, LCD verde 8-bit.
import threading, queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json, time, traceback, datetime, hashlib, zipfile
from types import SimpleNamespace

# --- cartella dati dell'app: registro errori/tempi e salvataggio automatico
APP_DIR = Path.home() / ".scontrini_pdf"
LOG_FILE = APP_DIR / "log.txt"
AUTOSAVE_FILE = APP_DIR / "autosave.json"
PROJECT_VERSION = 1
PREPROC_VERSION = 1          # da incrementare se cambia il pre-trattamento (es. rotazione) che definisce le coordinate salvate
# variabili dell'interfaccia che finiscono nel progetto
PROJECT_VARS = ("v_tpl", "v_out", "v_tess", "v_thr", "v_D", "v_dir", "v_fmt", "v_pad", "v_hw", "v_top",
                "v_bot", "v_ref", "v_omode", "v_orect", "v_opdf", "v_ochk", "v_oover", "v_ofit",
                "v_ow", "v_oh", "v_pdf", "v_pdf_rotate", "v_opaper")


PREFS_FILE = APP_DIR / "prefs.json"


def load_prefs():
    try:
        d = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_prefs(d):
    try:
        APP_DIR.mkdir(exist_ok=True)
        tmp = PREFS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(PREFS_FILE)
    except OSError as ex:
        file_log(f"preferenze non salvate: {ex}")


def file_log(msg):
    """Registro su file (errori, tempi). Non deve mai far fallire l'app."""
    try:
        APP_DIR.mkdir(exist_ok=True)
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 1_000_000:
            LOG_FILE.replace(LOG_FILE.with_suffix(".old"))
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S  ") + msg + "\n")
    except OSError:
        pass


CACHE_DIR = APP_DIR / "cache"
CACHE_VERSION = 1            # da incrementare se cambia il contenuto salvato nella cache dei riferimenti PDF
CACHE_KEEP = 2               # quante cache PDF tenere su disco (le piu' recenti)


def file_fingerprint(path):
    """Impronta leggera di un file ([dimensione, data di modifica]): serve a capire se e' cambiato."""
    try:
        st = Path(path).stat()
        return [int(st.st_size), int(st.st_mtime_ns)]
    except (OSError, TypeError):
        return None


def engine_geom_key(geom):
    return [round(float(geom[k]), 3) for k in ("ref_d", "half_w", "top", "bottom")]


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    return v


def _match_from_json(m):
    out = dict(m)
    for k in ("pdf_H", "pdf_sim"):
        if k in out:
            out[k] = np.array(out[k], np.float64)
    if "pdf_size" in out:
        out["pdf_size"] = tuple(int(v) for v in out["pdf_size"])
    return out


def _saved_match_usable(d, geom):
    """L'aggancio al PDF salvato nel progetto vale solo se PDF, scansione e geometria sono quelli di allora."""
    eng = _PDF_ENGINE
    m = d.get("match")
    if not m or eng is None or "pdf_num" not in m or "pdf_H" not in m:
        return False
    fp = getattr(eng, "fingerprint", None)
    # le coordinate salvate valgono solo nello stesso orientamento e con lo stesso pre-trattamento
    # (se la chiave manca, e' un progetto vecchio: nessun controllo, come prima)
    cur_rot = str(Path(d.get("scan", "")).resolve()) in _ROTATE_SCANS
    if ("rot" in d and bool(d["rot"]) != cur_rot) or d.get("prep", PREPROC_VERSION) != PREPROC_VERSION:
        return False
    return (fp is not None and d.get("pdf_fp") == list(fp)
            and d.get("scan_fp") is not None and d["scan_fp"] == file_fingerprint(d.get("scan"))
            and d.get("geom_key") == engine_geom_key(geom))


def make_frame_with_match(img, pair, D, geom, path, run, idx, col, match, thumb_w=360):
    """Come make_frame() quando il PDF e' gia' stato riconosciuto: salta la ricerca sulle 364 pagine."""
    D = math.hypot(pair["bot"][0] - pair["top"][0], pair["bot"][1] - pair["top"][1])
    f = dict(scan=str(path), pair=pair, D=D, run=run, idx=idx, col=col, ocr=None, conf=0., roi=None,
             thumb=None, include=True, rect_score=0., num=None, locked=True, status="")
    _apply_match(f, img, geom, match, thumb_w)
    return f


class ScanCache:
    """Cache LRU minima delle scansioni lette: evita riletture ripetute senza riempire la memoria."""

    def __init__(self, size=2):
        self.size, self._d, self._lock = size, {}, threading.Lock()

    def get(self, path):
        # la chiave comprende l'orientamento: la stessa scansione ruotata e non ruotata sono due immagini diverse
        key = (path, str(Path(path).resolve()) in _ROTATE_SCANS)
        with self._lock:
            if key in self._d:
                self._d[key] = self._d.pop(key)
                return self._d[key]
        img = imread_unicode(path)          # (versione con rotazione, definita piu' sotto)
        if img is not None:
            with self._lock:
                self._d[key] = img
                while len(self._d) > self.size:
                    self._d.pop(next(iter(self._d)))
        return img

    def clear(self):
        with self._lock:
            self._d.clear()


SORT_TITLES = {"scan": "SCANSIONE", "pos": "COL/POS", "rett": "RETTANGOLO", "ocr": "OCR",
               "num": "N°", "stato": "STATO"}


def frame_sort_value(f, col):
    """Valore di ordinamento di una colonna. None = cella vuota (va sempre in fondo)."""
    top_y = f["pair"]["top"][1]
    c = f["col"] if f["col"] >= 0 else 99
    if col == "scan":
        return (Path(f["scan"]).name.lower(), c, top_y)
    if col == "pos":
        return (c, top_y)
    if col == "rett":
        return float(f["rect_score"])
    if col in ("ocr", "num"):
        v = f["ocr"] if col == "ocr" else f.get("num")
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    if col == "stato":
        return str(f["status"]).lower()
    return None


def sorted_order(frames, col, reverse):
    """Indici dei frame nell'ordine di visualizzazione. L'elenco dei frame NON viene toccato
    (infer_numbers e salvataggio dipendono dall'ordine naturale): cambia solo la vista."""
    idx = list(range(len(frames)))
    if not col:
        return idx
    vals = {i: frame_sort_value(frames[i], col) for i in idx}
    have = [i for i in idx if vals[i] is not None]
    empty = [i for i in idx if vals[i] is None]
    have.sort(key=lambda i: vals[i], reverse=reverse)
    return have + empty


def gaps_text(frames):
    """Riepilogo informativo dei buchi fra il minimo e il massimo dei frame inclusi. Si lavora per porzioni del
    PDF: i buchi possono essere voluti e NON bloccano mai salvataggio o esportazione."""
    inc = [f for f in frames if f["include"]]
    nums = sorted({int(f["num"]) for f in inc if f.get("num") is not None})
    no_num = sum(1 for f in inc if f.get("num") is None)
    parts = []
    if no_num:
        parts.append(f"SENZA NUMERO: {no_num}")
    if len(nums) >= 2:
        holes, total = [], 0
        for a, b in zip(nums, nums[1:]):
            if b - a > 1:
                lo, hi = a + 1, b - 1
                holes.append(str(lo) if lo == hi else f"{lo}-{hi}")
                total += hi - lo + 1
        if holes:
            txt = ", ".join(holes)
            if len(txt) > 90:
                txt = txt[:90].rsplit(",", 1)[0] + ", ..."
            parts.append(f"buchi fra {nums[0]} e {nums[-1]} (solo informativo, {total} numeri non caricati): {txt}")
        else:
            parts.append(f"nessun buco fra {nums[0]} e {nums[-1]}")
    return "  |  ".join(parts)


def next_missing(frames, order, current):
    """iid del prossimo frame (nell'ordine MOSTRATO in tabella) incluso e senza numero.
    Cerca dopo `current`; se in fondo non ce ne sono riparte dall'inizio. None = non ne restano."""
    if current in order:
        k = order.index(current)
        seq = order[k + 1:] + order[:k]
    else:
        seq = list(order)
    for iid in seq:
        f = frames[int(iid)]
        if f["include"] and f.get("num") is None:
            return iid
    return None


def wheel_notches(delta):
    """Tacche della rotellina nella convenzione di yview_scroll: negativo = su, positivo = giu'."""
    if not delta:
        return 0
    n = max(1, abs(int(delta)) // 120)
    return -n if delta > 0 else n


def _can_scroll_y(w):
    try:
        if isinstance(w, tk.Listbox):
            return w.size() > int(w.cget("height"))
        a, b = w.yview()
        return a > 0.0 or b < 1.0
    except (tk.TclError, ValueError, TypeError):
        return False


def scroll_target(w, top, viewport):
    """Chi scorre con la rotellina, in base a cio' che sta SOTTO il puntatore:
    elenco frame, console o lista scansioni (se hanno qualcosa da scorrere); altrimenti il pannello principale.
    Nelle altre finestre (menu a tendina, ecc.) scorre solo il widget scorrevole sotto il puntatore."""
    x = w
    while x is not None:
        if isinstance(x, (ttk.Treeview, tk.Listbox, tk.Text)) and _can_scroll_y(x):
            return x
        x = getattr(x, "master", None)
    if w is not None and w.winfo_toplevel() is top:
        return viewport
    return None


def install_wheel(root, viewport, app):
    """Gestione della rotellina per contesto. Su Windows l'evento arriva al widget con il FOCUS, non a quello
    sotto il puntatore: per questo si cerca il widget sotto il puntatore e si instrada l'evento di conseguenza."""
    def on_wheel(ev):
        n = wheel_notches(getattr(ev, "delta", 0))
        if not n:
            return None
        try:
            w = root.winfo_containing(ev.x_root, ev.y_root)
        except (KeyError, tk.TclError):
            w = None
        if w is None:
            w = ev.widget
        ed = getattr(app, "_editor", None)
        try:
            if ed is not None and ed.win.winfo_exists() and w is ed.cv:
                if ev.widget is not ed.cv:      # evento arrivato al widget col focus: lo giro all'editor
                    ed._wheel(SimpleNamespace(delta=ev.delta, state=ev.state,
                                              x=ev.x_root - ed.cv.winfo_rootx(),
                                              y=ev.y_root - ed.cv.winfo_rooty()))
                return "break"                   # editor: rotella = scorri, Ctrl+rotella = zoom
        except tk.TclError:
            pass
        target = scroll_target(w, root, viewport)
        if target is None:
            return None
        if target is viewport:
            if ev.state & 0x1:                   # Maiusc + rotella = scorrimento laterale
                viewport.xview_scroll(n, "units")
            else:
                viewport.yview_scroll(n, "units")
        else:
            target.yview_scroll(n * 3, "units")
        return "break"
    root.bind_all("<MouseWheel>", on_wheel)
    # widget che di default reagiscono da soli alla rotellina (anche quando hanno il focus):
    # li instradiamo qui, cosi' non scorrono in doppio e Combobox/Spinbox non cambiano valore per sbaglio
    for cls in ("Treeview", "Listbox", "Text", "TCombobox", "Spinbox", "TScrollbar"):
        root.bind_class(cls, "<MouseWheel>", on_wheel)


def merge_decisions(current, extra):
    """Decisioni dei frame attuali + quelle conservate che non coincidono con nessuna di quelle gia' raccolte.
    Il confronto e' con l'insieme che si sta costruendo (non solo con `current`): cosi' i duplicati interni a
    `extra` si fondono e ripetere il merge non fa crescere l'elenco. A parita' di coppia vince la prima."""
    out = list(current)
    for d in extra:
        if not any(c["scan"] == d["scan"] and _near({"top": c["top"], "bot": c["bot"]}, d["top"], d["bot"])
                   for c in out):
            out.append(d)
    return out


def _frame_sig(f):
    """Firma di cio' che determina l'aspetto di un frame: pagina PDF, numero E trasformazione.
    Serve a sapere se l'anteprima va rifatta anche quando cambia solo la matrice, a numero invariato."""
    p = f["pair"]
    H = p.get("pdf_H")
    h = None if H is None else tuple(round(float(v), 3) for v in np.asarray(H, np.float64).ravel())
    return (p.get("pdf_num"), f.get("num"), h)


def _near(pair, top, bot, tol=12.0):
    return (math.hypot(pair["top"][0] - top[0], pair["top"][1] - top[1]) < tol and
            math.hypot(pair["bot"][0] - bot[0], pair["bot"][1] - bot[1]) < tol)


def image_size(path):
    """(larghezza, altezza) leggendo solo l'intestazione del file; None se non leggibile."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def adapt_rotation(items, rot_now, size_of, log=None):
    """Decisioni / cancellazioni salvate con le coordinate in un orientamento diverso da quello attuale della
    scansione: si trasformano con la stessa rotazione di 180 gradi (x->W-1-x, y->H-1-y; il crocino alto e quello
    basso si scambiano). L'aggancio al PDF salvato non e' piu' valido e viene tolto. Voci senza 'rot' (progetti
    vecchi) o gia' coerenti restano com'erano."""
    out, n = [], 0
    for d in items:
        cur = bool(rot_now(d["scan"]))
        if "rot" not in d or bool(d["rot"]) == cur:
            out.append(d)
            continue
        size = size_of(d["scan"])
        if size is None:
            out.append(d)                    # scansione non leggibile: non c'e' nulla a cui applicarla comunque
            continue
        W, H = size
        top, bot = d["top"], d["bot"]
        d2 = dict(d, top=[W - 1 - bot[0], H - 1 - bot[1]], bot=[W - 1 - top[0], H - 1 - top[1]], rot=cur)
        for k in ("match", "pdf_fp", "scan_fp", "geom_key"):
            d2.pop(k, None)
        out.append(d2)
        n += 1
    if n and log:
        log(f"Orientamento delle scansioni cambiato: trasformate {n} correzioni/cancellazioni salvate (rotazione 180°).")
    return out


def apply_tombstones(frames, pending, tombs, tol=12.0):
    """Toglie i frame cancellati con la gomma (e le decisioni pendenti su di essi). Non tocca i frame aggiunti a
    mano ne' le decisioni 'added'. Ritorna (frame rimasti, pendenti rimaste, numero rimossi)."""
    def hit(scan, pair):
        return any(t["scan"] == scan and _near(pair, t["top"], t["bot"], tol) for t in tombs)
    kept = [f for f in frames if f.get("manual") or not hit(f["scan"], f["pair"])]
    pend = [d for d in pending if d.get("added") or not hit(d["scan"], {"top": d["top"], "bot": d["bot"]})]
    return kept, pend, len(frames) - len(kept)


def output_order(todo):
    """Ordine di elaborazione/scrittura: per scansione, poi l'ordine dei frame. Un solo ordine, usato sia per
    assegnare i nomi sia per scrivere i file."""
    return [f for path in sorted({f["scan"] for f in todo}) for f in todo if f["scan"] == path]


def plan_output_names(ordered, pad, ext, out, check, policy, exists=None):
    """Nome di ogni file di uscita. `policy` per i file GIA' presenti in cartella:
    'rinomina' (aggiunge _2, _3...), 'salta' (non li scrive), 'sovrascrivi' (li sostituisce).
    I nomi doppi dentro il lotto ricevono sempre un suffisso. Ritorna una lista di dict
    (f, name, orig, action, existed): `existed` = il nome previsto esisteva gia' (anche solo in _controllo)."""
    exists = exists or (lambda p: Path(p).exists())
    out = Path(out)

    def taken(n):
        return exists(out / (n + ext)) or (check and exists(out / "_controllo" / (n + ".png")))
    assigned, seen, unk, plan = set(), {}, 0, []
    for f in ordered:
        if f.get("num") is None:
            unk += 1
            base = f"senza_numero_{unk:02d}"
        else:
            base = str(f["num"]).zfill(pad) if pad else str(f["num"])
        k = seen.get(base, 0) + 1                     # doppioni nel lotto: conteggio sul nome di partenza
        seen[base] = k
        name = base if k == 1 else f"{base}_{k}"
        orig = name
        existed = taken(name)
        action = "scrivi"
        if existed and policy == "salta":
            action = "salta"
        elif existed and policy == "rinomina":
            while name in assigned or taken(name):
                k += 1
                name = f"{base}_{k}"
        while name in assigned:                       # mai due frame con lo stesso nome finale
            k += 1
            name = f"{base}_{k}"
        assigned.add(name)
        plan.append(dict(f=f, name=name, orig=orig, action=action, existed=existed))
    return plan


def summarize_save(r):
    """Esito dell'export -> (titolo LCD, riga LCD, livello 'info'|'warning'|'error'|'log', testo completo)."""
    st, w, p, s = r["status"], r["written"], r["planned"], r["skipped"]
    fl, ne, cf = r["failed"], r["not_exported"], r["check_failed"]
    lines = [f"Scritti {w} file su {p} previsti in: {r['out']}"]
    if s:
        lines.append(f"Saltati perche' gia' presenti: {s}")
    if fl:
        lines.append(f"ERRORI su {len(fl)} file: " + "; ".join(f"{n} ({m})" for n, m in fl[:5])
                     + (" ..." if len(fl) > 5 else ""))
    if r.get("fatal"):
        lines.append(f"Errore che ha interrotto il salvataggio: {r['fatal']}")
    if ne and st != "completed":
        lines.append(f"NON esportati: {len(ne)} ({', '.join(ne[:10])}{', ...' if len(ne) > 10 else ''})")
    if cf:
        lines.append(f"Controllo sovrapposizione non riuscito per {len(cf)} file "
                     "(i ritagli principali sono stati scritti)")
    title = {"completed": "SALVATO", "cancelled": "ANNULLATO", "partial": "PARZIALE", "failed": "ERRORE"}[st]
    level = {"completed": "info", "cancelled": "log", "partial": "warning", "failed": "error"}[st]
    sub = f"{w}/{p} file in {r['out']}"
    return title, sub, level, "\n".join(lines)


def restore_added_frames(decisions, existing, tpl, geom, cache, log):
    """Ricrea i frame aggiunti a mano nell'editor che l'analisi automatica non ritrova."""
    out, reused = [], 0
    for n, d in enumerate(decisions):
        if not d.get("added"):
            continue
        top, bot = tuple(d["top"]), tuple(d["bot"])
        if any(f["scan"] == d["scan"] and _near(f["pair"], top, bot) for f in list(existing) + out):
            continue
        img = cache.get(d["scan"])
        if img is None:
            log(f"Frame manuale NON ripristinato: scansione non leggibile ({Path(d['scan']).name})")
            continue
        pair = dict(top=top, bot=bot, score=1.0)
        if d.get("anchor"):
            pair["anchor"] = d["anchor"]
        D = float(d.get("D") or math.hypot(bot[0] - top[0], bot[1] - top[1]))
        if _saved_match_usable(d, geom):          # aggancio al PDF gia' noto: niente nuova ricerca
            fr = make_frame_with_match(img, pair, D, geom, d["scan"], (d["scan"], "p", n), 0, -1,
                                       _match_from_json(d["match"]))
            reused += 1
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fr = make_frame(img, gray, pair, D, geom, d["scan"], (d["scan"], "p", n), 0, -1)
        fr["manual"] = True
        out.append(fr)
    if reused:
        log(f"{reused} frame manuali ripristinati con l'aggancio al PDF salvato nel progetto (nessuna nuova ricerca).")
    return out

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

# --- TEMI: tutti i colori dell'interfaccia vivono qui. Nessun colore va scritto nei singoli widget.
# "chiaro" = look originale. "scuro" = SOLO la palette scelta dall'utente (SPECIFICHE_REVISIONE, par. 4).
THEMES = {
    "chiaro": dict(
        BODY="#E6DFCB", PANEL="#EEE8D6", KEY="#F3EFE1", KEY_HI="#FFFFFF", EDGE="#2A2A27", SHADOW="#B9B19A",
        INK="#1F2A24", MUTED="#7B7766", ORANGE="#FF4F12", ACTIVE="#FF4F12", GREEN="#0B5B43",
        LCD_BG="#1B9C73", LCD_OFF="#178A66", LCD_DARK="#0A3A2B", LCD_LIGHT="#9BF7CF",
        NIGHT="#222222", BLUE="#5E7BD8", RED="#E8405A",
        FIELD="#FBF8EE", DISABLED="#D8D2BE", LED_OFF="#B5AE98", LED_EDGE="#2A2A27", HOVER_EDGE="#2A2A27",
        HEAD_BG="#0B5B43", HEAD_FG="#F3EFE1", SEL_FG="#FFFFFF", CANVAS_FG="#888888",
        WARN_BG="#F7D9A0", WARN_FG="#1F2A24", BAD_BG="#F4A38A", BAD_FG="#1F2A24", OFF_FG="#999999",
        ON_ORANGE="#FFFFFF", ON_GREEN="#F3EFE1", ON_BLUE="#FFFFFF", ON_RED="#FFFFFF"),
    "scuro": dict(
        # EDGE/SHADOW (#4D4D4D): antracite piu' chiaro, scelto su richiesta per far vedere contorni e ombre dei tasti
        # palette utente: #262626 sfondo, #343434 pannelli, #BF4968 rosa/active, #5166A6 blu/secondary,
        # #41BF84 verde/online, #D18A43 ambra/confirm, #F0F0F0 testo, #AAAAAA testo secondario
        BODY="#262626", PANEL="#343434", KEY="#343434", KEY_HI="#343434", EDGE="#4D4D4D", SHADOW="#4D4D4D",
        INK="#F0F0F0", MUTED="#AAAAAA", ORANGE="#D18A43", ACTIVE="#BF4968", GREEN="#41BF84",
        LCD_BG="#343434", LCD_OFF="#262626", LCD_DARK="#41BF84", LCD_LIGHT="#262626",
        NIGHT="#262626", BLUE="#5166A6", RED="#BF4968",
        FIELD="#262626", DISABLED="#262626", LED_OFF="#262626", LED_EDGE="#AAAAAA", HOVER_EDGE="#BF4968",
        HEAD_BG="#262626", HEAD_FG="#F0F0F0", SEL_FG="#F0F0F0", CANVAS_FG="#AAAAAA",
        WARN_BG="#343434", WARN_FG="#D18A43", BAD_BG="#D18A43", BAD_FG="#262626", OFF_FG="#AAAAAA",
        ON_ORANGE="#262626", ON_GREEN="#262626", ON_BLUE="#F0F0F0", ON_RED="#F0F0F0"),
}
# nel tema chiaro piu' ruoli hanno lo stesso colore: nelle mappature di widget vince il ruolo "principale"
THEME_PRIMARY = ("BODY", "PANEL", "KEY", "KEY_HI", "SHADOW", "MUTED", "ORANGE", "ACTIVE", "GREEN", "BLUE", "RED",
                 "NIGHT", "FIELD", "CANVAS_FG", "INK", "EDGE")
CURRENT_THEME = "chiaro"
FONT = "Consolas"


def key_kinds():
    return {"cream": (KEY, INK), "orange": (ORANGE, ON_ORANGE), "green": (GREEN, ON_GREEN),
            "blue": (BLUE, ON_BLUE), "red": (RED, ON_RED)}


def set_theme_globals(name):
    """Rende corrente un tema: aggiorna le costanti di modulo usate da tutti i widget."""
    if name not in THEMES:
        name = "chiaro"
    g = globals()
    g.update(THEMES[name])
    g["CURRENT_THEME"] = name
    if "Key" in g:
        Key.KINDS = key_kinds()
    if "PageEditor" in g:
        PageEditor.TOOLS = {"add": ("+ CROCINI", "orange", ORANGE, "crosshair"),
                            "rect": ("RETTANGOLO", "blue", BLUE, "tcross"),
                            "erase": ("GOMMA", "red", RED, "X_cursor")}


globals().update(THEMES["chiaro"])

RECEIPT = ["#########.",
           "#.......#.",
           "#.#####.#.",
           "#.......#.",
           "#.#####.#.",
           "#.......#.",
           "#.###...#.",
           "#.......#.",
           "#########.",
           "#.#.#.#.#."]
INV_A = ["..#.....#..",
         "...#...#...",
         "..#######..",
         ".##.###.##.",
         "###########",
         "#.#######.#",
         "#.#.....#.#",
         "...##.##..."]
INV_B = ["..#.....#..",
         "#..#...#..#",
         "#.#######.#",
         "###.###.###",
         "###########",
         ".#########.",
         "..#.....#..",
         ".#.......#."]

# --- animazioni del display: "icon" = figura grande a sinistra (si alterna mentre lavora),
#     "runner" = figurina che corre sulla barra (None = nessuna). Gli sprite sono 11 colonne al massimo.
PRINTER_A = ["..#######..", "..#.....#..", ".#########.", "#.........#", "#.....#...#",
             "###########", ".#.......#.", ".#.#####.#.", ".#.......#.", ".#########."]
PRINTER_B = ["..#######..", "..#.....#..", ".#########.", "#.........#", "#.....#...#",
             "###########", ".#.#####.#.", ".#.#...#.#.", ".#.#####.#.", ".#########."]
SCANNER_A = ["###########", "#.........#", "#.#######.#", "#.#.....#.#", "#.#.....#.#",
             "#.........#", "###########", "#.#.#.#.#.#"]
SCANNER_B = ["###########", "#.........#", "#.#.....#.#", "#.#.....#.#", "#.#######.#",
             "#.........#", "###########", "#.#.#.#.#.#"]
FILM_A = ["#.#.#.#.#.#", "###########", "#.###.###.#", "#.###.###.#", "#.###.###.#",
          "###########", "#.#.#.#.#.#"]
FILM_B = [".#.#.#.#.#.", "###########", "#.###.###.#", "#.###.###.#", "#.###.###.#",
          "###########", ".#.#.#.#.#."]
PAPER_A = [".#####.", ".#...#.", ".#.#.#.", ".#...#.", ".#.#.#.", ".#...#.", ".#####."]
PAPER_B = [".#####.", ".#...#.", ".#...#.", ".#.#.#.", ".#...#.", ".#.#.#.", ".#####."]
SPARK_A = ["..#..", "..#..", "#####", "..#..", "..#.."]
SPARK_B = ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"]
FRAME_A = ["#######", "#.....#", "#.###.#", "#.###.#", "#.###.#", "#.....#", "#######"]
FRAME_B = ["#######", "#.....#", "#.....#", "#.....#", "#.....#", "#.....#", "#######"]
DEFAULT_ANIM = "Scontrino + omino"
ANIMS = {
    DEFAULT_ANIM: dict(icon=[RECEIPT], runner=[INV_A, INV_B]),
    "Stampante": dict(icon=[PRINTER_A, PRINTER_B], runner=[PAPER_A, PAPER_B]),
    "Scanner": dict(icon=[SCANNER_A, SCANNER_B], runner=[SPARK_A, SPARK_B]),
    "Pellicola": dict(icon=[FILM_A, FILM_B], runner=[FRAME_A, FRAME_B]),
    "Essenziale": dict(icon=None, runner=None),
}


def bgr_to_tk(arr, max_w=None, max_h=None):
    im = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)) if arr.ndim == 3 else Image.fromarray(arr)
    if max_w or max_h:
        im.thumbnail((max_w or 10**6, max_h or 10**6))
    return ImageTk.PhotoImage(im)


def sprite(cv, rows, x, y, s, color, tag="sp"):
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch == "#":
                cv.create_rectangle(x + i * s, y + j * s, x + (i + 1) * s, y + (j + 1) * s,
                                    fill=color, outline="", tags=tag)


class Key(tk.Canvas):
    """Tasto rialzato in stile TE (crema / arancio / verde scuro)."""
    KINDS = {}      # riempito da key_kinds() a ogni cambio di tema

    def __init__(self, master, text, command, kind="cream", width=150, height=40, bg=None):
        super().__init__(master, width=width + 5, height=height + 5, bg=bg or BODY, highlightthickness=0, bd=0)
        self.text, self.command, self.kind = text, command, kind
        self.w, self.h = width, height
        self.enabled, self.hover, self.pressed = True, False, False
        self.bind("<Enter>", lambda e: self._set(hover=True))
        self.bind("<Leave>", lambda e: self._set(hover=False, pressed=False))
        self.bind("<ButtonPress-1>", lambda e: self._set(pressed=True))
        self.bind("<ButtonRelease-1>", self._up)
        self._draw()

    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._draw()

    def _up(self, _e):
        was = self.pressed
        self._set(pressed=False)
        if was and self.enabled and self.hover:
            self.command()

    def enable(self, flag=True):
        self.enabled = flag
        self._draw()

    def set_kind(self, kind):
        self.kind = kind
        self._draw()

    def _draw(self):
        self.delete("all")
        fill, fg = self.KINDS[self.kind]
        if not self.enabled:
            fill, fg = DISABLED, MUTED
        elif self.hover and self.kind == "cream":
            fill = KEY_HI
        edge = HOVER_EDGE if (self.hover and self.enabled) else EDGE
        off = 3 if (self.pressed and self.enabled) else 0
        if not off:
            self.create_rectangle(4, 4, self.w + 3, self.h + 3, fill=EDGE, outline="")
        self.create_rectangle(1 + off, 1 + off, self.w + off, self.h + off, fill=fill, outline=edge, width=2)
        if self.kind == "cream" and self.enabled:
            self.create_line(4 + off, 4 + off, self.w - 3 + off, 4 + off, fill=KEY_HI)
        self.create_text(self.w // 2 + off, self.h // 2 + off, text=self.text, fill=fg,
                         font=(FONT, 11, "bold"))


Key.KINDS = key_kinds()


class Led(tk.Canvas):
    def __init__(self, master, text, color=None, bg=None):
        super().__init__(master, width=70, height=16, bg=bg or BODY, highlightthickness=0)
        self.text, self.color, self.on = text, color or ACTIVE, False
        self.set(False)

    def redraw(self):
        self.set(self.on)

    def set(self, on):
        self.on = bool(on)
        self.delete("all")
        self.create_oval(2, 3, 12, 13, fill=self.color if on else LED_OFF, outline=LED_EDGE)
        self.create_text(18, 8, text=self.text, anchor="w", fill=INK, font=(FONT, 8, "bold"))


class Lcd(tk.Canvas):
    """Display LCD verde con caricamento a blocchi e sprite 8-bit animato."""
    W, H = 1080, 142
    N, X0, CW, GAP, BARY = 30, 100, 26, 4, 120

    def __init__(self, master):
        super().__init__(master, width=self.W, height=self.H, bg=EDGE, highlightthickness=0)
        self.t = 0
        self.mode, self.title, self.sub = "idle", "PRONTO", "carica le scansioni e premi ANALIZZA"
        self.done, self.total = 0, 0
        self.flags = {"OCR": False, "RECT": False}
        self.anim = DEFAULT_ANIM
        self.after(120, self._tick)

    def set(self, mode=None, title=None, sub=None, done=None, total=None):
        if mode is not None:
            self.mode = mode
        if title is not None:
            self.title = title
        if sub is not None:
            self.sub = sub
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total

    def _tick(self):
        self.t += 1
        try:
            self._draw()
        except tk.TclError:
            return
        self.after(120, self._tick)

    def _draw(self):
        self.delete("all")
        W, H, t = self.W, self.H, self.t
        busy = self.mode == "busy"
        self.create_rectangle(0, 0, W, H, fill=EDGE, outline="")
        self.create_rectangle(6, 6, W - 6, H - 6, fill=LCD_BG, outline=LCD_DARK, width=2)
        # righe di scansione LCD
        for y in range(10, H - 8, 4):
            self.create_line(8, y, W - 8, y, fill=LCD_OFF)
        # icone di stato (stile display Riddim)
        x = W - 30
        for name in ("BUSY", "RECT", "OCR"):
            on = busy if name == "BUSY" else self.flags.get(name, False)
            wd = 14 + 9 * len(name)
            x -= wd
            self.create_rectangle(x, 14, x + wd, 32, fill=LCD_DARK if on else LCD_BG,
                                  outline=LCD_DARK, width=2)
            self.create_text(x + wd // 2, 23, text=name, font=(FONT, 8, "bold"),
                             fill=LCD_LIGHT if on else LCD_DARK)
            x -= 8
        self.create_text(24, 22, text="SCONTRINI > FRAME", anchor="w", fill=LCD_DARK, font=(FONT, 10, "bold"))
        # scontrino pixel che "salta" mentre lavora
        bob = 3 if (busy and t % 4 < 2) else 0
        anim = ANIMS.get(self.anim) or ANIMS[DEFAULT_ANIM]
        if anim["icon"]:
            icon = anim["icon"][(t // 2) % len(anim["icon"])] if busy else anim["icon"][0]
            sprite(self, icon, 24, 40 - bob, 5, LCD_DARK)
        # titolo + cursore lampeggiante
        cur = "_" if (t // 3) % 2 == 0 else " "
        self.create_text(100, 56, text=self.title + cur, anchor="w", fill=LCD_DARK, font=(FONT, 26, "bold"))
        self.create_text(100, 90, text=self.sub[:90], anchor="w", fill=LCD_DARK, font=(FONT, 10))
        # percentuale grande a destra
        if self.total > 0:
            pct = int(round(100.0 * self.done / self.total))
            self.create_text(W - 34, 62, text="%03d" % pct, anchor="e", fill=LCD_DARK, font=(FONT, 36, "bold"))
            self.create_text(W - 30, 70, text="%", anchor="w", fill=LCD_DARK, font=(FONT, 12, "bold"))
        # barra a blocchi
        filled = int(round(self.N * self.done / self.total)) if self.total > 0 else 0
        if busy and self.total <= 0:      # indeterminata: pacchetto che rimbalza
            p = t % (2 * (self.N - 4))
            head = p if p < self.N - 4 else 2 * (self.N - 4) - p
        else:
            head = min(filled, self.N - 1)
        for i in range(self.N):
            x0 = self.X0 + i * (self.CW + self.GAP)
            lit = (i < filled) if self.total > 0 else (busy and head <= i < head + 4)
            if busy and i == head and self.total > 0 and (t % 4 < 2):
                lit = True
            self.create_rectangle(x0, self.BARY, x0 + self.CW, self.BARY + 14,
                                  fill=LCD_DARK if lit else LCD_OFF, outline=LCD_DARK if lit else "")
        # omino 8-bit che corre sulla barra
        if busy and anim["runner"]:
            hx = self.X0 + head * (self.CW + self.GAP)
            rows = anim["runner"][(t // 2) % len(anim["runner"])]
            sprite(self, rows, hx + 2, self.BARY - 3 - 2 * len(rows), 2, LCD_DARK)


class PageEditor:
    """Editor di pagina: strumenti a colori (crocini / rettangolo / gomma), riquadri
    semitrasparenti del taglio e conteggio degli scontrini.
    ZOOM: Ctrl+rotellina (centrato sul cursore), rotella = scorri, Maiusc+rotella = scorri di lato,
    tasto centrale o SPAZIO+trascina = sposta la vista, tasti + / - / 0 (adatta) / 9 (100%)."""
    TOOLS = {"add": ("+ CROCINI", "orange", ORANGE, "crosshair"),
             "rect": ("RETTANGOLO", "blue", BLUE, "tcross"),
             "erase": ("GOMMA", "red", RED, "X_cursor")}
    HINTS = {"add": "ARANCIO  ·  clicca il crocino ALTO e poi quello BASSO (tasto destro = annulla il primo clic)  ·  Ctrl+rotella = zoom",
             "rect": "BLU  ·  trascina per disegnare il rettangolo della foto  ·  Ctrl+rotella = zoom, spazio+trascina = sposta",
             "erase": "ROSSO  ·  clicca un riquadro (o la sua linea) per cancellare il frame sbagliato"}
    COLORS = {"ok": (52, 211, 153), "check": (255, 79, 18), "manual": (94, 123, 216), "off": (130, 130, 130)}
    ZMIN, ZMAX = 0.03, 24.0
    BG = (34, 34, 34)

    def __init__(self, app, path):
        self.app, self.path = app, path
        self.k, self.tool, self.undo, self.clicks, self.drag = 1.0, "add", [], [], None
        self.org = [0.0, 0.0]          # angolo alto-sinistra della vista, in px della scansione
        self.shapes, self.img = [], None
        self._pan, self._space, self._later, self._mouse = None, False, None, None
        w = self.win = tk.Toplevel(app.root)
        w.configure(bg=BODY)
        w.title("EDITOR PAGINA")
        bar = tk.Frame(w, bg=BODY)
        bar.pack(fill="x", padx=10, pady=8)
        self.keys = {}
        for name in ("add", "rect", "erase"):
            kk = Key(bar, self.TOOLS[name][0], lambda n=name: self.set_tool(n), width=140, height=38)
            kk.pack(side="left", padx=4)
            self.keys[name] = kk
        Key(bar, "ANNULLA", self.undo_last, width=100, height=38).pack(side="left", padx=(16, 4))
        Key(bar, "ADATTA", self.zoom_fit, width=80, height=38).pack(side="left", padx=(16, 2))
        Key(bar, "100%", self.zoom_100, width=64, height=38).pack(side="left", padx=2)
        self.zoom_lbl = tk.Label(bar, text="", bg=BODY, fg=INK, font=(FONT, 10, "bold"), width=8)
        self.zoom_lbl.pack(side="left", padx=4)
        self.cb = ttk.Combobox(bar, state="readonly", width=30,
                               values=[Path(s).name for s in app.scans])
        self.cb.pack(side="left", padx=12)
        self.cb.bind("<<ComboboxSelected>>", lambda e: self.load_page(app.scans[self.cb.current()]))
        self.counter = tk.Label(bar, text="", bg=LCD_BG, fg=LCD_DARK, font=(FONT, 12, "bold"),
                                padx=12, pady=6, highlightthickness=2, highlightbackground=EDGE)
        self.counter.pack(side="right")
        self.hint = tk.Label(w, text="", bg=BODY, fg=INK, font=(FONT, 9, "bold"), anchor="w")
        self.hint.pack(side="bottom", fill="x", padx=10, pady=4)
        wrap = tk.Frame(w, bg=BODY)
        wrap.pack(fill="both", expand=True, padx=10)
        self.cv = tk.Canvas(wrap, bg=NIGHT, highlightthickness=3, highlightbackground=ORANGE,
                            width=1280, height=820)
        self.ys = ttk.Scrollbar(wrap, command=self._yview)
        self.xs = ttk.Scrollbar(wrap, orient="horizontal", command=self._xview)
        self.ys.pack(side="right", fill="y")
        self.xs.pack(side="bottom", fill="x")
        self.cv.pack(fill="both", expand=True)
        cv = self.cv
        cv.bind("<ButtonPress-1>", self.on_press)
        cv.bind("<B1-Motion>", self.on_drag)
        cv.bind("<ButtonRelease-1>", self.on_release)
        cv.bind("<Motion>", self.on_motion)
        cv.bind("<Leave>", lambda e: cv.delete("hover"))
        cv.bind("<Button-3>", lambda e: self.cancel_pending())
        cv.bind("<ButtonPress-2>", self._pan_start)
        cv.bind("<B2-Motion>", self._pan_move)
        cv.bind("<ButtonRelease-2>", lambda e: setattr(self, "_pan", None))
        cv.bind("<MouseWheel>", self._wheel)                              # Windows / macOS
        cv.bind("<Button-4>", lambda e: self._wheel(e, 1))                # Linux
        cv.bind("<Button-5>", lambda e: self._wheel(e, -1))
        cv.bind("<Configure>", lambda e: self._soon())
        w.bind("<Control-z>", lambda e: self.undo_last())
        w.bind("1", lambda e: self.set_tool("add"))
        w.bind("2", lambda e: self.set_tool("rect"))
        w.bind("3", lambda e: self.set_tool("erase"))
        w.bind("<plus>", lambda e: self.zoom_at(1.25))
        w.bind("<equal>", lambda e: self.zoom_at(1.25))
        w.bind("<minus>", lambda e: self.zoom_at(1 / 1.25))
        w.bind("0", lambda e: self.zoom_fit())
        w.bind("9", lambda e: self.zoom_100())
        w.bind("<Control-0>", lambda e: self.zoom_fit())
        w.bind("<Control-1>", lambda e: self.zoom_100())
        w.bind("<KeyPress-space>", lambda e: setattr(self, "_space", True))
        w.bind("<KeyRelease-space>", lambda e: setattr(self, "_space", False))
        for key, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            w.bind(f"<{key}>", lambda e, dx=dx, dy=dy: self.pan_by(dx * .15, dy * .15))
        self.item = None
        self.load_page(path)
        self.set_tool("add")

    # ------------------------------------------------------- vista / zoom
    def view_size(self):
        return max(1, self.cv.winfo_width() - 6), max(1, self.cv.winfo_height() - 6)

    def _src(self, x, y):
        return self.org[0] + x / self.k, self.org[1] + y / self.k

    def _view(self, px, py):
        return (px - self.org[0]) * self.k, (py - self.org[1]) * self.k

    def _clamp(self):
        H, W = self.img.shape[:2]
        vw, vh = self.view_size()
        for i, (size, view) in enumerate(((W, vw / self.k), (H, vh / self.k))):
            if view >= size:
                self.org[i] = -(view - size) / 2
            else:
                m = 0.1 * view
                self.org[i] = min(max(self.org[i], -m), size - view + m)

    def _soon(self):
        if self.img is None:
            return
        if self._later:
            self.win.after_cancel(self._later)
        self._later = self.win.after(30, self.redraw)

    def zoom_fit(self):
        H, W = self.img.shape[:2]
        vw, vh = self.view_size()
        self.k = min(vw / W, vh / H)
        self.org = [0.0, 0.0]
        self.redraw()

    def zoom_100(self):
        vw, vh = self.view_size()
        self.zoom_to(1.0, vw / 2, vh / 2)

    def zoom_to(self, z, cx, cy):
        z = min(max(z, self.ZMIN), self.ZMAX)
        sx, sy = self._src(cx, cy)
        self.k = z
        self.org = [sx - cx / z, sy - cy / z]
        self.redraw()

    def zoom_at(self, factor, cx=None, cy=None):
        vw, vh = self.view_size()
        if cx is None:
            cx, cy = vw / 2, vh / 2
        self.zoom_to(self.k * factor, cx, cy)

    def pan_by(self, fx, fy):
        vw, vh = self.view_size()
        self.org[0] += fx * vw / self.k
        self.org[1] += fy * vh / self.k
        self.redraw()

    def _wheel(self, ev, d=None):
        if d is None:
            d = 1 if ev.delta > 0 else -1
        if ev.state & 0x4:                       # Ctrl: zoom sul cursore
            self.zoom_at(1.2 if d > 0 else 1 / 1.2, ev.x - 3, ev.y - 3)
        elif ev.state & 0x1:                     # Maiusc: scorri di lato
            self.pan_by(-d * 0.12, 0)
        else:
            self.pan_by(0, -d * 0.12)

    def _pan_start(self, ev):
        self._pan = (ev.x, ev.y, self.org[0], self.org[1])

    def _pan_move(self, ev):
        if self._pan:
            x0, y0, ox, oy = self._pan
            self.org = [ox - (ev.x - x0) / self.k, oy - (ev.y - y0) / self.k]
            self.redraw()

    def _yview(self, *a):
        H = self.img.shape[0]
        vh = self.view_size()[1] / self.k
        if a[0] == "moveto":
            self.org[1] = float(a[1]) * H
        else:
            self.org[1] += int(a[1]) * vh * (0.1 if a[2] == "units" else 0.9)
        self.redraw()

    def _xview(self, *a):
        W = self.img.shape[1]
        vw = self.view_size()[0] / self.k
        if a[0] == "moveto":
            self.org[0] = float(a[1]) * W
        else:
            self.org[0] += int(a[1]) * vw * (0.1 if a[2] == "units" else 0.9)
        self.redraw()

    def render_base(self):
        vw, vh = self.view_size()
        z, (ox, oy) = self.k, self.org
        H, W = self.img.shape[:2]
        if z >= 1.0:
            M = np.float32([[z, 0, -ox * z], [0, z, -oy * z]])
            return cv2.warpAffine(self.img, M, (vw, vh),
                                  flags=cv2.INTER_NEAREST if z >= 6 else cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=self.BG)
        out = np.full((vh, vw, 3), self.BG, np.uint8)
        x0, y0 = max(0, int(math.floor(ox))), max(0, int(math.floor(oy)))
        x1, y1 = min(W, int(math.ceil(ox + vw / z)) + 1), min(H, int(math.ceil(oy + vh / z)) + 1)
        if x1 > x0 and y1 > y0:
            sm = cv2.resize(self.img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
            dx, dy = int(round((x0 - ox) * z)), int(round((y0 - oy) * z))
            sx0, sy0 = max(0, -dx), max(0, -dy)
            tx0, ty0 = max(0, dx), max(0, dy)
            w_ = min(sm.shape[1] - sx0, vw - tx0)
            h_ = min(sm.shape[0] - sy0, vh - ty0)
            if w_ > 0 and h_ > 0:
                out[ty0:ty0 + h_, tx0:tx0 + w_] = sm[sy0:sy0 + h_, sx0:sx0 + w_]
        return out

    # ------------------------------------------------------------ pagina
    def load_page(self, path):
        a = self.app
        self.path = path
        self.cb.set(Path(path).name)
        self.win.title(f"EDITOR PAGINA — {Path(path).name}")
        self.counter.configure(text="CARICO...")
        self.win.update_idletasks()
        if a.tpl is None:
            a.tpl = extract_mark_template(a.v_tpl.get().strip() or None)
        setup_tesseract(a.v_tess.get().strip() or None)
        self.img = imread_unicode(path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        H, W = self.gray.shape
        meta = a.meta.get(path, {})
        self.D = meta.get("D") or (float(a.v_D.get()) or 0.187 * W)
        self.dm = meta.get("d") or 0.0125 * W
        self.cv.delete("all")
        self.item = self.cv.create_image(3, 3, anchor="nw")
        self.undo.clear()
        self.cancel_pending()
        self.win.update_idletasks()
        self.zoom_fit()

    def frames_here(self):
        return [f for f in self.app.frames if f["scan"] == self.path]

    def redraw(self):
        if self.img is None:
            return
        self._later = None
        a, z = self.app, self.k
        self._clamp()
        ox, oy = self.org
        geom = a.geom()
        self.shapes = []
        for f in self.frames_here():
            if not f["include"]:
                kind = "off"
            elif f.get("manual"):
                kind = "manual"
            elif f.get("num") is None or not (f["status"] in ("ok", "dedotto") or f["status"].startswith("manuale")
                                              or f["status"].startswith("PDF p.")):
                kind = "check"
            else:
                kind = "ok"
            poly = [(x - ox * z, y - oy * z) for x, y in footprint_poly(f["pair"], f["D"], geom, z)]
            self.shapes.append(dict(f=f, poly=poly,
                                    top=self._view(*f["pair"]["top"]), bot=self._view(*f["pair"]["bot"]),
                                    color=self.COLORS[kind]))
        base = Image.fromarray(cv2.cvtColor(self.render_base(), cv2.COLOR_BGR2RGB)).convert("RGBA")
        comp = compose_overlay(base, self.shapes)
        self._ph = ImageTk.PhotoImage(comp.convert("RGB"))
        self.cv.itemconfig(self.item, image=self._ph)
        self.cv.delete("lab")
        vw, vh = self.view_size()
        for s in self.shapes:
            f = s["f"]
            x, y = s["poly"][0]
            if not (-200 < x < vw + 200 and -200 < y < vh + 200):
                continue
            txt = "?" if f.get("num") is None else str(f["num"])
            self.cv.create_rectangle(x + 12, y + 12, x + 18 + 11 * len(txt), y + 34, fill=LCD_BG,
                                     outline=LCD_DARK, width=2, tags="lab")
            self.cv.create_text(x + 15, y + 23, text=txt, anchor="w", fill=LCD_DARK,
                                font=(FONT, 11, "bold"), tags="lab")
        self.draw_tmp()
        n = sum(1 for f in self.frames_here() if f["include"])
        self.counter.configure(text=f"{n} SCONTRINO RICONOSCIUTO" if n == 1 else f"{n} SCONTRINI RICONOSCIUTI")
        self.zoom_lbl.configure(text=f"{self.k * 100:.0f}%")
        H, W = self.img.shape[:2]
        sw, sh = vw / z, vh / z
        self.xs.set(max(0, ox / W), min(1, (ox + sw) / W))
        self.ys.set(max(0, oy / H), min(1, (oy + sh) / H))

    def draw_tmp(self):
        self.cv.delete("tmp")
        for (px, py) in self.clicks:
            x, y = self._view(px, py)
            self.cv.create_oval(x - 7 + 3, y - 7 + 3, x + 7 + 3, y + 7 + 3, outline=ORANGE, width=3, tags="tmp")
        if self.tool == "rect" and self.drag and self._mouse:
            x0, y0 = self._view(*self.drag)
            self.cv.create_rectangle(x0 + 3, y0 + 3, self._mouse[0] + 3, self._mouse[1] + 3,
                                     outline=BLUE, width=3, dash=(6, 4), tags="tmp")

    # ---------------------------------------------------------- strumenti
    def set_tool(self, name):
        self.tool = name
        for n, kk in self.keys.items():
            kk.set_kind(self.TOOLS[n][1] if n == name else "cream")
        col, cur = self.TOOLS[name][2], self.TOOLS[name][3]
        self.cv.configure(highlightbackground=col)
        try:
            self.cv.configure(cursor=cur)
        except tk.TclError:
            self.cv.configure(cursor="crosshair")
        self.hint.configure(text=self.HINTS[name], fg=col)
        self.cancel_pending()

    def cancel_pending(self):
        self.clicks.clear()
        self.drag = None
        self.cv.delete("tmp")
        self.cv.delete("hover")

    def target_at(self, x, y):
        best, bd = None, 1e9
        for s in self.shapes:
            poly = np.array(s["poly"], np.float32)
            inside = cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
            d = seg_dist(x, y, s["top"], s["bot"])
            if inside or d < 25:
                if d < bd:
                    best, bd = s, d
        return best

    def on_motion(self, ev):
        x, y = ev.x - 3, ev.y - 3
        self._mouse = (x, y)
        self.cv.delete("hover")
        if self.tool == "erase":
            s = self.target_at(x, y)
            if s:
                pts = [c + 3 for p in s["poly"] for c in p]
                self.cv.create_polygon(*pts, outline=RED, fill="", width=3, tags="hover")
        else:
            # mirino a croce sottile su tutta la vista: serve per puntare con precisione
            vw, vh = self.view_size()
            col = ORANGE if self.tool == "add" else BLUE
            self.cv.create_line(3, y + 3, vw + 3, y + 3, fill=col, width=1, dash=(2, 4), tags="hover")
            self.cv.create_line(x + 3, 3, x + 3, vh + 3, fill=col, width=1, dash=(2, 4), tags="hover")
            if self.tool == "rect" and self.drag:
                self.draw_tmp()

    def on_press(self, ev):
        x, y = ev.x - 3, ev.y - 3
        if self._space:
            self._pan_start(ev)
            self._pan_by_left = True
            return
        if self.tool == "erase":
            s = self.target_at(x, y)
            if s and self.app.is_busy:
                self.app.write("Operazione in corso: attendi o premi ANNULLA prima di cancellare frame.")
                return
            if s:
                self.app.frames.remove(s["f"])
                self.app.add_tombstone(s["f"])           # la gomma resta anche dopo una nuova analisi
                self.undo.append(("del", s["f"]))
                self.changed(f"Frame cancellato (era n° {s['f'].get('num')}).")
        elif self.tool == "add":
            sx, sy = self._src(x, y)
            px, py = refine_mark(self.gray, sx, sy, self.D, self.dm, self.app.tpl)
            self.clicks.append((px, py))
            self.draw_tmp()
            if len(self.clicks) == 2:
                (tx, ty), (bx, by) = sorted(self.clicks, key=lambda c: c[1])
                self.clicks.clear()
                self.commit(dict(top=(tx, ty), bot=(bx, by), score=1.0), self.D, "crocini")
        elif self.tool == "rect":
            self.drag = self._src(x, y)
            self._mouse = (x, y)
            self.draw_tmp()

    def on_drag(self, ev):
        if self._pan and getattr(self, "_pan_by_left", False):
            self._pan_move(ev)
            return
        if self.tool == "rect" and self.drag:
            self._mouse = (ev.x - 3, ev.y - 3)
            self.draw_tmp()

    def on_release(self, ev):
        if getattr(self, "_pan_by_left", False):
            self._pan_by_left = False
            self._pan = None
            return
        if self.tool != "rect" or not self.drag:
            return
        x, y = self._src(ev.x - 3, ev.y - 3)
        x0, x1 = sorted((self.drag[0], x))
        y0, y1 = sorted((self.drag[1], y))
        self.drag = None
        self.cv.delete("tmp")
        w, h = x1 - x0, y1 - y0
        if w * self.k < 60 or h * self.k < 40:
            return
        ref = self.app.geom()["ref_d"]
        s = 0.5 * (w / RECT["w"] + h / RECT["h"])            # px scansione per px di riferimento
        D = s * ref
        cx = (x0 + x1) / 2 - RECT["dx"] * s
        ty = y0 - RECT["dy"] * s
        self.commit(dict(top=(cx, ty), bot=(cx, ty + D), score=1.0), D, "rettangolo")

    # ---------------------------------------------------------- modifiche
    def commit(self, pair, D, how):
        a = self.app
        self.win.configure(cursor="watch")
        self.win.update_idletasks()
        a.manual_counter += 1
        fr = make_frame(self.img, self.gray, pair, D, a.geom(), self.path,
                        (self.path, "m", a.manual_counter), 0, -1)
        fr["manual"] = True
        a.frames.append(fr)
        self.undo.append(("add", fr))
        self.win.configure(cursor="")
        self.changed(f"Frame aggiunto da {how} (OCR: {fr['ocr']}, rettangolo {fr['rect_score']:.2f}).")

    def undo_last(self):
        if not self.undo:
            return
        kind, fr = self.undo.pop()
        if kind == "add" and fr in self.app.frames:
            self.app.frames.remove(fr)
        elif kind == "del":
            self.app.frames.append(fr)
            self.app.remove_tombstone(fr)
        self.changed("Annullato.")

    def changed(self, msg):
        a = self.app
        infer_numbers(a.frames, a.direction())
        a.refresh()
        a.write(msg)
        self.redraw()


class App:
    def __init__(self, root):
        self.root = root
        root.title("SCONTRINI-01  ·  frame cutter")
        root.configure(bg=BODY)
        self.scans, self.frames, self.meta = [], [], {}
        self.tpl = None
        self.q = queue.Queue()
        self._jq = queue.Queue()                 # coda delle operazioni in background (anteprime, riconoscimento)
        self.is_busy = False
        self.cancel_ev = threading.Event()
        self.scan_cache = ScanCache(2)
        self._pending = []                       # decisioni manuali da riapplicare dopo l'analisi
        self._erased = []                        # frame cancellati con la gomma: non vanno riproposti alla rianalisi
        self._frames_rot = set()                 # scansioni ruotate (180) quando sono stati creati i frame attuali
        self._t_job = None
        self.sort_col, self.sort_rev = None, False      # ordinamento della tabella (solo vista)
        self.v_gaps = tk.StringVar(value="")
        self._sec_labels = []
        self.manual_counter = 0
        self._photo = self._roi_photo = None

        self.v_tpl, self.v_out, self.v_tess = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.v_thr = tk.DoubleVar(value=0.5)
        self.v_D = tk.DoubleVar(value=0)
        self.v_dir = tk.StringVar(value="Auto")
        self.v_fmt = tk.StringVar(value="PNG")
        self.v_pad = tk.IntVar(value=0)
        self.v_hw = tk.DoubleVar(value=DEFAULT_GEOM["half_w"])
        self.v_top = tk.DoubleVar(value=DEFAULT_GEOM["top"])
        self.v_bot = tk.DoubleVar(value=DEFAULT_GEOM["bottom"])
        self.v_ref = tk.DoubleVar(value=DEFAULT_GEOM["ref_d"])
        self.v_num = tk.StringVar()
        self.v_native = tk.BooleanVar(value=False)
        self.v_omode = tk.StringVar(value=OUT_MODES[0])
        self.v_orect = tk.BooleanVar(value=True)
        self.v_opdf = tk.BooleanVar(value=True)
        self.v_ochk = tk.BooleanVar(value=True)
        self.v_opaper = tk.BooleanVar(value=True)      # zone senza carta nella scansione: tinta della carta, non bianco
        self.v_oover = tk.DoubleVar(value=1.0)
        self.v_ofit = tk.StringVar(value=OUT_FITS[0])
        self.v_ow, self.v_oh = tk.IntVar(value=1080), tk.IntVar(value=1080)
        self.v_osz = tk.StringVar(value="")
        self._style()
        self._build()
        try:
            self.lcd.flags["OCR"] = bool(setup_tesseract(None))
            self.led_ocr.set(self.lcd.flags["OCR"])
        except Exception:
            pass

    # ------------------------------------------------------------- stile
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Treeview", background=KEY, fieldbackground=KEY, foreground=INK,
                     rowheight=24, font=(FONT, 9), bordercolor=EDGE, borderwidth=2)
        st.map("Treeview", background=[("selected", ACTIVE)], foreground=[("selected", SEL_FG)])
        st.configure("Treeview.Heading", background=HEAD_BG, foreground=HEAD_FG, font=(FONT, 9, "bold"),
                     relief="flat", borderwidth=0)
        st.map("Treeview.Heading", background=[("active", ACTIVE)], foreground=[("active", SEL_FG)])
        st.configure("TCombobox", fieldbackground=FIELD, background=KEY, foreground=INK,
                     arrowcolor=INK, bordercolor=EDGE, lightcolor=KEY, darkcolor=KEY)
        st.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", INK)])
        st.configure("Vertical.TScrollbar", background=KEY, troughcolor=BODY, bordercolor=EDGE,
                     arrowcolor=INK)
        st.configure("Horizontal.TScrollbar", background=KEY, troughcolor=BODY, bordercolor=EDGE,
                     arrowcolor=INK)
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACTIVE)
        self.root.option_add("*TCombobox*Listbox.foreground", INK)
        self.root.option_add("*TCombobox*Listbox.font", (FONT, 9))

    def _section(self, parent, title, color, collapsible=False):
        box = tk.Frame(parent, bg=BODY)
        head = tk.Frame(box, bg=BODY)
        head.pack(fill="x")
        lab = tk.Label(head, text=title, bg=BODY, fg=color, font=(FONT, 9, "bold"))
        lab.pack(side="left")
        self._sec_labels.append(lab)
        tk.Frame(head, bg=color, height=2).pack(side="left", fill="x", expand=True, padx=(6, 0), pady=2)
        body = tk.Frame(box, bg=BODY)
        body.pack(fill="both", expand=True, pady=(4, 0))
        if collapsible:
            # sezione richiudibile: clic sul titolo o sulla freccia; lo stato resta nelle preferenze
            arrow = tk.Label(head, text="▾", bg=BODY, fg=color, font=(FONT, 9, "bold"), cursor="hand2")
            arrow.pack(side="left", before=lab, padx=(0, 4))
            lab.configure(cursor="hand2")
            state = {"open": True}

            def set_open(flag, save=True):
                state["open"] = bool(flag)
                if flag:
                    body.pack(fill="both", expand=True, pady=(4, 0))
                else:
                    body.pack_forget()
                arrow.configure(text="▾" if flag else "▸")
                if save:
                    prefs = load_prefs()
                    shut = [t for t in prefs.get("collapsed", []) if t != title]
                    if not flag:
                        shut.append(title)
                    prefs["collapsed"] = shut
                    save_prefs(prefs)
            for w in (arrow, lab):
                w.bind("<Button-1>", lambda e: set_open(not state["open"]))
            if title in load_prefs().get("collapsed", []):
                set_open(False, save=False)
        return box, body

    def _entry(self, parent, var, width=30):
        return tk.Entry(parent, textvariable=var, width=width, bg=FIELD, fg=INK, relief="flat",
                        font=(FONT, 10), highlightthickness=2, highlightbackground=EDGE,
                        highlightcolor=ACTIVE, insertbackground=ACTIVE)

    def _spin(self, parent, var, lo, hi, inc, width=7):
        return tk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=inc, width=width,
                          bg=FIELD, fg=INK, relief="flat", font=(FONT, 10), highlightthickness=2,
                          highlightbackground=EDGE, highlightcolor=ACTIVE, buttonbackground=KEY)

    def _lab(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg=BODY, fg=INK, font=(FONT, 9), **kw)

    # ------------------------------------------------------------ layout
    def _build(self):
        r = self.root
        self.lcd = Lcd(r)
        self.lcd.pack(padx=12, pady=(12, 8))

        row = tk.Frame(r, bg=BODY)
        row.pack(fill="x", padx=12)
        row.columnconfigure(0, weight=3)
        row.columnconfigure(1, weight=2)

        # --- FILES
        files_box, f = self._section(row, "FILES", BLUE)
        files_box.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        f.columnconfigure(1, weight=1)
        self._lab(f, "TEMPLATE").grid(row=0, column=0, sticky="w")
        self._entry(f, self.v_tpl).grid(row=0, column=1, sticky="we", padx=6, pady=2)
        Key(f, "...", self.pick_tpl, width=40, height=26).grid(row=0, column=2)
        self._lab(f, "SCANSIONI").grid(row=1, column=0, sticky="nw", pady=4)
        self.lb = tk.Listbox(f, height=3, bg=FIELD, fg=INK, relief="flat", font=(FONT, 9),
                             highlightthickness=2, highlightbackground=EDGE, selectbackground=ACTIVE)
        self.lb.grid(row=1, column=1, sticky="we", padx=6, pady=4)
        kb = tk.Frame(f, bg=BODY)
        kb.grid(row=1, column=2, sticky="n")
        Key(kb, "FILE+", self.add_files, width=70, height=24).pack()
        Key(kb, "DIR+", self.add_folder, width=70, height=24).pack()
        Key(kb, "CLEAR", self.clear_scans, width=70, height=24).pack()
        self._lab(f, "OUTPUT").grid(row=2, column=0, sticky="w")
        self._entry(f, self.v_out).grid(row=2, column=1, sticky="we", padx=6, pady=2)
        Key(f, "...", self.pick_out, width=40, height=26).grid(row=2, column=2)
        self._lab(f, "TESSERACT").grid(row=3, column=0, sticky="w")
        self._entry(f, self.v_tess).grid(row=3, column=1, sticky="we", padx=6, pady=2)
        Key(f, "...", self.pick_tess, width=40, height=26).grid(row=3, column=2)

        # --- PARAM
        par_box, p = self._section(row, "PARAM", ORANGE, collapsible=True)
        par_box.grid(row=0, column=1, sticky="nsew")

        def line(r_, text, widget):
            self._lab(p, text).grid(row=r_, column=0, sticky="e", padx=(0, 6), pady=1)
            widget.grid(row=r_, column=1, sticky="w", pady=1)
        line(0, "SOGLIA CROCINI", self._spin(p, self.v_thr, 0.2, 0.95, 0.05))
        line(1, "D CROCINI px (0=AUTO)", self._spin(p, self.v_D, 0, 5000, 10))
        cb = ttk.Combobox(p, textvariable=self.v_dir, state="readonly", width=12,
                          values=["Auto", "Decrescenti", "Crescenti"])
        line(2, "NUMERI SCENDENDO", cb)
        fm = ttk.Combobox(p, textvariable=self.v_fmt, state="readonly", width=6, values=["PNG", "JPEG"])
        line(3, "FORMATO", fm)
        line(4, "ZERI NEL NOME", self._spin(p, self.v_pad, 0, 6, 1, 4))
        g = tk.Frame(p, bg=BODY)
        g.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        for i, (lab, var, hi) in enumerate((("MEZZA L.", self.v_hw, 5000), ("SOPRA", self.v_top, 2000),
                                            ("SOTTO", self.v_bot, 2000), ("D RIF.", self.v_ref, 5000))):
            tk.Label(g, text=lab, bg=BODY, fg=MUTED, font=(FONT, 7, "bold")).grid(row=0, column=i, padx=3)
            self._spin(g, var, 0, hi, 10, 6).grid(row=1, column=i, padx=3)

        # --- barra azioni
        act = tk.Frame(r, bg=BODY)
        act.pack(fill="x", padx=12, pady=10)
        self.btn_an = Key(act, "1  ANALIZZA", self.analyze, kind="orange", width=170, height=44)
        self.btn_an.pack(side="left", padx=(0, 10))
        self.btn_add = Key(act, "EDITOR PAGINA", self.manual_add, width=170, height=44)
        self.btn_add.pack(side="left", padx=(0, 10))
        self.btn_save = Key(act, "2  SALVA", self.save_all, kind="green", width=170, height=44)
        self.btn_save.pack(side="left")
        ob, ofr = self._section(act, "USCITA (uguale per tutti)", GREEN)
        ob.pack(side="left", padx=(16, 0))
        cbm = ttk.Combobox(ofr, textvariable=self.v_omode, state="readonly", width=19, values=OUT_MODES)
        cbm.grid(row=0, column=0, columnspan=3, sticky="w", pady=1)
        self._spin(ofr, self.v_ow, 16, 20000, 2, 6).grid(row=0, column=3, padx=(6, 0))
        tk.Label(ofr, text="x", bg=BODY, fg=INK, font=(FONT, 9)).grid(row=0, column=4)
        self._spin(ofr, self.v_oh, 16, 20000, 2, 6).grid(row=0, column=5)
        cbf = ttk.Combobox(ofr, textvariable=self.v_ofit, state="readonly", width=19, values=OUT_FITS)
        cbf.grid(row=1, column=0, columnspan=3, sticky="w", pady=1)
        tk.Label(ofr, textvariable=self.v_osz, bg=BODY, fg=GREEN, font=(FONT, 9, "bold")).grid(
            row=1, column=3, columnspan=3, sticky="w", padx=(6, 0))
        tk.Checkbutton(ofr, text="PDF PRIORITARIO (sovrapposizione)", variable=self.v_opdf, bg=BODY, fg=INK,
                       activebackground=BODY, selectcolor=FIELD, font=(FONT, 8),
                       command=self.update_osz).grid(row=2, column=0, columnspan=3, sticky="w")
        tk.Checkbutton(ofr, text="RETTANGOLO (frame senza PDF)", variable=self.v_orect, bg=BODY, fg=INK,
                       activebackground=BODY, selectcolor=FIELD, font=(FONT, 8),
                       command=self.update_osz).grid(row=3, column=0, columnspan=3, sticky="w")
        tk.Checkbutton(ofr, text="SALVA CONTROLLO SOVRAPPOSIZIONE", variable=self.v_ochk, bg=BODY, fg=INK,
                       activebackground=BODY, selectcolor=FIELD, font=(FONT, 8)).grid(
                           row=4, column=0, columnspan=4, sticky="w")
        tk.Checkbutton(ofr, text="CARTA MANCANTE: TINTA DELLA CARTA (non bianco)", variable=self.v_opaper, bg=BODY,
                       fg=INK, activebackground=BODY, selectcolor=FIELD, font=(FONT, 8),
                       command=self.update_osz).grid(row=5, column=0, columnspan=6, sticky="w")
        tk.Label(ofr, text="TAGLIO BORDO px", bg=BODY, fg=INK, font=(FONT, 8)).grid(row=2, column=3, columnspan=2, sticky="e")
        self._spin(ofr, self.v_oover, 0, 20, 0.5, 4).grid(row=2, column=5)
        cbm.bind("<<ComboboxSelected>>", lambda e: self.update_osz())
        cbf.bind("<<ComboboxSelected>>", lambda e: self.update_osz())
        leds = tk.Frame(act, bg=BODY)
        leds.pack(side="right")
        self.led_ocr = Led(leds, "OCR", GREEN)
        self.led_ocr.pack(side="left")
        self.led_busy = Led(leds, "BUSY", ACTIVE)
        self.led_busy.pack(side="left")

        # --- tabella + anteprima
        mid = tk.Frame(r, bg=BODY)
        mid.pack(fill="both", expand=True, padx=12)
        tbox, tb = self._section(mid, "FRAMES", GREEN)
        tbox.pack(side="left", fill="both", expand=True, padx=(0, 14))
        cols = ("scan", "pos", "rett", "ocr", "num", "stato")
        tk.Label(tb, textvariable=self.v_gaps, bg=BODY, fg=MUTED, font=(FONT, 9, "bold"), anchor="w",
                 justify="left", wraplength=560).pack(side="bottom", fill="x", pady=(4, 0))
        self.tree = ttk.Treeview(tb, columns=cols, show="headings", height=11, selectmode="browse")
        for c, w in (("scan", 150), ("pos", 70), ("rett", 110), ("ocr", 55), ("num", 60), ("stato", 170)):
            self.tree.heading(c, text=SORT_TITLES[c], command=lambda c=c: self.sort_by(c))
            self.tree.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(tb, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self._tree_tags()

        pbox, pv = self._section(mid, "PREVIEW", RED)
        pbox.pack(side="left", fill="y")
        self.canvas = tk.Label(pv, text="", bg=NIGHT, fg=CANVAS_FG, width=44, height=15, font=(FONT, 9))
        self.canvas.pack()
        self.roi_lbl = tk.Label(pv, text="", bg=NIGHT, width=44)
        self.roi_lbl.pack(pady=4, fill="x")
        ef = tk.Frame(pv, bg=BODY)
        ef.pack(fill="x", pady=2)
        self._lab(ef, "N°").pack(side="left")
        ent = tk.Entry(ef, textvariable=self.v_num, width=6, bg=FIELD, fg=INK, relief="flat",
                       font=(FONT, 16, "bold"), highlightthickness=2, highlightbackground=EDGE,
                       highlightcolor=ORANGE, justify="center")
        ent.pack(side="left", padx=6)
        ent.bind("<Return>", lambda e: self.set_num(False))
        Key(ef, "SET", lambda: self.set_num(False), width=60, height=30).pack(side="left", padx=2)
        Key(ef, "SET + SEQ", lambda: self.set_num(True), kind="orange", width=100, height=30).pack(side="left", padx=2)
        Key(pv, "INCLUDI / ESCLUDI", self.toggle, width=250, height=28).pack(pady=4, anchor="w")

        # --- console
        self.log = tk.Text(r, height=5, state="disabled", bg=LCD_DARK, fg=LCD_LIGHT, relief="flat",
                           font=(FONT, 9), highlightthickness=3, highlightbackground=EDGE,
                           insertbackground=LCD_LIGHT, padx=8, pady=4)
        self.log.pack(fill="x", padx=12, pady=12)

    # -------------------------------------------------------------- utilità
    def write(self, msg):
        file_log(msg)
        self.log.configure(state="normal")
        self.log.insert("end", "> " + msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def busy(self, flag):
        self.is_busy = bool(flag)
        for b in (self.btn_an, self.btn_add, self.btn_save):
            b.enable(not flag)
        for name in ("btn_prev", "btn_retry", "pdf_prepare_button", "btn_proj_save", "btn_proj_load", "btn_erased"):
            b = getattr(self, name, None)
            if b is not None:
                b.enable(not flag)
        b = getattr(self, "btn_cancel", None)
        if b is not None:
            b.enable(bool(flag))
        self.led_busy.set(flag)

    def _log_time(self, name):
        if self._t_job is not None:
            self.write(f"[tempo] {name}: {time.perf_counter() - self._t_job:.1f} s")

    def pick_tpl(self):
        p = filedialog.askopenfilename(filetypes=[("Immagini", "*.png *.jpg *.jpeg"), ("Tutti", "*.*")])
        if p:
            self.v_tpl.set(p)

    def pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.v_out.set(p)

    def pick_tess(self):
        p = filedialog.askopenfilename(filetypes=[("tesseract", "tesseract.exe"), ("Tutti", "*.*")])
        if p:
            self.v_tess.set(p)

    def add_files(self):
        for p in filedialog.askopenfilenames(filetypes=[("Immagini", "*.png *.jpg *.jpeg *.tif *.tiff"),
                                                        ("Tutti", "*.*")]):
            if p not in self.scans:
                self.scans.append(p)
                self.lb.insert("end", p)

    def add_folder(self):
        d = filedialog.askdirectory()
        if d:
            for p in sorted(Path(d).iterdir()):
                if p.suffix.lower() in IMG_EXT and str(p) not in self.scans:
                    self.scans.append(str(p))
                    self.lb.insert("end", str(p))

    def clear_scans(self):
        self.scans.clear()
        self.lb.delete(0, "end")

    def geom(self):
        return dict(ref_d=float(self.v_ref.get()), half_w=float(self.v_hw.get()),
                    top=float(self.v_top.get()), bottom=float(self.v_bot.get()))

    def direction(self):
        return {"Auto": 0, "Decrescenti": -1, "Crescenti": 1}[self.v_dir.get()]

    # ------------------------------------------------------------- analisi
    def analyze(self):
        if self.is_busy:
            return
        if not self.scans:
            messagebox.showerror("Errore", "Aggiungi almeno una scansione.")
            return
        mine = [f for f in self.frames if f.get("num_manual") or f.get("manual") or not f["include"]]
        if mine:
            ans = messagebox.askyesnocancel(
                "Decisioni manuali",
                f"Ci sono {len(mine)} decisioni manuali (numeri, esclusioni o frame aggiunti).\n\n"
                "SI = le conservo e le riapplico dopo la nuova analisi\n"
                "NO = le scarto\nANNULLA = non analizzare")
            if ans is None:
                return
            if ans:
                self._pending = merge_decisions(self.collect_decisions(), self._pending)
            else:
                self._pending = []
        ok = setup_tesseract(self.v_tess.get().strip() or None)
        self.lcd.flags["OCR"] = bool(ok)
        self.led_ocr.set(bool(ok))
        if not ok:
            self.write("ATTENZIONE: Tesseract/pytesseract non disponibile -> numeri da inserire a mano "
                       "(pip install pytesseract + installare Tesseract-OCR).")
        self.tpl = extract_mark_template(self.v_tpl.get().strip() or None)
        self.frames, self.meta = [], {}
        self.tree.delete(*self.tree.get_children())
        self.cancel_ev.clear()
        self.scan_cache.clear()
        self._t_job = time.perf_counter()
        self.busy(True)
        self.lcd.set("busy", "ANALISI", "avvio...", 0, len(self.scans))
        args = (list(self.scans), float(self.v_thr.get()), float(self.v_D.get()), self.geom())
        threading.Thread(target=self._worker, args=args, daemon=True).start()
        self.root.after(150, self._poll)

    def _worker(self, scans, thr, D, geom):
        n = len(scans)
        for i, p in enumerate(scans, 1):
            self.q.put(("prog", i - 1, n, "ANALISI", Path(p).name))
            try:
                fr, meta = analyze_scan(p, self.tpl, thr, D, geom, log=lambda m: self.q.put(("log", m)))
                self.q.put(("res", p, fr, meta))
            except Exception as ex:
                self.q.put(("log", f"ERRORE {Path(p).name}: {ex}"))
            self.q.put(("prog", i, n, "ANALISI", Path(p).name))
        self.q.put(("done",))

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                k = m[0]
                try:
                    if k == "log":
                        self.write(m[1])
                    elif k == "prog":
                        self.lcd.set("busy", m[3], m[4], m[1], m[2])
                    elif k == "res":
                        self.frames += m[2]
                        self.meta[m[1]] = m[3]
                    elif k == "add":
                        self.frames += m[1]
                    elif k == "done":
                        cancelled = self.cancel_ev.is_set()
                        if not cancelled:
                            self._apply_erased()
                            self._apply_pending_matches()
                        infer_numbers(self.frames, self.direction())
                        self.refresh()
                        self.lcd.flags["RECT"] = any(f["pair"].get("corr") is not None for f in self.frames)
                        self.lcd.set("idle", "ANNULLATO" if cancelled else "FATTO",
                                     f"{len(self.frames)} frame trovati: controlla le righe colorate",
                                     len(self.scans), len(self.scans))
                        self.busy(False)
                        ed = getattr(self, "_editor", None)
                        if ed is not None and ed.win.winfo_exists():
                            ed.redraw()
                        self.write(("Analisi ANNULLATA: " if cancelled else "Fatto: ") +
                                   f"{len(self.frames)} frame. Controlla le righe colorate, poi SALVA.")
                        self._log_time("analisi")
                        self.autosave()
                        return
                    elif k == "saved":
                        r = m[1]
                        title, sub, level, text = summarize_save(r)
                        self.lcd.set("idle", title, sub, r["planned"] - len(r["not_exported"]), max(1, r["planned"]))
                        self.busy(False)
                        self._log_time("salvataggio")
                        self.write(f"Salvataggio {title}: " + text.replace("\n", " | "))
                        if level == "info":
                            messagebox.showinfo("Finito", text)
                        elif level == "warning":
                            messagebox.showwarning("Salvataggio incompleto", text)
                        elif level == "error":
                            messagebox.showerror("Salvataggio non riuscito", text)
                        return
                except Exception as ex:
                    file_log(f"ERRORE gestione messaggio '{k}':\n{traceback.format_exc()}")
                    self.write(f"ERRORE interno ({k}): {ex}  (dettagli in {LOG_FILE})")
                    if k in ("done", "saved"):
                        self.busy(False)      # non lasciare mai l'app bloccata in stato 'occupato'
                        return
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    # ---------------------------------------------------------- tabella
    def sort_by(self, col):
        """Clic sull'intestazione: crescente -> decrescente -> ordine naturale."""
        if self.sort_col != col:
            self.sort_col, self.sort_rev = col, False
        elif not self.sort_rev:
            self.sort_rev = True
        else:
            self.sort_col, self.sort_rev = None, False
        self.refresh(keep_view=False)           # dopo un riordino la riga selezionata deve restare visibile

    def _update_headings(self):
        for c, t in SORT_TITLES.items():
            arrow = ""
            if c == self.sort_col:
                arrow = " ▼" if self.sort_rev else " ▲"
            self.tree.heading(c, text=t + arrow)

    def refresh(self, keep_view=True):
        sel = self.tree.selection()
        top = self.tree.yview()[0]               # posizione di scorrimento: la tabella viene ricostruita
        self.tree.delete(*self.tree.get_children())
        # ordine naturale dell'elenco (non cambia mai); l'ordinamento scelto riguarda solo la vista
        self.frames.sort(key=lambda f: (f["scan"], f["col"] if f["col"] >= 0 else 99, f["pair"]["top"][1]))
        counts = {}
        for f in self.frames:
            if f["include"] and f.get("num") is not None:
                counts[f["num"]] = counts.get(f["num"], 0) + 1
        for i in sorted_order(self.frames, self.sort_col, self.sort_rev):
            f = self.frames[i]
            st = f["status"]
            if f["include"] and f.get("num") is not None and counts.get(f["num"], 0) > 1:
                st += " | DUPLICATO"
            tag = ()
            if not f["include"]:
                tag, st = ("off",), "ESCLUSO"
            elif f.get("num") is None or "DUPLICATO" in st or "mancante" in st or "incerto" in st:
                tag = ("bad",)
            elif st not in ("ok", "dedotto", "manuale"):
                tag = ("warn",)
            # avvisi ed errori riconoscibili anche senza colore
            mark = "[X] " if tag == ("bad",) else ("[!] " if tag == ("warn",) else "")
            rect = ("agganciato %.2f" % f["rect_score"]) if f["pair"].get("corr") is not None \
                else ("solo crocini %.2f" % f["rect_score"])
            self.tree.insert("", "end", iid=str(i), tags=tag, values=(
                Path(f["scan"]).name, f"{f['col']}/{int(f['pair']['top'][1])}", rect,
                "" if f["ocr"] is None else f["ocr"], "" if f.get("num") is None else f["num"], mark + st))
        self._update_headings()
        self.v_gaps.set(gaps_text(self.frames))
        if keep_view:
            self.tree.update_idletasks()
            self.tree.yview_moveto(top)          # SET / SET SEQ non fanno tornare la lista in cima
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel[0])
            if not keep_view:
                self.tree.see(sel[0])
        self.update_osz()

    def cur(self):
        s = self.tree.selection()
        return (int(s[0]), self.frames[int(s[0])]) if s else (None, None)

    def on_select(self, _e=None):
        i, f = self.cur()
        if f is None:
            return
        self._photo = bgr_to_tk(f["thumb"], 380, 280)
        self.canvas.configure(image=self._photo, width=380, height=280)
        if f.get("roi") is not None and f["roi"].size:
            self._roi_photo = bgr_to_tk(f["roi"], 300, 90)
            self.roi_lbl.configure(image=self._roi_photo)
        self.v_num.set("" if f.get("num") is None else str(f["num"]))

    def set_num(self, renumber):
        i, f = self.cur()
        if f is None:
            return
        if self.is_busy:
            self.write("Operazione in corso: attendi o premi ANNULLA prima di cambiare i numeri.")
            return
        try:
            n = int(self.v_num.get())
        except ValueError:
            messagebox.showerror("Errore", "Numero non valido.")
            return
        d = f.get("dir") or self.direction() or -1
        was_missing = f.get("num") is None          # stavo riempiendo un frame senza numero?
        f["num"], f["status"], f["locked"], f["num_manual"] = n, "manuale", True, True
        if renumber:
            for g in self.frames:
                if g is not f and g["run"] == f["run"]:
                    g["num"] = n + d * (g["idx"] - f["idx"])
                    g["status"], g["locked"], g["num_manual"] = "manuale (sequenza)", True, True
        self.refresh()
        # riempiendo i numeri mancanti: passa da solo al prossimo frame senza numero; finiti quelli, resta dov'e'
        nxt = next_missing(self.frames, list(self.tree.get_children()), str(i)) if was_missing else None
        self.tree.selection_set(nxt or str(i))
        if nxt:
            self.tree.see(nxt)
        self.autosave()

    def toggle(self):
        i, f = self.cur()
        if f is not None:
            if self.is_busy:
                self.write("Operazione in corso: attendi o premi ANNULLA.")
                return
            f["include"] = not f["include"]
            self.refresh()
            self.tree.selection_set(str(i))
            self.autosave()

    # ----------------------------------------------------- aggiunta manuale
    def manual_add(self):
        if not self.scans:
            messagebox.showerror("Errore", "Aggiungi prima una scansione.")
            return
        sel = self.cur()[1]
        path = sel["scan"] if sel else self.scans[0]
        ed = getattr(self, "_editor", None)
        if ed is not None and ed.win.winfo_exists():
            ed.win.lift()
            if ed.path != path:
                ed.load_page(path)
            return
        self._editor = PageEditor(self, path)

    # ---------------------------------------------------------------- salva
    def save_all(self):
        if self.is_busy:
            return
        outs = self.v_out.get().strip()
        out = Path(outs or ".")
        if not outs or not out.is_dir():
            messagebox.showerror("Errore", "Scegli una cartella OUTPUT valida.")
            return
        todo = [f for f in self.frames if f["include"]]
        if not todo:
            messagebox.showinfo("Info", "Nessun frame da salvare.")
            return
        missing = [f for f in todo if f.get("num") is None]
        if missing and not messagebox.askyesno(
                "Numeri mancanti",
                f"{len(missing)} frame non hanno un numero: verranno salvati come 'senza_numero_XX'. Continuare?"):
            return
        ext = ".png" if self.v_fmt.get() == "PNG" else ".jpg"
        try:
            spec = self.output_spec(todo)
        except (ValueError, tk.TclError) as ex:
            messagebox.showerror("Uscita", str(ex))
            return
        self.write(f"Uscita: {spec['size'][0]}x{spec['size'][1]} px per TUTTI i frame "
                   f"({self.v_omode.get()}, {self.v_ofit.get()}).")
        if spec["warn"]:
            self.write(spec["warn"])
        # nomi dei file e protezione di quelli gia' presenti in cartella (nessuna sovrascrittura non voluta)
        pad, check = int(self.v_pad.get()), bool(spec.get("check"))
        ordered = output_order(todo)
        policy = "rinomina"
        while True:
            plan = plan_output_names(ordered, pad, ext, out, check, policy)
            clash = [e for e in plan if e["existed"]]
            if not clash:
                break
            choice = self._ask_collision_policy(len(clash), [e["orig"] + ext for e in clash][:6], out)
            if choice is None:
                self.write("Salvataggio annullato: nella cartella ci sono gia' file con gli stessi nomi.")
                return
            if choice == "cartella":
                d = filedialog.askdirectory(title="Nuova cartella di uscita", initialdir=str(out))
                if d:
                    out = Path(d)
                    self.v_out.set(d)
                continue
            policy = choice
            plan = plan_output_names(ordered, pad, ext, out, check, policy)
            self.write(f"File gia' presenti: {len(clash)}. Scelta: {policy.upper()}.")
            break
        self.cancel_ev.clear()
        self._t_job = time.perf_counter()
        self.busy(True)
        self.lcd.set("busy", "SALVO", "", 0, len(plan))
        threading.Thread(target=self._save_worker,
                         args=(todo, out, ext, pad, self.geom(), spec, plan),
                         daemon=True).start()
        self.root.after(150, self._poll)

    def _ask_collision_policy(self, n, examples, out):
        """Dialogo modale: cosa fare dei file gia' presenti. Ritorna 'rinomina' | 'salta' | 'sovrascrivi' |
        'cartella' | None (annulla)."""
        res = {"v": None}
        top = tk.Toplevel(self.root)
        top.title("File gia' presenti")
        top.configure(bg=BODY)
        top.transient(self.root.winfo_toplevel())
        txt = (f"Nella cartella ci sono gia' {n} file con lo stesso nome:\n   " + "\n   ".join(examples)
               + ("\n   ..." if n > len(examples) else "") + f"\n\n{out}\n\nCosa faccio?")
        tk.Label(top, text=txt, bg=BODY, fg=INK, font=(FONT, 10), justify="left").pack(padx=18, pady=(14, 10))
        row = tk.Frame(top, bg=BODY)
        row.pack(padx=12, pady=(0, 14))

        def pick(v):
            res["v"] = v
            top.destroy()
        for text, v, kind, w in (("RINOMINA (_2, _3...)", "rinomina", "green", 190), ("SALTA", "salta", "cream", 90),
                                 ("SOVRASCRIVI", "sovrascrivi", "red", 140), ("ALTRA CARTELLA", "cartella", "blue", 160),
                                 ("ANNULLA", None, "cream", 100)):
            Key(row, text, lambda v=v: pick(v), kind=kind, width=w, height=32).pack(side="left", padx=4)
        top.protocol("WM_DELETE_WINDOW", lambda: pick(None))
        top.grab_set()
        top.wait_window()
        return res["v"]

    def output_spec(self, frames=None):
        custom = (int(float(self.v_ow.get())), int(float(self.v_oh.get())))
        use = frames if frames is not None else [f for f in self.frames if f["include"]]
        return compute_spec(self.v_omode.get(), self.v_ofit.get(), custom, use, self.geom(),
                            _PDF_ENGINE, self.v_tpl.get().strip(),
                            use_rect=bool(self.v_orect.get()), overscan=float(self.v_oover.get()),
                            pdf_first=bool(self.v_opdf.get()), check=bool(self.v_ochk.get()),
                            paper=bool(self.v_opaper.get()))

    def update_osz(self):
        try:
            sp = self.output_spec()
            self.v_osz.set(f"-> {sp['size'][0]} x {sp['size'][1]} px")
        except Exception:
            self.v_osz.set("-> ?")

    # ------------------------------------------------------------- tema
    _COLOR_OPTS = ("background", "foreground", "activebackground", "activeforeground", "selectcolor",
                   "selectbackground", "selectforeground", "highlightbackground", "highlightcolor",
                   "insertbackground", "buttonbackground", "troughcolor", "disabledforeground")

    def _tree_tags(self):
        self.tree.tag_configure("warn", background=WARN_BG, foreground=WARN_FG)
        self.tree.tag_configure("bad", background=BAD_BG, foreground=BAD_FG)
        self.tree.tag_configure("off", foreground=OFF_FG)

    @staticmethod
    def _theme_map(name):
        """Mappa colore-del-tema-chiaro -> colore del tema scelto. L'interfaccia nasce sempre chiara e ogni
        widget ricorda il proprio colore chiaro originale: cosi' si puo' passare da un tema all'altro e tornare
        indietro senza ambiguita' (nel tema scuro piu' ruoli condividono lo stesso esadecimale)."""
        light, target, m = THEMES["chiaro"], THEMES[name], {}
        for k in light:
            m[light[k].upper()] = target[k]
        for k in THEME_PRIMARY:
            m[light[k].upper()] = target[k]
        return m

    def _recolor(self, widget, lmap):
        store = widget.__dict__.setdefault("_orig_colors", {})
        for opt in self._COLOR_OPTS:
            try:
                cur = str(widget.cget(opt))
            except tk.TclError:
                continue                      # opzione assente (es. widget ttk): ignorata
            store.setdefault(opt, cur)
            new = lmap.get(store[opt].upper())
            if new and new.upper() != cur.upper():
                try:
                    widget.configure(**{opt: new})
                except tk.TclError:
                    pass
        if isinstance(widget, Led):
            if not hasattr(widget, "_orig_color"):
                widget._orig_color = widget.color
            widget.color = lmap.get(widget._orig_color.upper(), widget._orig_color)
            widget.redraw()
        elif isinstance(widget, Key):
            widget._draw()
        for child in widget.winfo_children():
            self._recolor(child, lmap)

    def apply_theme(self, name):
        if name not in THEMES:
            name = "chiaro"
        set_theme_globals(name)
        self._style()
        self._tree_tags()
        self._recolor(self.root.winfo_toplevel(), self._theme_map(name))
        for lab in self._sec_labels:          # testo delle sezioni leggibile; il colore resta sulla riga
            lab.configure(fg=INK if name == "scuro" else lab._orig_colors.get("foreground", INK))
        btn = getattr(self, "btn_theme", None)
        if btn is not None:
            btn.text = "TEMA CHIARO" if name == "scuro" else "TEMA SCURO"
            btn._draw()
        self.on_select()

    def toggle_theme(self):
        ed = getattr(self, "_editor", None)
        if ed is not None and ed.win.winfo_exists():
            self.write("Chiudi l'EDITOR PAGINA prima di cambiare tema.")
            return
        new = "chiaro" if CURRENT_THEME == "scuro" else "scuro"
        self.apply_theme(new)
        prefs = load_prefs()
        prefs["theme"] = new
        save_prefs(prefs)
        self.write(f"Tema {new} attivo (salvato nelle preferenze).")

    # ------------------------------------------------ operazioni in background
    def cancel(self):
        if self.is_busy:
            self.cancel_ev.set()
            self.write("ANNULLA richiesto: mi fermo al prossimo punto sicuro.")

    def _start_job(self, name, func, on_done=None):
        """Una sola operazione pesante alla volta: func(progress) gira in un thread,
        il risultato torna nell'interfaccia tramite on_done(risultato, annullato)."""
        if self.is_busy:
            self.write(f"{name}: c'e' gia' un'operazione in corso. Attendi oppure premi ANNULLA.")
            return False
        self.cancel_ev.clear()
        self.busy(True)
        self.lcd.set("busy", name, "avvio...", 0, 1)
        t0 = time.perf_counter()

        def progress(i, n, text=""):
            self._jq.put(("prog", i, n, text))

        def run():
            try:
                self._jq.put(("ok", func(progress)))
            except Exception as ex:
                file_log(f"ERRORE {name}:\n{traceback.format_exc()}")
                self._jq.put(("err", str(ex)))
        threading.Thread(target=run, daemon=True).start()
        self.root.after(100, lambda: self._job_poll(name, t0, on_done))
        return True

    def _job_poll(self, name, t0, on_done):
        try:
            while True:
                m = self._jq.get_nowait()
                if m[0] == "prog":
                    self.lcd.set("busy", name, m[3], m[1], m[2])
                elif m[0] == "log":
                    self.write(m[1])
                else:
                    dt = time.perf_counter() - t0
                    cancelled = self.cancel_ev.is_set()
                    self.busy(False)
                    if m[0] == "err":
                        self.lcd.set("idle", "ERRORE", m[1][:80], 0, 1)
                        self.write(f"ERRORE {name}: {m[1]}  (dettagli in {LOG_FILE})")
                    else:
                        try:
                            if on_done:
                                on_done(m[1], cancelled)
                        except Exception as ex:
                            file_log(f"ERRORE esito {name}:\n{traceback.format_exc()}")
                            self.write(f"ERRORE {name}: {ex}  (dettagli in {LOG_FILE})")
                    self.write(f"[tempo] {name}: {dt:.1f} s" + (" (annullato)" if cancelled else ""))
                    return
        except queue.Empty:
            pass
        self.root.after(100, lambda: self._job_poll(name, t0, on_done))

    # ------------------------------------------- anteprime (solo grafica)
    def refresh_previews(self, frames=None):
        """Aggiorna SOLO le miniature, in background. Non rilancia il riconoscimento PDF e non
        cambia numeri, inclusioni, stati o decisioni manuali. `frames` limita l'aggiornamento."""
        if not self.frames:
            return
        targets = list(frames) if frames is not None else list(self.frames)
        if not targets:
            return
        geom = self.geom()
        try:
            spec = self.output_spec([f for f in self.frames if f["include"]] or self.frames)
        except (ValueError, tk.TclError):
            spec = None

        def job(progress):
            return self._previews_job(targets, geom, spec, progress)
        self._start_job("ANTEPRIME", job, self._after_previews)

    def _previews_job(self, targets, geom, spec, progress):
        gk = _geom_key(geom)
        by_scan = {}
        for f in targets:
            by_scan.setdefault(f["scan"], []).append(f)
        out, n = [], len(targets)
        for path, fs in by_scan.items():
            if self.cancel_ev.is_set():
                break
            image = self.scan_cache.get(path)
            if image is None:
                continue
            for f in fs:
                if self.cancel_ev.is_set():
                    break
                pair = f["pair"]
                if "pdf_H" not in pair and pair.get("_rk") != gk:      # solo se la geometria e' cambiata
                    refine_rect(image, pair, f["D"], geom)
                    pair["_rk"] = gk
                if spec is not None:      # anteprima = esattamente cio' che verra' salvato
                    th = render_output(image, f, spec, geom, _PDF_ENGINE, cv2.INTER_LINEAR,
                                       360. / spec["size"][0], quality=False)
                else:
                    width = pair["pdf_size"][0] if "pdf_H" in pair else 2 * geom["half_w"]
                    th = crop_frame(image, pair, f["D"], geom, 360. / width, cv2.INTER_LINEAR)
                out.append((f, th))
                progress(len(out), n, Path(path).name)
        return out

    def _after_previews(self, res, cancelled):
        for f, th in res:
            f["thumb"] = th
        self.on_select()
        self.update_osz()
        self.lcd.set("idle", "PRONTO", f"{len(res)} anteprime aggiornate", len(res), len(res))
        self.write(f"Anteprime aggiornate: {len(res)} (numeri, riconoscimenti e decisioni manuali invariati).")

    # ------------------------------------- riconoscimento PDF (esplicito)
    def retry_recognition(self):
        """Nuova verifica dei frame sul PDF, avviata SOLO dall'utente. I numeri manuali restano intatti."""
        if self.is_busy:
            self.write("Operazione in corso: attendi o premi ANNULLA.")
            return
        if _PDF_ENGINE is None:
            messagebox.showinfo("PDF", "Prepara prima il PDF (PREPARA PDF).")
            return
        if not self.frames:
            messagebox.showinfo("PDF", "Nessun frame: esegui prima ANALIZZA.")
            return
        frames, geom = list(self.frames), self.geom()
        before = {id(f): _frame_sig(f) for f in frames}
        protected = sum(1 for f in frames if f.get("num_manual"))
        for f in frames:                      # senza questo, resolve() salterebbe i frame gia' provati
            f.pop("pdf_gen", None)
            f.pop("pdf_seq", None)

        def job(progress):
            progress(0, 1, "confronto con il PDF")
            _PDF_ENGINE.resolve(frames, geom, log=lambda m: self._jq.put(("log", m)),
                                cancel=self.cancel_ev.is_set)

        def done(_res, cancelled):
            infer_numbers(self.frames, self.direction())
            changed = [f for f in frames if before.get(id(f)) != _frame_sig(f)]
            self.refresh()
            self.lcd.set("idle", "ANNULLATO" if cancelled else "PRONTO",
                         f"{len(changed)} frame cambiati", 1, 1)
            self.write(f"Riconoscimento PDF {'ANNULLATO (parziale)' if cancelled else 'ripetuto'}: "
                       f"{len(changed)} frame cambiati, {protected} numeri manuali protetti.")
            self.autosave()
            if changed:
                self.refresh_previews(changed)      # solo le anteprime interessate
        self._start_job("RICONOSCIMENTO PDF", job, done)

    # ------------------------------------------------------------ progetto
    def collect_decisions(self):
        """Decisioni dell'utente: numeri manuali, esclusioni, frame aggiunti a mano."""
        out = []
        for f in self.frames:
            if not (f.get("num_manual") or f.get("manual") or not f["include"]):
                continue
            p = f["pair"]
            d = dict(scan=f["scan"], top=[float(p["top"][0]), float(p["top"][1])],
                     bot=[float(p["bot"][0]), float(p["bot"][1])], D=float(f["D"]),
                     include=bool(f["include"]), num_manual=bool(f.get("num_manual")),
                     num=(f.get("num") if f.get("num_manual") else None),
                     added=bool(f.get("manual")), anchor=p.get("anchor"),
                     rot=self._rot_of(f["scan"]), prep=PREPROC_VERSION)
            if f.get("manual") and p.get("pdf_num") is not None and "pdf_H" in p and _PDF_ENGINE is not None:
                gk = getattr(_PDF_ENGINE, "geom_key", None)
                d.update(match={k: _jsonable(v) for k, v in p.items() if k.startswith("pdf_")},
                         pdf_fp=_jsonable(getattr(_PDF_ENGINE, "fingerprint", None)),
                         scan_fp=file_fingerprint(f["scan"]),
                         geom_key=_jsonable(gk) if gk is not None else None)
            out.append(d)
        return out

    def collect_project(self):
        settings = {}
        for name in PROJECT_VARS:
            var = getattr(self, name, None)
            if var is not None:
                try:
                    settings[name] = var.get()
                except tk.TclError:
                    pass
        return dict(version=PROJECT_VERSION, saved=datetime.datetime.now().isoformat(timespec="seconds"),
                    settings=settings, scans=list(self.scans),
                    decisions=merge_decisions(self.collect_decisions(), self._pending),
                    erased=list(self._erased))

    def _write_project(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.collect_project(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)                      # scrittura atomica: un crash non lascia file a meta'

    def autosave(self):
        if not (self.scans or self.frames or self._pending or self._erased):
            return                            # non sovrascrivere una sessione precedente con una vuota
        try:
            self._write_project(AUTOSAVE_FILE)
        except OSError as ex:
            file_log(f"autosave fallito: {ex}")

    def save_project(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Progetto scontrini", "*.json")])
        if not p:
            return
        try:
            self._write_project(p)
            self.write(f"Progetto salvato: {p} ({len(self.collect_decisions())} decisioni manuali).")
        except OSError as ex:
            messagebox.showerror("Progetto", f"Salvataggio non riuscito: {ex}")

    @staticmethod
    def _read_project(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != PROJECT_VERSION:
            raise ValueError("File progetto non riconosciuto")
        return data

    def _apply_project(self, data):
        for name, val in data.get("settings", {}).items():
            var = getattr(self, name, None) if name in PROJECT_VARS else None
            if var is not None:
                try:
                    var.set(val)
                except (tk.TclError, ValueError):
                    pass
        self.clear_scans()
        for p in data.get("scans", []):
            if p not in self.scans:
                self.scans.append(p)
                self.lb.insert("end", p)
        self.frames, self.meta = [], {}
        self.tree.delete(*self.tree.get_children())
        self._pending = list(data.get("decisions", []))
        self._erased = list(data.get("erased", []))
        self.update_osz()
        gone = [p for p in self.scans if not Path(p).is_file()]
        if gone:
            self.write(f"ATTENZIONE: {len(gone)} scansioni del progetto non esistono piu' (es. {Path(gone[0]).name}).")

    def load_project(self):
        if self.is_busy:
            return
        p = filedialog.askopenfilename(filetypes=[("Progetto scontrini", "*.json"), ("Tutti", "*.*")])
        if not p:
            return
        try:
            data = self._read_project(p)
        except (OSError, ValueError) as ex:
            messagebox.showerror("Progetto", f"Impossibile aprire il progetto: {ex}")
            return
        self._apply_project(data)
        self.write(f"Progetto caricato: {len(self.scans)} scansioni, {len(self._pending)} decisioni manuali, "
                   f"{len(self._erased)} frame cancellati con la gomma. "
                   "Prepara il PDF (se serve) e premi ANALIZZA: le correzioni verranno riapplicate.")

    def restore_autosave(self):
        try:
            data = self._read_project(AUTOSAVE_FILE) if AUTOSAVE_FILE.is_file() else None
        except (OSError, ValueError):
            return
        if not data or not (data.get("scans") or data.get("decisions")):
            return
        if messagebox.askyesno("Sessione precedente",
                               f"Trovata l'ultima sessione: {len(data.get('scans', []))} scansioni, "
                               f"{len(data.get('decisions', []))} correzioni manuali.\nRipristinarla?"):
            self._apply_project(data)
            self.write("Sessione ripristinata: prepara il PDF (se serve) e premi ANALIZZA per riapplicare le correzioni.")

    # --------------------------------------------- gomma persistente (tombstone)
    def _rot_of(self, scan):
        """True se nei frame attuali questa scansione e' stata ruotata di 180 gradi."""
        try:
            return str(Path(scan).resolve()) in self._frames_rot
        except OSError:
            return False

    def add_tombstone(self, f):
        """Ricorda che l'utente ha cancellato questo frame (falso positivo): la rianalisi non lo ripropone.
        I frame aggiunti a mano non servono: non sono nell'analisi automatica."""
        if f.get("manual"):
            return
        p = f["pair"]
        if any(t["scan"] == f["scan"] and _near(p, t["top"], t["bot"]) for t in self._erased):
            return
        self._erased.append(dict(scan=f["scan"], top=[float(p["top"][0]), float(p["top"][1])],
                                 bot=[float(p["bot"][0]), float(p["bot"][1])], D=float(f["D"]),
                                 rot=self._rot_of(f["scan"])))

    def remove_tombstone(self, f):
        p = f["pair"]
        self._erased = [t for t in self._erased if not (t["scan"] == f["scan"] and _near(p, t["top"], t["bot"]))]

    def _apply_erased(self):
        if not self._erased:
            return
        kept, pend, n = apply_tombstones(self.frames, self._pending, self._erased)
        if n:
            self.frames[:] = kept
            self.write(f"Frame cancellati con la gomma e NON riproposti: {n} (RIPRISTINA CANCELLATI per riaverli).")
        self._pending = pend

    def restore_erased(self):
        if self.is_busy:
            return
        if not self._erased:
            messagebox.showinfo("Gomma", "Nessun frame cancellato da ripristinare.")
            return
        if messagebox.askyesno("Ripristina cancellati",
                               f"{len(self._erased)} frame cancellati con la gomma torneranno a essere proposti "
                               "alla prossima ANALISI. Procedere?"):
            n = len(self._erased)
            self._erased = []
            self.autosave()
            self.write(f"Ripristinati {n} frame cancellati: riappariranno alla prossima ANALISI.")

    def _apply_pending_matches(self):
        """Dopo una nuova analisi, riapplica le decisioni manuali ai frame ritrovati (stessa scansione, stessa
        posizione dei crocini). Le decisioni di scansioni NON analizzate in questa sessione restano conservate
        (e nel salvataggio automatico): non sono 'perse', semplicemente non c'e' nulla a cui applicarle."""
        pend, self._pending = self._pending, []
        if not pend:
            return
        analyzed = set(self.meta)                   # scansioni analizzate in questa sessione
        ok = lost = 0
        keep = []
        for d in pend:
            top, bot = d["top"], d["bot"]
            best, bd = None, 12.0
            for f in self.frames:
                if f["scan"] != d["scan"]:
                    continue
                p = f["pair"]
                dist = max(math.hypot(p["top"][0] - top[0], p["top"][1] - top[1]),
                           math.hypot(p["bot"][0] - bot[0], p["bot"][1] - bot[1]))
                if dist < bd:
                    best, bd = f, dist
            if best is None:
                # nessuna decisione viene eliminata in silenzio: resta in _pending (e nel salvataggio automatico)
                # finche' non ritrova il suo frame. Il motivo serve a capire perche' non e' stata applicata.
                if d["scan"] in analyzed:
                    lost += 1                       # scansione analizzata ma il frame non e' stato rilevato
                    d = dict(d, motivo="frame non rilevato nell'ultima analisi")
                else:
                    d = dict(d, motivo="scansione non analizzata in questa sessione")
                keep.append(d)
                continue
            best["include"] = bool(d.get("include", True))
            if d.get("num_manual") and d.get("num") is not None:
                best["num"], best["status"] = int(d["num"]), "manuale"
                best["locked"], best["num_manual"] = True, True
            ok += 1
        self._pending = keep
        msg = f"Correzioni manuali riapplicate: {ok}"
        if lost:
            msg += (f". NON ritrovate: {lost} (frame non rilevati in scansioni analizzate: CONSERVATE, "
                    "si riapplicheranno se il frame viene rilevato o aggiunto di nuovo; controlla)")
        other = len(keep) - lost
        if other > 0:
            n_scans = len({d["scan"] for d in keep if d.get("motivo", "").startswith("scansione non")})
            msg += f". Conservate {other} correzioni di {n_scans} scansioni non analizzate in questa sessione"
        self.write(msg + ".")

    def _save_worker(self, todo, out, ext, pad, geom, spec, plan=None):
        """Scrive i file e riporta un ESITO distinto: completed / cancelled / partial / failed, con file
        previsti, scritti, saltati, falliti e non esportati. Un errore su un file non ferma gli altri; un errore
        del solo controllo di sovrapposizione non annulla il file principale gia' scritto."""
        if plan is None:
            plan = plan_output_names(output_order(todo), pad, ext, out, bool(spec.get("check")), "rinomina")
        out = Path(out)
        res = dict(status="completed", planned=len(plan), written=0, skipped=0, failed=[], check_failed=[],
                   not_exported=[], out=str(out), cancelled=False, fatal=None)
        params = [cv2.IMWRITE_JPEG_QUALITY, 100] if ext == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 9]
        checks, handled = [], set()

        def progress(name):
            self.q.put(("prog", len(handled), len(plan), "SALVO", name))
        try:
            by_scan = {}
            for e in plan:
                by_scan.setdefault(e["f"]["scan"], []).append(e)
            for path, entries in by_scan.items():
                if self.cancel_ev.is_set():
                    break
                try:
                    img = imread_unicode(path)
                    if img is None:
                        raise RuntimeError("immagine non decodificabile")
                except Exception as ex:
                    for e in entries:
                        res["failed"].append((e["name"] + ext, f"scansione non leggibile ({Path(path).name}): {ex}"))
                        handled.add(id(e))
                    progress(Path(path).name)
                    continue
                for e in entries:
                    if self.cancel_ev.is_set():
                        break
                    f, name = e["f"], e["name"]
                    if e["action"] == "salta":
                        res["skipped"] += 1
                    else:
                        try:
                            crop = render_output(img, f, spec, geom, _PDF_ENGINE)
                            imwrite_atomic(out / (name + ext), crop, params)
                        except Exception as ex:
                            file_log(f"ERRORE file {name}{ext}:\n" + traceback.format_exc())
                            res["failed"].append((name + ext, str(ex) or type(ex).__name__))
                        else:
                            res["written"] += 1          # il file principale c'e': conta anche se il controllo fallisce
                            if spec.get("check") and spec.get("pdf_first", True) and "pdf_H" in f["pair"]:
                                try:
                                    ana, cc = check_overlay(crop, f, spec, _PDF_ENGINE)
                                    if ana is not None:
                                        (out / "_controllo").mkdir(exist_ok=True)
                                        imwrite_atomic(out / "_controllo" / (name + ".png"), ana)
                                        checks.append((name, cc))
                                except Exception as ex:
                                    file_log(f"ERRORE controllo {name}:\n" + traceback.format_exc())
                                    res["check_failed"].append((name, str(ex) or type(ex).__name__))
                    handled.add(id(e))
                    progress(name + ext)
                del img
        except Exception as ex:
            file_log("ERRORE salvataggio:\n" + traceback.format_exc())
            res["fatal"] = str(ex) or type(ex).__name__
            self.q.put(("log", f"ERRORE salvataggio: {res['fatal']}"))
        res["not_exported"] = [e["name"] + ext for e in plan if id(e) not in handled]
        res["cancelled"] = self.cancel_ev.is_set()
        if res["cancelled"] and res["not_exported"]:
            res["status"] = "cancelled"
        elif res["written"] == 0 and (res["failed"] or res["fatal"]):
            res["status"] = "failed"
        elif res["failed"] or res["fatal"] or res["not_exported"]:
            res["status"] = "partial"
        if checks:
            vals = np.array([c for _, c in checks])
            low = [n for n, c in checks if c < 0.25 or c < float(np.median(vals)) - 0.2]
            self.q.put(("log", f"Controllo sovrapposizione: correlazione mediana {np.median(vals):.2f} "
                               f"su {len(checks)} frame. Immagini in _controllo (grigio = allineato, "
                               f"frange rosse/ciano = disallineato)."))
            if low:
                self.q.put(("log", "DA CONTROLLARE: " + ", ".join(low[:30])))
        self.q.put(("saved", res))



# ===================== PDF REFERENCE EDITION v6 =====================
# Novita' rispetto alla v5:
#  * l'allineamento col PDF e' sempre una SIMILITUDINE (rotazione + scala + traslazione):
#    i riquadri restano rettangoli perfetti, niente piu' prospettive "storte";
#  * riconoscimento pagina a due stadi (voto SIFT globale -> allineamento locale sulla
#    sola pagina vincente) con RANSAC a 4 gradi di liberta' e raffinamento ECC;
#  * calibrazione automatica del layout "pagina PDF -> crocini" dai match sicuri, poi
#    ricerca GUIDATA per i frame difficili (confronto a bassa risoluzione con tutte le pagine);
#  * propagazione per sequenza: i frame di una stessa colonna hanno pagine consecutive,
#    quindi i frame non riconosciuti vengono verificati sulla pagina attesa;
#  * l'ancoraggio manuale sui crocini non viene piu' "sovrascritto" da un'omografia:
#    il PDF e' usato solo se coerente con i crocini cliccati.
from collections import Counter

_PDF_ENGINE = None
_ROTATE_SCANS = set()
_read_base = imread_unicode
_crop_base = crop_frame
_footprint_base = footprint_poly
_make_base = make_frame
_infer_base = infer_numbers


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    image = _read_base(path, flags)
    if image is not None and str(Path(path).resolve()) in _ROTATE_SCANS:
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image


# ---------------------------------------------------------------- utilita'
def _as3(A):
    A = np.asarray(A, np.float64)
    return A if A.shape == (3, 3) else np.vstack([A, [0.0, 0.0, 1.0]])


def _Q(f):
    """Matrice 3x3 che porta le coordinate a piena risoluzione a quelle di un'immagine
    ridimensionata del fattore f (convenzione cv2.resize: centri dei pixel)."""
    o = 0.5 * f - 0.5
    return np.array([[f, 0, o], [0, f, o], [0, 0, 1.0]])


def _T(x, y):
    return np.array([[1.0, 0, x], [0, 1.0, y], [0, 0, 1.0]])


def _sim_rs(M):
    return math.hypot(M[0, 0], M[1, 0]), math.atan2(M[1, 0], M[0, 0])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sim_make(s, th, tx, ty):
    c, sn = s * math.cos(th), s * math.sin(th)
    return np.array([[c, -sn, tx], [sn, c, ty], [0, 0, 1.0]])


def _no_shear(M):
    """Affine -> rotazione + scala x/y (niente shear): il rettangolo resta un rettangolo."""
    M = _as3(M)
    th = math.atan2(M[1, 0], M[0, 0])
    sx = math.hypot(M[0, 0], M[1, 0])
    sy = M[0, 1] * (-math.sin(th)) + M[1, 1] * math.cos(th)
    c, sn = math.cos(th), math.sin(th)
    return np.array([[sx * c, -sy * sn, M[0, 2]], [sx * sn, sy * c, M[1, 2]], [0, 0, 1.0]])


def _dog(img, lo=1.0, hi=9.0):
    f = img.astype(np.float32)
    return cv2.GaussianBlur(f, (0, 0), lo) - cv2.GaussianBlur(f, (0, 0), hi)


def _ncc(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-6 else 0.0


def _cross_matrix(pair, D, geom):
    """canonico (ritaglio a ref_d) -> scansione, SOLO dai crocini (similitudine esatta)."""
    tmp = dict(pair)
    tmp["corr"] = None
    A, _ = affine_for_pair(tmp, D, geom, 1.0)
    return _as3(A)


class PDFReferences:
    FULL_LONG = 1500.0     # lato lungo della pagina "ufficiale" (= dimensione dei frame in uscita)
    SIFT_LONG = 1000.0     # risoluzione di lavoro delle feature
    MID_LONG = 520.0       # risoluzione per ECC
    THUMB_LONG = 120.0     # risoluzione per il confronto globale
    # soglie (tutte in un posto: se serve tarare, e' qui)
    MIN_INLIERS = 12       # inlier minimi (similitudine) per fidarsi del solo SIFT
    SOFT_CC = 0.22         # correlazione minima dopo l'allineamento fine
    GUIDED_NCC = 0.30      # correlazione minima nella ricerca guidata
    GUIDED_MARGIN = 0.06   # distacco minimo dalla seconda pagina nella ricerca guidata
    STRONG_INL = 14
    STRONG_CC = 0.45

    def __init__(self, path=None, progress=None, gray_pages=None, report=None, use_cache=True):
        progress = progress or (lambda m: None)
        report = report or (lambda m: None)           # messaggi per la console (tempi, cache)
        self.fingerprint = file_fingerprint(path) if path else None
        self._t_sift = 0.0
        t_all = time.perf_counter()
        self.pages, self.mid, self.mid_f, self.thumb, self.thumb_f = [], [], [], [], []
        self.slices = []
        self.prior = None
        self._samples = []
        self.prior_gen = 0
        self.geom_key = None
        self.layout = None
        self._feats, self._owners, self._xy = [], [], []
        self._sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.02)
        cache_file = None
        if gray_pages is None and path and use_cache:
            try:
                cache_file = self.cache_path(path)
            except OSError:
                cache_file = None
            if cache_file is not None and self._load_cache(cache_file, progress, report):
                return
        t_render = 0.0
        if gray_pages is not None:
            for i, g in enumerate(gray_pages):
                self._add_page(g)
                progress(f"PDF: pagina {i + 1}/{len(gray_pages)}")
        else:
            try:
                import pymupdf
            except ImportError:
                raise RuntimeError("Installa PyMuPDF: py -m pip install pymupdf")
            with pymupdf.open(path) as doc:
                for i, page in enumerate(doc):
                    t1 = time.perf_counter()
                    factor = self.FULL_LONG / max(page.rect.width, page.rect.height)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(factor, factor),
                                          colorspace=pymupdf.csGRAY, alpha=False)
                    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width].copy()
                    t_render += time.perf_counter() - t1
                    self._add_page(gray)
                    progress(f"PDF: pagina {i + 1}/{len(doc)}")
        if not self._feats:
            raise RuntimeError("Nessun dettaglio riconoscibile nel PDF")
        self.des = np.vstack(self._feats).astype(np.float32)
        self.owners = np.asarray(self._owners, np.int32)
        self.xy = np.asarray(self._xy, np.float32)
        self.slices = []
        for i in range(len(self.pages)):
            idx = np.where(self.owners == i)[0]
            self.slices.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0))
        self._feats = self._owners = self._xy = None
        t1 = time.perf_counter()
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.matcher.add([self.des])
        self.matcher.train()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        t_idx = time.perf_counter() - t1
        total = time.perf_counter() - t_all
        other = max(0.0, total - t_render - self._t_sift - t_idx)
        self.timing = dict(total=total, render=t_render, sift=self._t_sift, index=t_idx, other=other)
        report(f"PDF: {len(self.pages)} pagine costruite in {total:.0f} s  (lettura pagine {t_render:.0f} s, "
               f"dettagli SIFT {self._t_sift:.0f} s, indice di ricerca {t_idx:.0f} s, altro {other:.0f} s)")
        if cache_file is not None:
            self._save_cache(cache_file, report)

    # ---------------------------------------------------------------- cache su disco
    @classmethod
    def cache_path(cls, path):
        """File di cache legato al PDF (percorso, dimensione, data) e ai parametri che ne cambiano il contenuto."""
        st = Path(path).stat()
        params = (CACHE_VERSION, str(Path(path).resolve()), st.st_size, st.st_mtime_ns, cls.FULL_LONG,
                  cls.SIFT_LONG, cls.MID_LONG, cls.THUMB_LONG, 1200, 0.02, cv2.__version__)
        return CACHE_DIR / ("refs_" + hashlib.sha1(repr(params).encode()).hexdigest()[:16] + ".npz")

    def _save_cache(self, file, report):
        tmp = Path(str(file) + ".tmp")
        try:
            t0 = time.perf_counter()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            arrays = dict(meta=np.frombuffer(json.dumps(dict(version=CACHE_VERSION, n=len(self.pages))).encode(),
                                             np.uint8),
                          pages=np.asarray(self.pages, np.int32), mid_f=np.asarray(self.mid_f, np.float64),
                          thumb_f=np.asarray(self.thumb_f, np.float64), des=self.des, owners=self.owners,
                          xy=self.xy, page0=self._page0)
            for i, a in enumerate(self.mid):
                arrays[f"m{i}"] = a
            for i, a in enumerate(self.thumb):
                arrays[f"t{i}"] = a
            with open(tmp, "wb") as fh:
                np.savez(fh, **arrays)               # non compresso: si rilegge molto piu' in fretta
            tmp.replace(file)
            for old in sorted(CACHE_DIR.glob("refs_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)[CACHE_KEEP:]:
                old.unlink()
            report(f"PDF: cache salvata ({Path(file).stat().st_size / 1e6:.0f} MB in {time.perf_counter() - t0:.1f} s): "
                   "la prossima preparazione dello stesso PDF sara' molto piu' rapida.")
        except (OSError, ValueError, TypeError) as ex:
            report(f"PDF: cache non salvata ({ex}). Nessun problema: funziona comunque, solo piu' lento.")
            try:
                tmp.unlink()
            except OSError:
                pass

    def _load_cache(self, file, progress, report):
        file = Path(file)
        if not file.is_file():
            return False
        t0 = time.perf_counter()
        try:
            progress("PDF: carico i riferimenti dalla cache...")
            with np.load(file, allow_pickle=False) as z:
                meta = json.loads(bytes(z["meta"]).decode())
                if meta.get("version") != CACHE_VERSION:
                    raise ValueError("versione della cache diversa")
                n = int(meta["n"])
                pages = [tuple(int(v) for v in r) for r in z["pages"]]
                mid = [z[f"m{i}"] for i in range(n)]
                thumb = [z[f"t{i}"] for i in range(n)]
                mid_f = [float(v) for v in z["mid_f"]]
                thumb_f = [float(v) for v in z["thumb_f"]]
                des, owners, xy, page0 = z["des"], z["owners"], z["xy"], z["page0"]
            if not (len(pages) == len(mid) == len(thumb) == len(mid_f) == len(thumb_f) == n and n > 0
                    and des.ndim == 2 and des.shape[1] == 128 and len(des) == len(owners) == len(xy)):
                raise ValueError("contenuto incoerente")
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as ex:
            report(f"PDF: cache non utilizzabile ({ex}): rifaccio la preparazione completa.")
            try:
                file.unlink()
            except OSError:
                pass
            return False
        t_read = time.perf_counter() - t0
        self.pages, self.mid, self.mid_f, self.thumb, self.thumb_f = pages, mid, mid_f, thumb, thumb_f
        self.des = np.ascontiguousarray(des, np.float32)
        self.owners, self.xy, self._page0 = owners, xy, page0
        try:
            self.layout = layout_from_gray(page0)
        except Exception:
            self.layout = None
        self.slices = []
        for i in range(len(self.pages)):
            idx = np.where(self.owners == i)[0]
            self.slices.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0))
        self._feats = self._owners = self._xy = None
        t1 = time.perf_counter()
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.matcher.add([self.des])
        self.matcher.train()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        t_idx = time.perf_counter() - t1
        self.timing = dict(total=time.perf_counter() - t0, read=t_read, index=t_idx, cached=True)
        report(f"PDF: {len(self.pages)} pagine caricate dalla cache in {self.timing['total']:.1f} s "
               f"(lettura file {t_read:.1f} s, indice di ricerca {t_idx:.1f} s)")
        return True

    # ------------------------------------------------------------- pagine
    @staticmethod
    def _root(des):
        des = des.astype(np.float32)
        des /= (des.sum(axis=1, keepdims=True) + 1e-7)
        return np.sqrt(des)

    def _add_page(self, gray):
        i = len(self.pages)
        h, w = gray.shape
        self.pages.append((w, h))
        if i == 0:
            self._page0 = gray
            try:
                self.layout = layout_from_gray(gray)
            except Exception:
                self.layout = None
        L = float(max(w, h))
        f = self.SIFT_LONG / L
        g = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), 0.8)
        t_s = time.perf_counter()
        kp, des = self._sift.detectAndCompute(g, None)
        self._t_sift += time.perf_counter() - t_s
        if des is not None and len(des) >= 8:
            self._feats.append(self._root(des))
            self._owners.extend([i] * len(des))
            # coordinate riportate nello spazio pagina (centri dei pixel)
            self._xy.extend([((k.pt[0] + .5) / f - .5, (k.pt[1] + .5) / f - .5) for k in kp])
        m = self.MID_LONG / L
        self.mid.append(cv2.resize(gray, None, fx=m, fy=m, interpolation=cv2.INTER_AREA))
        self.mid_f.append(self.mid[-1].shape[1] / w)
        t = self.THUMB_LONG / L
        th = cv2.resize(gray, None, fx=t, fy=t, interpolation=cv2.INTER_AREA)
        self.thumb.append(cv2.GaussianBlur(th.astype(np.float32), (0, 0), 1.0))
        self.thumb_f.append(th.shape[1] / w)

    # ------------------------------------------------------------ regione
    def _region(self, image, pair, D):
        tx, ty = pair["top"]
        bx, by = pair["bot"]
        cx, cy = (tx + bx) / 2, (ty + by) / 2
        H, W = image.shape[:2]
        x0, x1 = max(0, int(cx - 1.1 * D)), min(W, int(cx + 1.1 * D))
        y0, y1 = max(0, int(cy - .85 * D)), min(H, int(cy + .85 * D))
        if x1 - x0 < 50 or y1 - y0 < 50:
            return None
        reg = image[y0:y1, x0:x1]
        g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
        k = float(np.clip(self.SIFT_LONG / (1.455 * D), 0.2, 2.5))
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (0, 0), 0.8)
        S = _Q(k) @ _T(-x0, -y0)                       # scansione -> regione
        return dict(g=g, k=k, S=S, c=np.array([(cx - x0 + .5) * k - .5, (cy - y0 + .5) * k - .5]),
                    D=D * k)

    # ------------------------------------------------------------ stadio 1
    def _blind(self, reg):
        kp, des = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.02).detectAndCompute(reg["g"], None)
        if des is None or len(des) < 12:
            return []
        des = self._root(des)
        pts = np.float32([k.pt for k in kp])
        knn = self.matcher.knnMatch(des, k=2)
        votes = Counter()
        for row in knn:
            if len(row) == 2 and row[0].distance < .8 * row[1].distance:
                votes[int(self.owners[row[0].trainIdx])] += 1
        out = []
        for page, n in votes.most_common(6):
            if n < 5:
                break
            r = self._align_page(page, des, pts)
            if r:
                out.append(r)
        out.sort(key=lambda r: -r["inl"])
        return out

    def _align_page(self, page, des, pts):
        a0, a1 = self.slices[page]
        if a1 - a0 < 8:
            return None
        knn = self.bf.knnMatch(des, self.des[a0:a1], k=2)
        best = {}
        for row in knn:
            if len(row) == 2 and row[0].distance < .85 * row[1].distance:
                m = row[0]
                if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                    best[m.trainIdx] = m
        good = list(best.values())
        if len(good) < 8:
            return None
        src = np.float32([self.xy[a0 + m.trainIdx] for m in good])
        dst = np.float32([pts[m.queryIdx] for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                             maxIters=3000, confidence=.995, refineIters=10)
        if M is None or inl is None:
            return None
        inl = inl.ravel().astype(bool)
        n = int(inl.sum())
        if n < 8:
            return None
        T = _as3(M)
        res = np.linalg.norm(cv2.transform(src[None], M)[0][inl] - dst[inl], axis=1)
        return dict(page=page, T=T, inl=n, n=len(good), err=float(np.median(res)))

    def _plausible(self, reg, T_R, page, exp_R):
        w, h = self.pages[page]
        s, th = _sim_rs(T_R)
        if not (np.isfinite(T_R).all() and s > 1e-3):
            return False
        centre = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), T_R)[0, 0]
        if np.linalg.norm(centre - reg["c"]) > 0.5 * reg["D"]:
            return False
        if exp_R is not None:
            s0, th0 = _sim_rs(exp_R)
            if abs(math.log(s / s0)) > 0.10 or abs(_wrap(th - th0)) > math.radians(3.5):
                return False
        return True

    # ------------------------------------------------------------ stadio 2
    def _guided_scores(self, reg, T_R, pages, pad_frac=0.10):
        """Per ogni pagina: (ncc, T_R spostata) con T_R come stima iniziale."""
        s, _ = _sim_rs(T_R)
        out = []
        blurred = {}
        for i in pages:
            tf = self.thumb_f[i]
            key = round(s / tf, 1)
            if key not in blurred:
                f = s / tf
                blurred[key] = cv2.GaussianBlur(reg["g"], (0, 0), 0.5 * f) if f > 1.3 else reg["g"]
            tmpl = self.thumb[i]
            th_, tw_ = tmpl.shape
            pad = int(pad_frac * tw_) + 1
            B = T_R @ np.linalg.inv(_Q(tf))
            A = B @ _T(-pad, -pad)
            warped = cv2.warpAffine(blurred[key], A[:2], (tw_ + 2 * pad, th_ + 2 * pad),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            r = cv2.matchTemplate(warped, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(np.nan_to_num(r))
            T2 = B @ _T(ml[0] - pad, ml[1] - pad) @ _Q(tf)
            out.append((float(mv), i, T2))
        out.sort(key=lambda t: -t[0])
        return out

    def _refine(self, page, reg_g, T_R):
        """Raffinamento ECC (rotazione+traslazione, scala fissa). Ritorna (T_R, cc)."""
        Qm = _Q(self.mid_f[page])
        tmpl = self.mid[page]
        hm, wm = tmpl.shape
        A = T_R @ np.linalg.inv(Qm)
        s, _ = _sim_rs(A)
        src = cv2.GaussianBlur(reg_g, (0, 0), 0.5 * s) if s > 1.3 else reg_g
        warped = cv2.warpAffine(src, A[:2], (wm, hm), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        t, x = _dog(tmpl), _dog(warped)
        cc0 = _ncc(t, x)
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(t, x, W, cv2.MOTION_EUCLIDEAN,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), None, 5)
        except cv2.error:
            return T_R, cc0
        if math.hypot(W[0, 2], W[1, 2]) > 0.08 * wm or abs(math.atan2(W[1, 0], W[0, 0])) > math.radians(4):
            return T_R, cc0
        x2 = cv2.warpAffine(x, W, (wm, hm), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
        cc1 = _ncc(t, x2)
        if cc1 < cc0:
            return T_R, cc0
        return (A @ _as3(W.astype(np.float64))) @ Qm, cc1

    def _dense(self, reg, page, T_R, cc_ref):
        """Aggancio DENSO sul contenuto della pagina (ECC affine sulla foto), poi proiezione su
        'rotazione + scala x/y' (niente shear: il riquadro resta un rettangolo perfetto).
        Ritorna (T_R migliorata | None, cc)."""
        Qm = _Q(self.mid_f[page])
        tmpl = self.mid[page]
        hm, wm = tmpl.shape
        A = T_R @ np.linalg.inv(Qm)
        s, th0 = _sim_rs(A)
        src = cv2.GaussianBlur(reg["g"], (0, 0), 0.5 * s) if s > 1.3 else reg["g"]
        warped = cv2.warpAffine(src, A[:2], (wm, hm), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        t, x = _dog(tmpl), _dog(warped)
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(t, x, W, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6), None, 5)
        except cv2.error:
            return None, cc_ref
        M = _no_shear((A @ _as3(W.astype(np.float64))))
        sx, sy = math.hypot(M[0, 0], M[1, 0]), math.hypot(M[0, 1], M[1, 1])
        if not (0.92 < sx / s < 1.08 and 0.92 < sy / s < 1.08):
            return None, cc_ref
        if abs(_wrap(math.atan2(M[1, 0], M[0, 0]) - th0)) > math.radians(3):
            return None, cc_ref
        c0 = (A @ np.array([wm / 2, hm / 2, 1.0]))[:2]
        c1 = (M @ np.array([wm / 2, hm / 2, 1.0]))[:2]
        if np.linalg.norm(c1 - c0) > 0.05 * reg["D"]:
            return None, cc_ref
        x2 = cv2.warpAffine(src, M[:2], (wm, hm), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
        cc = _ncc(t, _dog(x2))
        if cc < cc_ref - 0.01:
            return None, cc_ref
        return M @ Qm, cc

    # --------------------------------------------------------- calibrazione
    def _learn(self, Xc, T_scan):
        """Impara M = pagina -> canonico (layout stampato) dai match sicuri."""
        M = np.linalg.inv(Xc) @ T_scan
        s, th = _sim_rs(M)
        self._samples.append((s, th, M[0, 2], M[1, 2]))
        if len(self._samples) < 4:
            return
        a = np.array(self._samples)
        ms = np.median(a[:, 0])
        mth = math.atan2(np.median(np.sin(a[:, 1])), np.median(np.cos(a[:, 1])))
        mtx, mty = np.median(a[:, 2]), np.median(a[:, 3])
        w, h = self.pages[0]
        ok = [abs(math.log(r[0] / ms)) < 0.05 and abs(_wrap(r[1] - mth)) < math.radians(2.0)
              and abs(r[2] - mtx) < 0.04 * w and abs(r[3] - mty) < 0.04 * h for r in a]
        if sum(ok) < 4:
            return
        sel = a[np.array(ok)]
        ms = float(np.median(sel[:, 0]))
        mth = math.atan2(np.median(np.sin(sel[:, 1])), np.median(np.cos(sel[:, 1])))
        new = _sim_make(ms, mth, float(np.median(sel[:, 2])), float(np.median(sel[:, 3])))
        if self.prior is None or np.abs(new - self.prior).max() > 1e-6:
            self.prior = new
            self.prior_gen += 1

    def reset_prior(self):
        self.prior, self._samples = None, []
        self.prior_gen += 1

    # ------------------------------------------------------------ pubblico
    def identify(self, image, pair, D, geom=None, anchor=None, guided_only=False, prior=None):
        """Trova la pagina del PDF e la similitudine pagina->scansione per questa coppia di crocini.
        Ritorna un dict (con 'pdf_H' solo se l'allineamento e' coerente con i crocini) oppure None."""
        geom = geom or DEFAULT_GEOM
        self.geom_key = tuple(round(float(geom[k]), 3) for k in ("ref_d", "half_w", "top", "bottom"))
        reg = self._region(image, pair, D)
        if reg is None:
            return None
        Xc = _cross_matrix(pair, D, geom)
        prior = prior if prior is not None else self.prior
        exp_R = reg["S"] @ Xc @ prior if prior is not None else None
        found = None
        if not guided_only:
            raw = [c for c in self._blind(reg) if c["inl"] >= 8]
            cands = [c for c in raw if self._plausible(reg, c["T"], c["page"], exp_R)]
            if not cands and exp_R is not None:
                # crocini cliccati male / layout diverso: riconosco comunque la pagina,
                # ma l'allineamento restera' sui crocini (vedi _finish)
                cands = [c for c in raw if self._plausible(reg, c["T"], c["page"], None)]
            ev = []
            for c in cands[:3]:
                T_R, cc = self._refine(c["page"], reg["g"], c["T"])
                ev.append((c, T_R, cc))
            if ev:
                ev.sort(key=lambda e: -(e[0]["inl"] + 40 * e[2]))
                c, T_R, cc = ev[0]
                rivals = [e for e in ev[1:] if e[0]["inl"] >= 0.7 * c["inl"] and e[0]["inl"] >= 10]
                if rivals and not (cc - max(e[2] for e in rivals) >= 0.05):
                    c = None
                if c is not None and (c["inl"] >= self.MIN_INLIERS or cc >= self.STRONG_CC) and cc >= self.SOFT_CC:
                    found = dict(page=c["page"], T_R=T_R, cc=cc, inl=c["inl"], err=c["err"], mode="SIFT")
        if found is None and prior is not None:
            sc = self._guided_scores(reg, exp_R, range(len(self.pages)))
            if sc and sc[0][0] >= self.GUIDED_NCC and (len(sc) == 1 or sc[0][0] - sc[1][0] >= self.GUIDED_MARGIN):
                ncc0, page, T2 = sc[0]
                T_R, cc = self._refine(page, reg["g"], T2)
                if cc >= 0.30:
                    found = dict(page=page, T_R=T_R, cc=cc, inl=0, err=0.0, mode="guida")
        if found is None:
            return None
        return self._finish(reg, Xc, found, anchor, prior)

    def _finish(self, reg, Xc, found, anchor, prior):
        T_scan = np.linalg.inv(reg["S"]) @ found["T_R"]
        out = dict(pdf_size=self.pages[found["page"]], pdf_num=found["page"] + 1, pdf_inliers=found["inl"],
                   pdf_ratio=float(np.clip(found["cc"], 0, 1)), pdf_cc=found["cc"], pdf_error=found["err"],
                   pdf_mode=found["mode"])
        aligned = True
        if prior is not None:
            # coerenza con i crocini: stessa posizione/angolo di quello atteso dal layout stampato
            exp = Xc @ prior
            w, h = self.pages[found["page"]]
            c_pdf = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), T_scan)[0, 0]
            c_exp = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), exp)[0, 0]
            D = reg["D"] / reg["k"]
            # il contenuto della foto e' una prova molto piu' forte dei crocini: si scarta solo
            # un allineamento PDF palesemente incoerente (pagina sbagliata / layout diverso)
            tol = {"crocini": 0.08, "rettangolo": 0.20}.get(anchor, 0.25)
            ang = abs(_wrap(_sim_rs(T_scan)[1] - _sim_rs(exp)[1]))
            if np.linalg.norm(c_pdf - c_exp) > tol * D or ang > math.radians(3.0 if anchor == "crocini" else 6.0):
                aligned = False
            out["pdf_dev"] = float(np.linalg.norm(c_pdf - c_exp) / D)
        if aligned:
            out["pdf_H"] = T_scan
            out["pdf_sim"] = T_scan
            T_R2, cc2 = self._dense(reg, found["page"], found["T_R"], found["cc"])
            if T_R2 is not None:
                out["pdf_H"] = np.linalg.inv(reg["S"]) @ T_R2
                out["pdf_cc"] = cc2
                out["pdf_ratio"] = float(np.clip(cc2, 0, 1))
                out["pdf_dense"] = True
            if found["mode"] == "SIFT" and anchor is None and (found["inl"] >= 30 or (found["inl"] >= 20 and found["cc"] >= 0.35)):
                self._learn(Xc, T_scan)
        return out

    # ----------------------------------------------- verifica di una pagina
    def verify_page(self, image, pair, D, geom, page, prior, anchor=None):
        """Verifica che la pagina `page` (0-based) sia quella giusta per il frame; None se no."""
        reg = self._region(image, pair, D)
        if reg is None or prior is None or not (0 <= page < len(self.pages)):
            return None
        Xc = _cross_matrix(pair, D, geom)
        exp_R = reg["S"] @ Xc @ prior
        sc = self._guided_scores(reg, exp_R, range(len(self.pages)))
        by = {p: (n, T2) for n, p, T2 in sc}
        n_c, T2 = by[page]
        best_other = max([n for n, p, _ in sc if p != page] or [0.0])
        if n_c < 0.18 or n_c < best_other - 0.05:
            return None
        T_R, cc = self._refine(page, reg["g"], T2)
        if cc < 0.20:
            return None
        found = dict(page=page, T_R=T_R, cc=cc, inl=0, err=0.0, mode="sequenza")
        return self._finish(reg, Xc, found, anchor, prior)

    # ------------------------------------------------ passata sui frame
    def resolve(self, frames, geom, log=print, imread=None, cancel=None):
        """Dopo l'analisi: 1) riprova (ricerca guidata) i frame non riconosciuti;
        2) propaga le pagine lungo le colonne e verifica/corregge i dubbi."""
        imread = imread or imread_unicode
        stats = Counter()
        by_scan = {}
        for f in frames:
            by_scan.setdefault(f["scan"], []).append(f)
        cache = {}

        def img_of(path):
            if path not in cache:
                cache.clear()
                cache[path] = imread(path)
            return cache[path]

        def conf(f):
            p = f["pair"]
            return p.get("pdf_num") is not None and (p.get("pdf_inliers", 0) >= self.STRONG_INL
                                                    or p.get("pdf_cc", 0) >= self.STRONG_CC)

        # --- fase 1: ricerca guidata sui frame senza pagina
        if self.prior is not None:
            for path, fs in by_scan.items():
                for f in fs:
                    if cancel and cancel():
                        log("PDF: operazione annullata.")
                        return stats
                    if (f["pair"].get("pdf_num") is not None or f.get("pdf_gen") == self.prior_gen
                            or f.get("num_manual")):
                        continue
                    f["pdf_gen"] = self.prior_gen
                    img = img_of(path)
                    if img is None:
                        continue
                    m = self.identify(img, f["pair"], f["D"], geom, anchor=f["pair"].get("anchor"), guided_only=True)
                    if m:
                        _apply_match(f, img, geom, m)
                        stats["guida"] += 1
        # --- fase 2: sequenze
        runs = {}
        for f in frames:
            runs.setdefault(f["run"], []).append(f)
        counts = {}
        per_run = {}
        for rid, fs in runs.items():
            if len(fs) < 2:
                continue
            votes = {1: Counter(), -1: Counter()}
            for f in fs:
                if conf(f):
                    for sg in (1, -1):
                        votes[sg][f["pair"]["pdf_num"] - 1 - sg * f["idx"]] += 1
            per_run[rid] = votes
            for sg in (1, -1):
                if votes[sg]:
                    counts[sg] = counts.get(sg, 0) + (max(votes[sg].values()) if max(votes[sg].values()) >= 2 else 0)
        gsign = None
        if counts.get(1, 0) != counts.get(-1, 0):
            gsign = 1 if counts.get(1, 0) > counts.get(-1, 0) else -1
        for rid, votes in per_run.items():
            if cancel and cancel():
                log("PDF: operazione annullata.")
                return stats
            fs = runs[rid]
            best = None
            for sg in ((gsign,) if gsign else (1, -1)):
                if votes[sg]:
                    base, n = votes[sg].most_common(1)[0]
                    if n >= 2 or (gsign and n >= 1):
                        if best is None or n > best[2]:
                            best = (sg, base, n)
            if best is None:
                continue
            sg, base, _ = best
            # prior locale della colonna (stesso layout stampato) -> piu' affidabile di quello globale
            loc = [np.linalg.inv(_cross_matrix(f["pair"], f["D"], geom)) @ f["pair"].get("pdf_sim", f["pair"]["pdf_H"])
                   for f in fs if conf(f) and "pdf_H" in f["pair"]
                   and f["pair"]["pdf_num"] - 1 == base + sg * f["idx"]]
            prior = None
            if loc:
                a = np.array([_sim_rs(m) + (m[0, 2], m[1, 2]) for m in loc])
                prior = _sim_make(float(np.median(a[:, 0])),
                                  math.atan2(np.median(np.sin(a[:, 1])), np.median(np.cos(a[:, 1]))),
                                  float(np.median(a[:, 2])), float(np.median(a[:, 3])))
            elif self.prior is not None:
                prior = self.prior
            if prior is None:
                continue
            for f in fs:
                if cancel and cancel():
                    log("PDF: operazione annullata.")
                    return stats
                if f.get("num_manual"):
                    continue
                want = base + sg * f["idx"]
                if not (0 <= want < len(self.pages)):
                    continue
                have = f["pair"].get("pdf_num")
                if have is not None and (have - 1 == want or conf(f)):
                    continue
                if f.get("pdf_seq") == (want, id(prior)):
                    continue
                f["pdf_seq"] = (want, id(prior))
                img = img_of(f["scan"])
                if img is None:
                    continue
                m = self.verify_page(img, f["pair"], f["D"], geom, want, prior, anchor=f["pair"].get("anchor"))
                if m and (have is None or m["pdf_cc"] > f["pair"].get("pdf_cc", 0) + 0.05):
                    _apply_match(f, img, geom, m)
                    stats["sequenza"] += 1
        total = len(frames)
        ok = sum(1 for f in frames if f["pair"].get("pdf_num") is not None)
        log(f"PDF: {ok}/{total} frame agganciati ({stats['guida']} con ricerca guidata, "
            f"{stats['sequenza']} per sequenza). Non agganciati: {total - ok}.")
        return stats


# ------------------------------------------------- integrazione nei frame
def crop_frame(img, pair, D, geom, scale_mult=1., interp=cv2.INTER_LANCZOS4):
    if "pdf_H" not in pair:
        return _crop_base(img, pair, D, geom, scale_mult, interp)
    w, h = pair["pdf_size"]
    A = (pair["pdf_H"] @ np.diag([1 / scale_mult, 1 / scale_mult, 1.]))[:2]
    return cv2.warpAffine(img, A, (max(1, round(w * scale_mult)), max(1, round(h * scale_mult))),
                          flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def footprint_poly(pair, D, geom, k):
    if "pdf_H" not in pair:
        return _footprint_base(pair, D, geom, k)
    w, h = pair["pdf_size"]
    p = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), pair["pdf_H"])[0]
    return [tuple(v * k) for v in p]


def _apply_match(f, img, geom, match, thumb_w=360):
    pair = f["pair"]
    for key in ("pdf_H", "pdf_dev"):
        pair.pop(key, None)
    pair.update(match)
    if "pdf_H" in pair:
        scale = thumb_w / pair["pdf_size"][0]
        how = ""
    else:
        if pair.get("corr") is None:
            refine_rect(img, pair, f["D"], geom)
        scale = thumb_w / (2 * geom["half_w"])
        how = " | allineato sui crocini"
    f["thumb"] = crop_frame(img, pair, f["D"], geom, scale, cv2.INTER_LINEAR)
    manual_num = bool(f.get("num_manual"))      # numero deciso dall'utente: il PDF non lo tocca
    if not manual_num:
        f["num"], f["locked"] = match["pdf_num"], True
    f["rect_score"] = match["pdf_ratio"]
    f["pdf_missing"] = False
    det = f"{match['pdf_inliers']} punti" if match["pdf_mode"] == "SIFT" else match["pdf_mode"]
    weak = " | VERIFICA" if match["pdf_cc"] < 0.25 else ""
    dense = " | denso" if match.get("pdf_dense") else ""
    if not manual_num:
        f["status"] = f"PDF p.{match['pdf_num']} | {det}{dense} | cc {match['pdf_cc']:.2f}{how}{weak}"


def make_frame(img, gray, pair, D, geom, path, run, idx, col, thumb_w=360):
    D = math.hypot(pair["bot"][0] - pair["top"][0], pair["bot"][1] - pair["top"][1])
    if _PDF_ENGINE is not None:
        try:
            match = _PDF_ENGINE.identify(img, pair, D, geom, anchor=pair.get("anchor"))
        except (cv2.error, ValueError, RuntimeError, np.linalg.LinAlgError):
            match = None
        if match:
            f = dict(scan=str(path), pair=pair, D=D, run=run, idx=idx, col=col, ocr=None, conf=0., roi=None,
                     thumb=None, include=True, rect_score=0., num=None, locked=True, status="")
            _apply_match(f, img, geom, match, thumb_w)
            return f
    frame = _make_base(img, gray, pair, D, geom, path, run, idx, col, thumb_w)
    frame["pdf_missing"] = _PDF_ENGINE is not None
    return frame


def infer_numbers(frames, direction=0):
    result = _infer_base(frames, direction)
    for f in frames:
        if f.get("pdf_missing"):
            f["status"] = "PDF non confermato | " + f["status"]
    return result
# ===================== FINE MOTORE PDF =====================





# ===================== USCITA UNIFORME (v8) =====================
# Tutti i PNG escono con la STESSA dimensione e lo STESSO rettangolo-foto negli stessi pixel:
# ogni frame viene portato nello spazio "canonico" dei crocini (o del PDF, se calibrato), i quattro
# bordi del rettangolo stampato vengono MISURATI frame per frame e fatti coincidere con quelli del
# riferimento (template, pagina PDF o geometria). Cosi' le bande bianche sono identiche per forza.
OUT_MODES = ["Come template", "Come PDF", "Personalizzata", "Originale (mediana)"]
OUT_FITS = ["Adatta (bande uguali)", "Riempi (taglia)"]


def _geom_key(geom):
    return tuple(round(float(geom[k]), 3) for k in ("ref_d", "half_w", "top", "bottom"))


def _prior_for(engine, geom):
    """Layout pagina->canonico appreso dal PDF, valido solo se la geometria non e' cambiata."""
    if engine is None or engine.prior is None:
        return None
    return engine.prior if getattr(engine, "geom_key", None) == _geom_key(geom) else None


def _even(v):
    v = int(round(v))
    return v + (v % 2)


def nominal_rect(geom):
    """Rettangolo-foto nominale (x0, y0, x1, y1) nello spazio canonico del ritaglio."""
    cx = geom["half_w"] + RECT["dx"]
    y0 = geom["top"] + RECT["dy"]
    return (cx - RECT["w"] / 2, y0, cx + RECT["w"] / 2, y0 + RECT["h"])


def canon_matrix(f, geom, engine):
    """canonico -> scansione. Usa il PDF (se calibrato) altrimenti i crocini (+ aggancio rettangolo)."""
    pair = f["pair"]
    prior = _prior_for(engine, geom)
    if "pdf_H" in pair and prior is not None:
        return pair["pdf_H"] @ np.linalg.inv(prior)
    A, _ = affine_for_pair(pair, f["D"], geom, 1.0)
    return _as3(A)


def _edge(prof, tau, lo, hi, first=True, hold=8):
    """Indice sub-pixel in cui il profilo supera tau (e ci resta) cercando in [lo, hi]."""
    n = len(prof)
    lo, hi = max(1, int(lo)), min(n - 2, int(hi))
    idx = range(lo, hi + 1) if first else range(hi, lo - 1, -1)
    for i in idx:
        a, b = (prof[i - 1], prof[i]) if first else (prof[i + 1], prof[i])
        if a < tau <= b:
            seg = prof[i:min(n, i + hold)] if first else prof[max(0, i - hold + 1):i + 1]
            if len(seg) and seg.min() >= 0.6 * tau:
                t = (tau - a) / (b - a) if b != a else 0.5
                return (i - 1 + t) if first else (i + 1 - t)
    return None


def find_rect(gray, nominal=None, thr=175):
    """Bordi sub-pixel (x0, y0, x1, y1) del rettangolo scuro della foto; i bordi non
    affidabili sono None. Usa le fasce laterali (dove la foto e' sempre scura: al centro c'e'
    l'alone chiaro) per i bordi alto/basso e le righe centrali per quelli laterali."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (0, 0), 2.0)
    dark = (g < thr).astype(np.float32)
    nx0, ny0, nx1, ny1 = nominal if nominal else (0.0, 0.2 * H, float(W), 0.8 * H)
    nw, nh = nx1 - nx0, ny1 - ny0

    def cols(a, b):
        return dark[:, int(max(0, nx0 + a * nw)):int(min(W, nx0 + b * nw))]
    prof = 0.5 * (cols(.03, .20).mean(1) + cols(.80, .97).mean(1))
    inner = prof[int(max(0, ny0 + .25 * nh)):int(min(H, ny1 - .25 * nh))]
    out = [None, None, None, None]
    if inner.size and np.median(inner) > 0.35:
        tau = 0.5 * float(np.median(inner))
        w = 0.15 * nh
        out[1] = _edge(prof, tau, ny0 - w, ny0 + w, True)
        out[3] = _edge(prof, tau, ny1 - w, ny1 + w, False)
    ya = out[1] if out[1] is not None else ny0
    yb = out[3] if out[3] is not None else ny1
    r0, r1 = int(max(0, ya + .2 * (yb - ya))), int(min(H, yb - .2 * (yb - ya)))
    if r1 > r0 + 4:
        cp = dark[r0:r1].mean(0)
        side = cp[int(max(0, nx0 + .03 * nw)):int(min(W, nx0 + .2 * nw))]
        if side.size and np.median(side) > 0.3:
            tau = 0.5 * float(np.median(side))
            w = 0.08 * nw
            out[0] = _edge(cp, tau, nx0 - w, nx0 + w, True)
            out[2] = _edge(cp, tau, nx1 - w, nx1 + w, False)
    return tuple(out)


def layout_from_gray(gray, snap=3.0):
    """Dimensione + rettangolo di un riferimento (template o pagina PDF) con la foto/rettangolo nero."""
    H, W = gray.shape
    b = (gray < 80)
    rows = np.where(b.mean(1) > 0.5)[0]
    if rows.size == 0:
        e = find_rect(gray)
    else:
        runs = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
        run = max(runs, key=len)
        c = np.where(b[run[0]:run[-1] + 1].mean(0) > 0.5)[0]
        e = (float(c[0]), float(run[0]), float(c[-1] + 1), float(run[-1] + 1)) if c.size else (None,) * 4
    if None in e:
        return dict(W=W, H=H, rect=None)
    x0, y0, x1, y1 = e
    if x0 <= snap:
        x0 = 0.0
    if W - x1 <= snap:
        x1 = float(W)
    return dict(W=W, H=H, rect=(x0, y0, x1, y1))


_LAYOUT_CACHE = {}


def template_layout(path):
    """Con cache: update_osz() la richiama a ogni aggiornamento della tabella."""
    if not path or not Path(path).is_file():
        return None
    st = Path(path).stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE.clear()
        _LAYOUT_CACHE[key] = _template_layout_uncached(path)
    lay = _LAYOUT_CACHE[key]
    return dict(lay) if lay else lay


def _template_layout_uncached(path):
    """Layout del template: dimensioni, rettangolo e (se ce n'e' UNA sola coppia) crocini."""
    if not path or not Path(path).is_file():
        return None
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    lay = layout_from_gray(img)
    H, W = img.shape
    _, b = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    n, _, st, cen = cv2.connectedComponentsWithStats(b)
    cr = [(float(cen[i][0]), float(cen[i][1])) for i in range(1, n)
          if 30 <= st[i][2] <= 200 and 0.8 <= st[i][2] / st[i][3] <= 1.25 and st[i][4] > 150]
    if len(cr) == 2:
        (x1, y1), (x2, y2) = sorted(cr, key=lambda c: c[1])
        if abs(x1 - x2) <= 0.03 * W and y2 - y1 >= 0.1 * H:
            lay.update(xc=(x1 + x2) / 2, yt=y1, dy=y2 - y1)
    return lay


def compute_spec(mode, fit, custom, frames, geom, engine, tpl_path, use_rect=True, overscan=1.0,
                 pdf_first=True, check=False, paper=True):
    nom = nominal_rect(geom)
    bw, bh = 2 * geom["half_w"], geom["top"] + geom["ref_d"] + geom["bottom"]
    spec = dict(mode=mode, fill=str(fit).startswith("Riempi"), use_rect=bool(use_rect),
                over=float(overscan), warn="", ref=None, pdf_first=bool(pdf_first), check=bool(check),
                paper=bool(paper))
    lay = None
    if mode == "Come template":
        lay = template_layout(tpl_path)
        if lay is None:
            raise ValueError("Template non leggibile: scegli il file del template o un'altra modalita'.")
        spec["size"] = (_even(lay["W"]), _even(lay["H"]))
        spec["ref"] = "template"
    elif mode == "Come PDF":
        lay = getattr(engine, "layout", None) if engine is not None else None
        if engine is None or not engine.pages:
            raise ValueError("Per 'Come PDF' prepara prima il PDF.")
        spec["size"] = (_even(engine.pages[0][0]), _even(engine.pages[0][1]))
        spec["ref"] = "PDF"
    elif mode == "Personalizzata":
        W, H = int(custom[0]), int(custom[1])
        if not (16 <= W <= 20000 and 16 <= H <= 20000):
            raise ValueError("Larghezza e altezza devono essere tra 16 e 20000 px.")
        spec["size"] = (_even(W), _even(H))
    else:   # Originale (mediana)
        ss = [math.sqrt(abs(float(np.linalg.det(canon_matrix(f, geom, engine)[:2, :2])))) for f in frames]
        if not ss:
            raise ValueError("Nessun frame da salvare.")
        s = float(np.median(ss))
        spec["size"] = (_even(bw * s), _even(bh * s))
    W, H = spec["size"]
    rect = lay.get("rect") if lay else None
    if rect is not None and lay.get("W") and lay["W"] != W:      # arrotondamento a pari: solo bordo destro/basso
        pass
    if rect is None:
        if lay is not None and mode == "Come template" and "dy" in lay:
            # template senza rettangolo riconoscibile: ancoro sui crocini
            sc = lay["dy"] / geom["ref_d"]
            ox = lay["xc"] - geom["half_w"] * sc
            oy = lay["yt"] - geom["top"] * sc
            rect = (nom[0] * sc + ox, nom[1] * sc + oy, nom[2] * sc + ox, nom[3] * sc + oy)
            spec["warn"] = "Nel template non ho trovato il rettangolo: uso i crocini."
        else:
            s = max(W / bw, H / bh) if spec["fill"] else min(W / bw, H / bh)
            ox, oy = (W - bw * s) / 2, (H - bh * s) / 2
            rect = (nom[0] * s + ox, nom[1] * s + oy, nom[2] * s + ox, nom[3] * s + oy)
            if lay is not None:
                spec["warn"] = "Nel riferimento non ho trovato il rettangolo: uso la geometria dei crocini."
    spec["rect_t"] = tuple(float(v) for v in rect)
    if pdf_first and frames:
        n_pdf = sum(1 for f in frames if "pdf_H" in f["pair"])
        if 0 < n_pdf < len(frames):
            spec["warn"] = (spec["warn"] + " " if spec["warn"] else "") + \
                f"{len(frames) - n_pdf} frame su {len(frames)} non agganciati al PDF: usano il rettangolo/crocini."
    return spec


def canon_crop(img, Gc, geom, q=0.5):
    """Ritaglio canonico a bassa risoluzione (per misurare i bordi), con pre-filtro anti-retino."""
    bw, bh = 2 * geom["half_w"], geom["top"] + geom["ref_d"] + geom["bottom"]
    pts = cv2.perspectiveTransform(np.float32([[[0, 0], [bw, 0], [bw, bh], [0, bh]]]), Gc)[0]
    x0, y0 = max(0, int(pts[:, 0].min()) - 8), max(0, int(pts[:, 1].min()) - 8)
    x1, y1 = min(img.shape[1], int(pts[:, 0].max()) + 8), min(img.shape[0], int(pts[:, 1].max()) + 8)
    if x1 <= x0 + 20 or y1 <= y0 + 20:
        return None
    reg = img[y0:y1, x0:x1]
    reg = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
    f = math.sqrt(abs(float(np.linalg.det(Gc[:2, :2])))) / q            # px scansione per px del ritaglio
    if f > 1.2:
        reg = cv2.GaussianBlur(reg, (0, 0), 0.5 * f)
    M = (_T(-x0, -y0) @ Gc @ np.diag([1 / q, 1 / q, 1.0]))[:2]
    return cv2.warpAffine(reg, M, (max(1, round(bw * q)), max(1, round(bh * q))),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=255)


def measure_rect(img, Gc, geom, q=0.5):
    """Bordi del rettangolo stampato in coordinate canoniche. Un bordo non misurabile o troppo
    lontano da quello nominale (testo, timbri...) viene sostituito dal nominale."""
    nom = nominal_rect(geom)
    small = canon_crop(img, Gc, geom, q)
    if small is None:
        return nom, 0
    e = find_rect(small, tuple(v * q for v in nom))
    tol = 0.05 * geom["ref_d"]
    out, ok = [], 0
    for v, n in zip(e, nom):
        if v is not None and abs(v / q - n) <= tol:
            out.append(v / q)
            ok += 1
        else:
            out.append(n)
    return tuple(out), ok


def pdf_out_to_page(f, spec):
    """Pixel di uscita -> pixel della pagina PDF: la pagina intera entra nella tela (Adatta) o la
    riempie (Riempi); il 'taglio bordo' ingrandisce di o px per lato."""
    pw, ph = f["pair"]["pdf_size"]
    W, H = spec["size"]
    o = spec["over"]
    Wo, Ho = W + 2 * o, H + 2 * o
    s = max(Wo / pw, Ho / ph) if spec["fill"] else min(Wo / pw, Ho / ph)
    return np.array([[1 / s, 0, o / s + (pw - Wo / s) / 2], [0, 1 / s, o / s + (ph - Ho / s) / 2], [0, 0, 1.0]])


MAX_INTERMEDIATE_PX = 120_000_000      # tetto dei pixel dell'intermedio in _warp_out (~360 MB a 3 canali)


def _fill_missing_paper(out, src_hw, base, spec, scale):
    """Dove la scansione non arriva (il frame sporge dal bordo) al posto del bianco mette la tinta della carta,
    misurata nello STESSO frame: carta vera fuori dal rettangolo-foto, scartando testo e zone scure.
    E' una tinta UNIFORME: non prolunga la grana della carta (soluzione provvisoria, vedi Q-01). La carta reale
    non viene toccata. Se non c'e' abbastanza carta da campionare lascia il bianco (comportamento precedente)."""
    try:
        if out.ndim != 3:
            return out
        H, W = src_hw
        M = base @ np.diag([1 / scale, 1 / scale, 1.0])                  # uscita -> scansione
        corners = np.float32([[[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]]])
        poly = cv2.perspectiveTransform(corners, np.linalg.inv(M))[0]     # bordi della scansione, in uscita
        if not np.isfinite(poly).all() or np.abs(poly).max() > 1e7:
            return out
        oh, ow = out.shape[:2]
        valid = np.zeros((oh, ow), np.uint8)
        cv2.fillPoly(valid, [np.round(poly).astype(np.int32)], 1)
        inside = cv2.erode(valid, np.ones((3, 3), np.uint8))              # via 1 px di bordo (sfumato col bianco)
        if inside.all():
            return out                                                    # frame tutto dentro la scansione
        sample = cv2.erode(valid, np.ones((7, 7), np.uint8)).astype(bool)
        r = spec.get("rect_t")
        if r:                                                             # niente foto: solo carta intorno
            m = 12.0 * scale
            x0, y0, x1, y1 = (int(max(0, v)) for v in (r[0] * scale - m, r[1] * scale - m,
                                                        r[2] * scale + m, r[3] * scale + m))
            sample[y0:y1, x0:x1] = False
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(sample)
        if ys.size < 400:
            return out
        g = gray[ys, xs]
        keep = g >= np.median(g)                                          # meta' piu' chiara: scarta testo/ombre
        if int(keep.sum()) < 200:
            return out
        px = out[ys[keep], xs[keep]]
        if px.shape[0] > 200000:
            px = px[::px.shape[0] // 200000]
        tone = np.median(px, axis=0)
        if float(tone.mean()) < 120:                                      # campione scuro: non e' carta
            return out
        out[inside == 0] = tone.astype(np.uint8)
        return out
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return out


def _warp_out(img, base, spec, interp, scale, quality):
    out = _warp_out_raw(img, base, spec, interp, scale, quality)
    if spec.get("paper"):
        out = _fill_missing_paper(out, img.shape[:2], base, spec, scale)
    return out


def _warp_out_raw(img, base, spec, interp, scale, quality):
    W, H = spec["size"]
    nat = math.sqrt(abs(float(np.linalg.det(base[:2, :2]))))
    ow, oh = max(1, round(W * scale)), max(1, round(H * scale))
    if quality and np.isfinite(nat) and nat > 0:
        # il passaggio intermedio e' grande W*nat x H*nat: lo si limita a un budget di pixel, altrimenti
        # una scansione ad alta risoluzione (o una trasformazione anomala) puo' esaurire la RAM
        nat_eff = nat
        px = float(W) * nat * float(H) * nat
        if px > MAX_INTERMEDIATE_PX:
            nat_eff = nat * math.sqrt(MAX_INTERMEDIATE_PX / px)
        if nat_eff / scale > 1.25:
            # riduzione forte: prima a scala nativa (Lanczos), poi INTER_AREA -> niente moire' sul retino
            big = cv2.warpAffine(img, (base @ np.diag([1 / nat_eff, 1 / nat_eff, 1.0]))[:2],
                                 (max(1, round(W * nat_eff)), max(1, round(H * nat_eff))),
                                 flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
            return cv2.resize(big, (ow, oh), interpolation=cv2.INTER_AREA)
    A = (base @ np.diag([1 / scale, 1 / scale, 1.0]))[:2]
    return cv2.warpAffine(img, A, (ow, oh), flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def check_overlay(out, f, spec, engine):
    """Controllo visivo: pagina PDF (ciano) sovrapposta alla scansione ritagliata (rosso).
    Allineato = grigio; sfrangiature rosse/ciano = disallineamento. Ritorna (immagine, correlazione)."""
    p = f["pair"].get("pdf_num")
    if engine is None or p is None or not (1 <= p <= len(engine.pages)):
        return None, None
    mf = engine.mid_f[p - 1]
    W, H = spec["size"]
    ow, oh = out.shape[1], out.shape[0]
    k = ow / W
    M = (np.diag([mf, mf, 1.0]) @ pdf_out_to_page(f, spec) @ np.diag([1 / k, 1 / k, 1.0]))[:2]
    pg = cv2.warpAffine(engine.mid[p - 1], M, (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    sc = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    cc = _ncc(_dog(pg, 2.0 * k, 12.0 * k), _dog(sc, 2.0 * k, 12.0 * k))
    return np.dstack([pg, pg, sc]), cc


def render_output(img, f, spec, geom, engine, interp=cv2.INTER_LANCZOS4, scale=1.0, quality=True):
    """Ritaglio finale. Frame agganciato al PDF: la pagina del PDF e' il riferimento (aggancio
    sul contenuto della foto). Altrimenti: crocini + bordi del rettangolo misurati."""
    if spec.get("pdf_first", True) and "pdf_H" in f["pair"]:
        return _warp_out(img, f["pair"]["pdf_H"] @ pdf_out_to_page(f, spec), spec, interp, scale, quality)
    Gc = canon_matrix(f, geom, engine)
    nom = nominal_rect(geom)
    src = measure_rect(img, Gc, geom)[0] if spec["use_rect"] else nom
    t = list(spec["rect_t"])
    o = spec["over"]                       # leggero sovra-taglio: niente filo bianco sui bordi
    t = [t[0] - o, t[1] - o, t[2] + o, t[3] + o]
    kx, ky = (src[2] - src[0]) / (t[2] - t[0]), (src[3] - src[1]) / (t[3] - t[1])
    F = np.array([[kx, 0, src[0] - t[0] * kx], [0, ky, src[1] - t[1] * ky], [0, 0, 1.0]])
    return _warp_out(img, Gc @ F, spec, interp, scale, quality)
# ===================== FINE USCITA UNIFORME =====================


# ----------------------------------------------------------- interfaccia
_build_pdf_previous = App._build
_select_pdf_previous = App.on_select
_editor_pdf_previous = PageEditor.__init__
_commit_pdf_previous = PageEditor.commit


def _pdf_on_select(self, event=None):
    _select_pdf_previous(self, event)
    _, f = self.cur()
    if f is None:
        return
    if f.get('roi') is None:
        self._roi_photo = None
        self.roi_lbl.configure(image='', text='')
    p = f['pair']
    if 'pdf_H' in p:
        self.pdf_status.set(f"PDF p.{p['pdf_num']} | {p.get('pdf_mode')} | correlazione {p.get('pdf_cc', 0):.2f}"
                            f" | allineamento a similitudine: il riquadro e' un rettangolo perfetto")
    elif p.get('pdf_num') is not None:
        self.pdf_status.set(f"PDF p.{p['pdf_num']} riconosciuta, ma l'allineamento PDF non e' coerente con i crocini "
                            f"(scarto {p.get('pdf_dev', 0) * 100:.1f}% di D): ritaglio sui crocini")
    else:
        self.pdf_status.set('Frame non confermato dal PDF: ritaglio tramite crocini')


def _pdf_build(self):
    _build_pdf_previous(self)
    self.v_pdf = tk.StringVar()
    self.v_pdf_rotate = tk.BooleanVar(value=False)
    self.pdf_status = tk.StringVar(value='PDF non preparato: disponibile il motore a crocini')
    self._pdf_queue = queue.Queue()
    bar = tk.Frame(self.root, bg=BODY)
    bar.pack(after=self.lcd, fill='x', padx=12, pady=5)
    tk.Label(bar, text='PDF DI STAMPA', bg=BODY, fg=INK, font=(FONT, 9)).pack(side='left')
    self._entry(bar, self.v_pdf, width=45).pack(side='left', padx=5)

    def choose():
        p = filedialog.askopenfilename(filetypes=[('PDF', '*.pdf')])
        if p:
            self.v_pdf.set(p)
    Key(bar, '...', choose, width=40, height=28).pack(side='left')
    self.pdf_prepare_button = Key(bar, 'PREPARA PDF', self.prepare_pdf, kind='blue', width=150, height=28)
    self.pdf_prepare_button.pack(side='left', padx=5)
    tk.Checkbutton(bar, text='Scansioni capovolte: ruota 180', variable=self.v_pdf_rotate,
                   bg=BODY, fg=INK).pack(side='left')
    self.btn_cancel = Key(bar, 'ANNULLA', self.cancel, kind='red', width=110, height=28)
    self.btn_cancel.pack(side='left', padx=8)
    self.btn_cancel.enable(False)
    bar2 = tk.Frame(self.root, bg=BODY)
    bar2.pack(after=bar, fill='x', padx=12, pady=(0, 4))
    self.btn_proj_save = Key(bar2, 'SALVA PROGETTO', self.save_project, width=170, height=28)
    self.btn_proj_save.pack(side='left', padx=(0, 8))
    self.btn_proj_load = Key(bar2, 'APRI PROGETTO', self.load_project, width=170, height=28)
    self.btn_proj_load.pack(side='left', padx=(0, 8))
    self.btn_theme = Key(bar2, 'TEMA SCURO', self.toggle_theme, kind='blue', width=170, height=28)
    self.btn_theme.pack(side='left', padx=(0, 8))
    self.btn_erased = Key(bar2, 'RIPRISTINA CANCELLATI', self.restore_erased, width=210, height=28)
    self.btn_erased.pack(side='left', padx=(0, 8))
    tk.Label(bar2, text='ANIMAZIONE', bg=BODY, fg=INK, font=(FONT, 9)).pack(side='left', padx=(8, 4))
    prefs_ = load_prefs()
    self.v_anim = tk.StringVar(value=prefs_.get('anim') if prefs_.get('anim') in ANIMS else DEFAULT_ANIM)
    self.lcd.anim = self.v_anim.get()
    cb_anim = ttk.Combobox(bar2, textvariable=self.v_anim, state='readonly', width=18, values=list(ANIMS))
    cb_anim.pack(side='left')

    def anim_changed(_e=None):
        self.lcd.anim = self.v_anim.get()
        p = load_prefs()
        p['anim'] = self.v_anim.get()
        save_prefs(p)
    cb_anim.bind('<<ComboboxSelected>>', anim_changed)
    info = tk.Label(self.root, textvariable=self.pdf_status, bg=BODY, fg=INK, font=(FONT, 9),
                    anchor='w', wraplength=1050)
    info.pack(after=bar2, fill='x', padx=12)
    # due comandi SEPARATI: uno aggiorna la grafica, l'altro rifa' il riconoscimento
    self.btn_prev = Key(self.canvas.master, '↻ AGGIORNA ANTEPRIME', self.refresh_previews,
                        width=340, height=30)
    self.btn_prev.pack(before=self.canvas, anchor='w', pady=(0, 3))
    self.btn_retry = Key(self.canvas.master, 'RIPROVA RICONOSCIMENTO PDF', self.retry_recognition,
                         kind='blue', width=340, height=30)
    self.btn_retry.pack(before=self.canvas, anchor='w', pady=(0, 5))


def _pdf_prepare(self):
    if self.is_busy:
        return
    path = self.v_pdf.get().strip()
    if not Path(path).is_file():
        messagebox.showerror('PDF', 'Seleziona un PDF valido')
        return
    self.busy(True)
    self.pdf_prepare_button.enable(False)
    self.btn_cancel.enable(False)         # la preparazione del PDF non e' interrompibile
    self.pdf_status.set('Preparazione riferimenti PDF...')
    t0 = time.perf_counter()

    def worker():
        try:
            engine = PDFReferences(path, lambda msg: self._pdf_queue.put(('progress', msg)),
                                   report=lambda msg: self._pdf_queue.put(('log', msg)))
            self._pdf_queue.put(('ready', engine, str(Path(path).resolve())))
        except Exception as ex:
            file_log('ERRORE preparazione PDF:\n' + traceback.format_exc())
            self._pdf_queue.put(('error', str(ex)))
    threading.Thread(target=worker, daemon=True).start()

    def poll():
        global _PDF_ENGINE
        try:
            while True:
                item = self._pdf_queue.get_nowait()
                if item[0] == 'progress':
                    self.pdf_status.set(item[1])
                elif item[0] == 'log':
                    self.write(item[1])
                else:
                    self.busy(False)
                    self.pdf_prepare_button.enable(True)
                    if item[0] == 'ready':
                        _PDF_ENGINE = item[1]
                        self._prepared_pdf_path = item[2]
                        self.pdf_status.set(f'PDF pronto: {len(_PDF_ENGINE.pages)} pagine. Ora premi ANALIZZA.')
                        self.write(self.pdf_status.get())
                        self.write(f'[tempo] preparazione PDF: {time.perf_counter() - t0:.1f} s')
                    else:
                        messagebox.showerror('Preparazione PDF', item[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, poll)
    self.root.after(100, poll)


def _pdf_analyze(self):
    if self.is_busy:
        return
    if self.v_pdf.get().strip() and (_PDF_ENGINE is None or
            str(Path(self.v_pdf.get()).resolve()) != getattr(self, '_prepared_pdf_path', '')):
        messagebox.showinfo('PDF', 'Premi PREPARA PDF e attendi prima di analizzare.')
        return
    # Stato di sessione toccato PRIMA della conferma di App.analyze: lo si salva, cosi' se l'utente annulla
    # (o l'analisi non parte) i frame gia' visibili restano coerenti con rotazione e layout appresi.
    old_rot = set(_ROTATE_SCANS)
    self._frames_rot = set(old_rot)           # i frame attuali (e le decisioni raccolte nella conferma) sono in QUESTO orientamento
    old_prior = None
    if _PDF_ENGINE is not None:
        old_prior = (_PDF_ENGINE.prior, list(_PDF_ENGINE._samples), _PDF_ENGINE.prior_gen)
    _ROTATE_SCANS.clear()
    if self.v_pdf_rotate.get():
        _ROTATE_SCANS.update(str(Path(p).resolve()) for p in self.scans)
    if _PDF_ENGINE is not None:
        _PDF_ENGINE.reset_prior()
    res = _analyze_pdf_previous(self)
    if self.is_busy:                          # analisi partita: i nuovi frame nasceranno col nuovo orientamento
        self._frames_rot = set(_ROTATE_SCANS)
    if not self.is_busy:                      # l'analisi non e' partita (annullata, nessuna scansione...)
        _ROTATE_SCANS.clear()
        _ROTATE_SCANS.update(old_rot)
        if _PDF_ENGINE is not None and old_prior is not None:
            _PDF_ENGINE.prior, _PDF_ENGINE._samples, _PDF_ENGINE.prior_gen = old_prior
    return res


def _pdf_worker(self, scans, thr, D, geom):
    """Analisi in background: scansioni -> frame manuali ripristinati -> controllo sequenze PDF.
    Controlla self.cancel_ev ai punti sicuri (fra una scansione e l'altra)."""
    n = len(scans)
    collected = []
    if self._pending or self._erased:
        # le correzioni salvate in un orientamento diverso da quello di questa analisi vengono trasformate
        # (o l'aggancio PDF salvato viene scartato): niente riuso silenzioso di coordinate incompatibili
        try:
            rot_now = lambda s: str(Path(s).resolve()) in _ROTATE_SCANS
            lg = lambda m: self.q.put(("log", m))
            self._pending = adapt_rotation(self._pending, rot_now, image_size, lg)
            self._erased = adapt_rotation(self._erased, rot_now, image_size, None)
        except Exception as ex:
            file_log("ERRORE adattamento orientamento:\n" + traceback.format_exc())
            self.q.put(("log", f"ERRORE adattamento orientamento: {ex}"))
    for i, p in enumerate(scans, 1):
        if self.cancel_ev.is_set():
            self.q.put(("log", "Analisi annullata."))
            break
        self.q.put(("prog", i - 1, n, "ANALISI", Path(p).name))
        t0 = time.perf_counter()
        try:
            fr, meta = analyze_scan(p, self.tpl, thr, D, geom, log=lambda m: self.q.put(("log", m)))
            collected += fr
            self.q.put(("res", p, fr, meta))
        except Exception as ex:
            file_log(f"ERRORE analisi {Path(p).name}:\n{traceback.format_exc()}")
            self.q.put(("log", f"ERRORE {Path(p).name}: {ex}"))
        self.q.put(("log", f"[tempo] {Path(p).name}: {time.perf_counter() - t0:.1f} s"))
        self.q.put(("prog", i, n, "ANALISI", Path(p).name))
    if self._pending and not self.cancel_ev.is_set():
        try:
            extra = restore_added_frames(self._pending, collected, self.tpl, geom, self.scan_cache,
                                         lambda m: self.q.put(("log", m)))
            if extra:
                collected += extra
                self.q.put(("add", extra))
                self.q.put(("log", f"Ripristinati {len(extra)} frame aggiunti a mano."))
        except Exception as ex:
            file_log("ERRORE ripristino frame manuali:\n" + traceback.format_exc())
            self.q.put(("log", f"ERRORE ripristino frame manuali: {ex}"))
    if _PDF_ENGINE is not None and collected and not self.cancel_ev.is_set():
        try:
            self.q.put(("prog", n, n, "PDF", "controllo sequenze"))
            _PDF_ENGINE.resolve(collected, geom, log=lambda m: self.q.put(("log", m)),
                                cancel=self.cancel_ev.is_set)
        except Exception as ex:
            file_log("ERRORE controllo sequenze PDF:\n" + traceback.format_exc())
            self.q.put(("log", f"ERRORE controllo sequenze PDF: {ex}"))
    self.q.put(("done",))


def _pdf_editor_init(self, *args, **kwargs):
    _editor_pdf_previous(self, *args, **kwargs)

    def closed(event):
        if event.widget is self.win:
            path = self.path
            self.app.root.after_idle(lambda: self.app.refresh_previews(
                [f for f in self.app.frames if f['scan'] == path]))
            self.app.autosave()
    self.win.bind('<Destroy>', closed, add='+')


def _pdf_commit(self, pair, D, how):
    if self.app.is_busy:
        self.app.write('Operazione in corso: attendi o premi ANNULLA prima di aggiungere frame.')
        return
    pair['anchor'] = how      # 'crocini' / 'rettangolo': il PDF non puo' contraddire i crocini cliccati
    return _commit_pdf_previous(self, pair, D, how)


_analyze_pdf_previous = App.analyze
App._build = _pdf_build
App.prepare_pdf = _pdf_prepare
App.analyze = _pdf_analyze
App._worker = _pdf_worker
App.on_select = _pdf_on_select
PageEditor.__init__ = _pdf_editor_init
PageEditor.commit = _pdf_commit



def _virtual_screen(root):
    """(x, y, larghezza, altezza) dell'intero desktop. Su Windows include tutti i monitor; altrove
    ripiega sul monitor principale."""
    try:
        import ctypes
        gm = ctypes.windll.user32.GetSystemMetrics
        x, y, w, h = gm(76), gm(77), gm(78), gm(79)       # SM_[X|Y|CX|CY]VIRTUALSCREEN
        if w > 0 and h > 0:
            return x, y, w, h
    except (ImportError, AttributeError, OSError):
        pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def initial_window_geometry(root, prefs):
    """Dimensione e posizione della finestra: quelle salvate se ancora sensate, altrimenti quelle predefinite.
    Se il monitor usato e' stato scollegato (o la risoluzione e' cambiata) la finestra non finisce fuori schermo."""
    vx, vy, vw, vh = _virtual_screen(root)
    w = min(1250, max(600, root.winfo_screenwidth() - 80))
    h = min(900, max(400, root.winfo_screenheight() - 100))
    x = y = None
    zoomed = False
    g = prefs.get("geometry")
    if isinstance(g, dict):
        try:
            gw, gh, gx, gy = int(g["w"]), int(g["h"]), int(g["x"]), int(g["y"])
            gw, gh = min(max(600, gw), vw), min(max(400, gh), vh)
            # almeno una parte della barra del titolo deve restare visibile
            if vx - gw + 120 <= gx <= vx + vw - 120 and vy <= gy <= vy + vh - 80:
                w, h, x, y = gw, gh, gx, gy
                zoomed = bool(g.get("zoomed", False))
        except (KeyError, TypeError, ValueError):
            pass
    return w, h, x, y, zoomed


def save_window_geometry(root):
    try:
        m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$", root.geometry())
        if not m:
            return
        w, h, x, y = (int(v) for v in m.groups())
        prefs = load_prefs()
        prefs["geometry"] = dict(w=w, h=h, x=x, y=y, zoomed=(root.state() == "zoomed"))
        save_prefs(prefs)
    except (tk.TclError, ValueError):
        pass


def main():
    root=tk.Tk()
    root.title('SCONTRINI PDF · v17')
    root.configure(bg=BODY)
    _w, _h, _x, _y, _zoomed = initial_window_geometry(root, load_prefs())
    root.geometry(f'{_w}x{_h}' + (f'+{_x}+{_y}' if _x is not None else ''))
    if _zoomed:
        try:
            root.state('zoomed')
        except tk.TclError:
            pass
    root.minsize(600,400)
    root.rowconfigure(0,weight=1);root.columnconfigure(0,weight=1)
    viewport=tk.Canvas(root,bg=BODY,highlightthickness=0,yscrollincrement=40,xscrollincrement=40)
    ys=ttk.Scrollbar(root,orient='vertical',command=viewport.yview)
    xs=ttk.Scrollbar(root,orient='horizontal',command=viewport.xview)
    viewport.configure(yscrollcommand=ys.set,xscrollcommand=xs.set)
    viewport.grid(row=0,column=0,sticky='nsew')
    ys.grid(row=0,column=1,sticky='ns');xs.grid(row=1,column=0,sticky='ew')
    content=tk.Frame(viewport,bg=BODY)
    content.title=root.title
    item=viewport.create_window(0,0,window=content,anchor='nw')
    content.bind('<Configure>',lambda e:viewport.configure(scrollregion=viewport.bbox('all')))
    def resize(event):
        viewport.itemconfigure(item,width=max(event.width,content.winfo_reqwidth()))
        viewport.configure(scrollregion=viewport.bbox('all'))
    viewport.bind('<Configure>',resize)
    app = App(content)
    install_wheel(root, viewport, app)
    app.apply_theme(load_prefs().get('theme', 'chiaro'))     # l'interfaccia nasce chiara, poi si applica la preferenza

    def on_close():
        if app.is_busy and not messagebox.askyesno(
                'Operazione in corso', 'Un\'operazione e\' ancora in corso. Chiudere comunque?'):
            return
        app.autosave()
        save_window_geometry(root)
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', on_close)
    root.after(300, app.restore_autosave)
    root.mainloop()

if __name__ == '__main__':
    main()
