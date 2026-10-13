"""Optical mouse drift simulator.

python main.py             live view of one sensor tracking the route
python main.py --batch     RUNS x LAPS Monte Carlo: drift charts + density overlay
python main.py --route     diagram of the test route
python main.py --optimize  tune tracker parameters, saves best_params.json
python main.py --optimize --replot   redraw the optimizer chart from saved history
python main.py --robust    untuned vs tuned on random routes and surfaces
python main.py --sweep     tuned tracker over sensor size x sensor rate (--replot to redraw)
"""
import functools
import json
import math
import multiprocessing
import os
import sys
import time
import numpy as np
import scipy.fft as sfft  # 5-10x faster than np.fft for these small float32 batches

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # quiet in batch worker processes
import pygame

# ==========================================
# 1. KNOBS
# ==========================================
def knob(name, default):
  """Sensor knob, overridable with env var MOUSESIM_<NAME> (worker processes inherit it)."""
  return type(default)(os.environ.get(f"MOUSESIM_{name}", default))


WIDTH, HEIGHT = 800, 600
START = (100.0, 100.0)
VIEW = knob("VIEW", 24)  # sensor viewport size, pixels per side
SUBSTEPS = knob("SUBSTEPS", 4)  # sensor frames per display frame (sensor rate / display rate)
SURFACE_BLUR = knob("SURFACE_BLUR", 1.0)  # texture grain (gaussian sigma, px); 0 = pure white noise
SENSOR_NOISE = knob("SENSOR_NOISE", 0.02)  # per-frame read noise (surface values are 0..1)
SUPERSAMPLE = knob("SUPERSAMPLE", 4)  # sub-samples per pixel axis: pixels integrate light over their area
CORRELATION = knob("CORRELATION", "circular")  # "linear" (zero-padded, overlap-normalised) or "circular"
SENSOR_WOBBLE_DEG = knob("SENSOR_WOBBLE_DEG", 0.5)  # std of smooth random sensor rotation (hand grip), deg
SEED = 4316  # surface texture seed for live view
RUNS = 10000  # batch: independent repeats of the route
LAPS = 1  # batch: laps per run, drift keeps accumulating across laps
CHUNK = 500  # batch: runs per worker task; each chunk gets its own texture
OPT_GENERATIONS = 10  # optimize: cross-entropy generations
OPT_POP = 24  # optimize: candidate parameter sets per generation
OPT_TEXTURES = 32  # optimize: cases per candidate: half README route on new textures, half random routes
OPT_RUNS = 5  # optimize: runs per case -> OPT_TEXTURES * OPT_RUNS runs per candidate
OPT_POLISH = 6  # optimize: pattern-search iterations after the cross-entropy generations
ROBUST_ROUTES = 48  # robust: random routes, each on its own random surface
ROBUST_RUNS = 5  # robust: runs per route
DENSITY_BLUR = 3.0  # density overlay gaussian kernel sigma, px

# Test route as simple turtle commands (heading 0 = right, +deg = clockwise):
#   ("go", speed, frames)            constant speed straight line
#   ("ramp", v0, v1, frames)         straight line, speed changes v0 -> v1
#   ("arc", speed, degrees, frames)  smooth curve, turns `degrees` total
#   ("turn", degrees)                instant sharp corner, no movement
#   ("wait", frames)                 stand still
#   ("home", speed)                  straight back to START (closes the loop)
ROUTE = [
    ("ramp", 0, 5, 30),  # accelerate from rest
    ("go", 5, 60),  # fast straight
    ("ramp", 5, 0, 30),  # decelerate to stop
    ("wait", 30),  # hold still
    ("turn", 90),  # sharp right-angle corner
    ("go", 1.5, 60),  # slow straight
    ("turn", 45),  # diagonal
    ("go", 3, 40),
    ("turn", -135),  # zigzag: repeated sharp changes of direction
    ("go", 3, 15), ("turn", 90), ("go", 3, 15), ("turn", -90),
    ("go", 3, 15), ("turn", 90), ("go", 3, 15), ("turn", -90),
    ("arc", 3, 180, 90),  # wide smooth U-turn
    ("arc", 2, -270, 45),  # tight fast loop the other way
    ("turn", 90),
    ("go", 0.3, 60),  # micro-movement (sub-noise speed)
    ("turn", 180),  # instant reversal
    ("go", 10, 8),  # fast flick
    ("arc", 3, 90, 40), ("arc", 3, -90, 40),  # S-curve
    ("ramp", 1, 6, 40),  # accelerate while moving
    ("home", 4),
]


def trace_route(cmds, start=START):
  """Expand route commands into per-frame (x, y, label) points."""
  x, y = start
  h = 0.0
  for n, cmd in enumerate(cmds, 1):
    kind, *a = cmd
    label = f"{n}. " + " ".join(map(str, cmd))
    if kind == "turn":
      h += math.radians(a[0])
      continue
    if kind == "wait":
      v0, v1, turn, frames = 0, 0, 0, a[0]
    elif kind == "go":
      v0, v1, turn, frames = a[0], a[0], 0, a[1]
    elif kind == "ramp":
      v0, v1, turn, frames = a[0], a[1], 0, a[2]
    elif kind == "arc":
      v0, v1, turn, frames = a[0], a[0], a[1], a[2]
    elif kind == "home":
      dx, dy = start[0] - x, start[1] - y
      h = math.atan2(dy, dx)
      frames = max(1, round(math.hypot(dx, dy) / a[0]))
      v0 = v1 = math.hypot(dx, dy) / frames  # land exactly on start
      turn = 0
    else:
      raise ValueError(f"unknown route command: {cmd}")
    for i in range(frames):
      speed = v0 + (v1 - v0) * i / frames
      h += math.radians(turn) / frames
      x += speed * math.cos(h)
      y += speed * math.sin(h)
      yield x, y, label


ROUTE_PATH = list(trace_route(ROUTE))
LABELS = [label for _, _, label in ROUTE_PATH]
PATH = np.array([(x, y) for x, y, _ in ROUTE_PATH])
MARGIN = VIEW / 2 + 2  # sensor patch (+1 px start jitter) must stay on surface
assert (PATH.min(0) >= MARGIN).all() and (PATH.max(0) <= (WIDTH - MARGIN, HEIGHT - MARGIN)).all(), \
    "route takes the sensor off the surface, adjust ROUTE"


def random_route(rng, commands=18):
  """Random route from the same command vocabulary, random start, closed with `home`.
  Redrawn until the sensor stays on the surface."""
  while True:
    cmds = []
    for _ in range(commands):
      kind = rng.choice(["go", "ramp", "arc", "turn", "wait"], p=[0.3, 0.15, 0.25, 0.2, 0.1])
      if kind == "go":
        cmds.append(("go", round(rng.uniform(0.3, 8.0), 1), int(rng.integers(5, 50))))
      elif kind == "ramp":
        cmds.append(("ramp", round(rng.uniform(0, 6), 1), round(rng.uniform(0, 6), 1), int(rng.integers(15, 50))))
      elif kind == "arc":
        cmds.append(("arc", round(rng.uniform(1, 4), 1), int(rng.integers(-270, 271)), int(rng.integers(20, 80))))
      elif kind == "turn":
        cmds.append(("turn", int(rng.choice([-135, -90, -45, 45, 90, 135, 180]))))
      else:
        cmds.append(("wait", int(rng.integers(5, 30))))
    start = (rng.uniform(150, WIDTH - 150), rng.uniform(150, HEIGHT - 150))
    cmds.append(("home", 4))
    path = np.array([(x, y) for x, y, _ in trace_route(cmds, start)])
    if (path.min(0) >= MARGIN).all() and (path.max(0) <= (WIDTH - MARGIN, HEIGHT - MARGIN)).all():
      return cmds, path


def route_at(t, path=PATH):
  """True position at fractional display frame t (route loops)."""
  i = int(t)
  a = t - i
  return path[i % len(path)] * (1 - a) + path[(i + 1) % len(path)] * a


# ==========================================
# 2. SURFACE + SENSOR + TRACKING ALGORITHM
# ==========================================
def gaussian_blur(img, sigma):
  """FFT gaussian blur (wraps at edges)."""
  fy = np.fft.fftfreq(img.shape[0])[:, None]
  fx = np.fft.fftfreq(img.shape[1])[None, :]
  return np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * (np.pi * sigma) ** 2 * (fx**2 + fy**2))).real


def make_surface(seed, blur=SURFACE_BLUR):
  """Every pixel random, lightly blurred like real surface grain."""
  s = gaussian_blur(np.random.default_rng(seed).random((HEIGHT, WIDTH)), blur)
  return ((s - s.min()) / np.ptp(s)).astype(np.float32)


surface = make_surface(SEED)
_SUB = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5
GRID = (np.arange(VIEW)[:, None] - VIEW / 2 + _SUB[None, :]).ravel()  # sub-sample offsets
HANN = np.outer(np.hanning(VIEW), np.hanning(VIEW)).astype(np.float32)
# Linear correlation: zero-pad by VIEW/2 so shifts up to VIEW/2 never wrap around
SIZE = VIEW + VIEW // 2 if CORRELATION == "linear" else VIEW
_u = (np.arange(SIZE) + SIZE // 2) % SIZE - SIZE // 2  # signed shift of each correlation bin
SEARCH = (np.abs(_u)[:, None] <= VIEW // 2) & (np.abs(_u)[None, :] <= VIEW // 2)  # valid peaks

# Tracker parameters, tuned by --optimize and saved to best_params.json:
#   whiten  correlation spectrum flattening: 0 = plain cross-correlation, 1 = phase correlation
#   window  edge taper on sensor image: 0 = none, 1 = full Hann
#   gain    scale correction on every measured shift (sensor "DPI" calibration)
#   gain3   cubic term of that calibration: shift * (gain + gain3 * |shift|^2)
#   rekey_q, search, loop, smooth: see Tracker
#   rekey   keyframe threshold, px: hold a reference frame until the shift from it
#           exceeds this, so errors add up per keyframe not per frame (0 = every frame)
BOUNDS = {"whiten": (0.0, 1.0), "window": (0.0, 1.0), "gain": (0.8, 1.3), "gain3": (-0.004, 0.004),
          "rekey": (0.0, 10.0), "rekey_q": (0.0, 12.0), "search": (0.0, 6.0), "loop": (0.0, 30.0),
          "smooth": (0.05, 1.0)}
DEFAULT_PARAMS = {"whiten": 0.0, "window": 1.0, "gain": 1.0, "gain3": 0.0, "rekey": 0.0, "rekey_q": 0.0,
                  "search": 0.0, "loop": 0.0, "smooth": 1.0}
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")  # all charts and README figures land here
PARAMS_FILE = os.path.join(HERE, "best_params.json")
HISTORY_FILE = os.path.join(HERE, "optimize_history.npz")  # lets --replot redraw without re-running
PARAMS = DEFAULT_PARAMS | (json.load(open(PARAMS_FILE)) if os.path.exists(PARAMS_FILE) else {})


def sense(pos, surf=surface, angle=0.0):
  """Clean sensor images (n, VIEW, VIEW) centred at float positions (n, 2).
  Each pixel averages SUPERSAMPLE^2 bilinear samples (area integration, like a photodiode);
  the pixel grid is rotated by `angle` radians."""
  gx, gy = GRID[None, None, :], GRID[None, :, None]
  if angle:
    c, s = math.cos(angle), math.sin(angle)
    gx, gy = c * gx - s * gy, s * gx + c * gy
  x = pos[:, 0, None, None] + gx
  y = pos[:, 1, None, None] + gy
  x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
  fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
  img = (surf[y0, x0] * (1 - fx) * (1 - fy) + surf[y0, x0 + 1] * fx * (1 - fy)
         + surf[y0 + 1, x0] * (1 - fx) * fy + surf[y0 + 1, x0 + 1] * fx * fy)
  return img.reshape(len(pos), VIEW, SUPERSAMPLE, VIEW, SUPERSAMPLE).mean((2, 4))


class Wobble:
  """Smooth random sensor rotation: mean-reverting random walk, stationary std SENSOR_WOBBLE_DEG."""

  def __init__(self, rng, rho=0.995):
    self.rng, self.rho, self.angle = rng, rho, 0.0
    self.step = math.radians(SENSOR_WOBBLE_DEG) * math.sqrt(1 - rho**2)

  def __call__(self):
    if self.step:
      self.angle = self.rho * self.angle + self.step * self.rng.standard_normal()
    return self.angle


def _per_run(p, name, ndim):
  """Parameter as a column broadcastable over n runs: scalar or one value per run."""
  return np.asarray(p[name], np.float32).reshape((-1,) + (1,) * ndim)


def spectrum(img, p):
  """Windowed FFT of sensor images, computed once per frame and reused."""
  w = _per_run(p, "window", 2)
  window = 1 - w + w * HANN
  return sfft.rfft2((img - img.mean((-2, -1), keepdims=True)) * window, s=(SIZE, SIZE))


@functools.lru_cache(maxsize=4096)
def _envelope(alpha):
  """Overlap envelope E(u) = window autocorrelation, peak 1. Dividing by it stops the
  shrinking overlap of shifted frames from pulling the peak toward zero."""
  window = 1 - alpha + alpha * HANN
  e = sfft.irfft2(np.abs(sfft.rfft2(window, s=(SIZE, SIZE))) ** 2, s=(SIZE, SIZE))
  return np.maximum(e / e.max(), 0.15).astype(np.float32)  # floor: tiny overlaps are noise


def envelope(p, rows):
  """Envelope per run (rows, SIZE, SIZE); computed once per distinct window value."""
  alphas = np.broadcast_to(np.asarray(p["window"], np.float32), (rows,))
  uniq, inv = np.unique(np.round(alphas, 6), return_inverse=True)
  return np.stack([_envelope(float(a)) for a in uniq])[inv]


def _parabola(left, mid, right):
  """Sub-pixel offset of a peak from its two neighbours. Clipped to +-0.5: beyond that the
  integer peak was not the maximum, and a flat neighbourhood would send the vertex to infinity."""
  return np.clip(0.5 * (left - right) / np.minimum(left - 2 * mid + right, -1e-12), -0.5, 0.5)


def track(key, cur, p, pred=None, quality=False):
  """Estimate sensor motion (n, 2) from frame spectrum `key` to `cur` by correlation.
  pred (n, 2): predicted shift; with p["search"] >= 1 the peak search is limited to that
  radius around it. quality=True also returns the peak-to-sidelobe ratio of each match."""
  r = key * np.conj(cur)
  whiten = _per_run(p, "whiten", 2)
  if whiten.any():
    r = r * np.exp(-whiten * np.log(np.abs(r) + np.float32(1e-12)))  # = r / |r|^whiten, 5x faster than **
  c = sfft.irfft2(r, s=(SIZE, SIZE))
  if CORRELATION == "linear":
    c = c / envelope(p, len(c))
  allowed = SEARCH
  radius = np.broadcast_to(_per_run(p, "search", 0), (len(c),))
  if pred is not None and (radius >= 1).any():
    radius = np.where(radius >= 1, radius, np.inf)[:, None, None]
    allowed = (SEARCH & (np.abs(_u[None, None, :] - pred[:, 0, None, None]) <= radius)
               & (np.abs(_u[None, :, None] - pred[:, 1, None, None]) <= radius))
  rows = np.arange(len(c))
  iy, ix = np.divmod(np.where(allowed, c, -np.inf).reshape(len(c), -1).argmax(1), SIZE)
  mid = c[rows, iy, ix]
  ox = _parabola(c[rows, iy, (ix - 1) % SIZE], mid, c[rows, iy, (ix + 1) % SIZE])
  oy = _parabola(c[rows, (iy - 1) % SIZE, ix], mid, c[rows, (iy + 1) % SIZE, ix])
  d = np.stack([_u[ix] + ox, _u[iy] + oy], 1)  # bin index -> signed shift
  d = d * (_per_run(p, "gain", 1) + _per_run(p, "gain3", 1) * (d**2).sum(1, keepdims=True))
  if not quality:
    return d
  side = c[:, SEARCH]
  return d, (mid - side.mean(1)) / (side.std(1) + 1e-12)


class Tracker:
  """Dead reckoning for n sensors: position = keyframe position + shift from the keyframe.
  The keyframe is replaced once the shift passes `rekey` px or the match quality
  (peak-to-sidelobe ratio) drops below `rekey_q`. Extras, each off at its default:
    search  limit the peak search to this radius around a constant-velocity prediction
    loop    when the estimate comes back within this many px of an old anchor frame and the
            frames match, snap to the anchor's position (loop closure)
    smooth  < 1: alpha-beta filter on the reported position (alpha = smooth)"""

  ANCHOR_EVERY = 80  # sensor frames between stored anchors
  ANCHOR_AGE = 400  # an anchor must be this old before it can close a loop
  LOOP_QUALITY = 6.0  # min peak-to-sidelobe ratio for a loop match (unrelated frames: p95 ~4.2 when whitened)

  def __init__(self, img, start, p):
    n = len(img)
    self.p, self.k, self.anchors = p, 0, []
    self.key = spectrum(img, p)
    self.key_pos = np.repeat(np.asarray(start, float)[None], n, 0)
    self.raw, self.vel = self.key_pos.copy(), np.zeros((n, 2))
    self.pos, self.fvel = self.key_pos.copy(), np.zeros((n, 2))
    self.loop = np.broadcast_to(np.asarray(p.get("loop", 0.0), float), (n,))
    if self.loop.any():
      self.anchors.append((0, self.key.copy(), self.key_pos.copy()))  # the start is the first anchor

  def update(self, img):
    p = self.p
    self.k += 1
    cur = spectrum(img, p)
    shift, q = track(self.key, cur, p, pred=self.raw + self.vel - self.key_pos, quality=True)
    raw = self.key_pos + shift
    rekey = (np.hypot(*shift.T) >= np.asarray(p["rekey"])) | (q < np.asarray(p.get("rekey_q", 0.0)))
    if self.loop.any():
      raw, rekey = self._close_loops(cur, raw, rekey)
      if self.k % self.ANCHOR_EVERY == 0:
        self.anchors.append((self.k, cur.copy(), raw.copy()))
    self.vel = 0.5 * (self.vel + raw - self.raw)  # ponytail: fixed EMA velocity, a Kalman gain if it matters
    self.raw = raw
    self.key[rekey], self.key_pos[rekey] = cur[rekey], raw[rekey]
    self._smooth()
    return self.pos

  def _close_loops(self, cur, raw, rekey):
    old = [a for a in self.anchors if self.k - a[0] >= self.ANCHOR_AGE]
    if not old:
      return raw, rekey
    dist = np.hypot(*(np.stack([a[2] for a in old], 1) - raw[:, None]).transpose(2, 0, 1))  # (n, anchors)
    near = dist.argmin(1)
    rows = np.flatnonzero(dist[np.arange(len(raw)), near] < self.loop)
    if not len(rows):
      return raw, rekey
    sub = {k: (np.asarray(v)[rows] if np.ndim(v) else v) for k, v in self.p.items()}
    spec = np.stack([old[j][1][r] for r, j in zip(rows, near[rows])])
    d, q = track(spec, cur[rows], sub, quality=True)
    ok = q >= self.LOOP_QUALITY
    if not ok.any():
      return raw, rekey
    rows, d = rows[ok], d[ok]
    raw, rekey = raw.copy(), rekey.copy()
    raw[rows] = np.stack([old[j][2][r] for r, j in zip(rows, near[rows])]) + d
    rekey[rows] = True
    return raw, rekey

  def _smooth(self):
    a = np.asarray(self.p.get("smooth", 1.0), float).reshape(-1, 1)
    if (a >= 1).all():
      self.pos = self.raw
      return
    predicted = self.pos + self.fvel
    resid = self.raw - predicted
    self.pos = np.where(a >= 1, self.raw, predicted + a * resid)
    self.fvel = self.fvel + a**2 / (2 - a) * resid  # critically damped beta


# ==========================================
# 3. BATCH: DRIFT OVER TIME, MANY RUNS
# ==========================================
def run_chunk(task):
  """n runs sharing one texture + start phase, differing only by sensor noise.
  task keys: seed, n, laps, stride, p, density, reps; optional path (route), blur.
  reps > 1 repeats those n runs for reps parameter sets (p values given per run, length
  reps * n) with identical noise, so candidates are compared on exactly the same data.
  Returns dict: drift (samples, reps*n), final (reps*n, 2), hist or None,
  err_sum (frames, 2) and err_sq (frames,): per-frame sums over runs for error attribution."""
  n, reps, laps, stride, p = task["n"], task.get("reps", 1), task["laps"], task["stride"], task["p"]
  path = task.get("path", PATH)
  rng = np.random.default_rng(task["seed"])
  surf = make_surface(task["seed"], task.get("blur", SURFACE_BLUR))
  offset = rng.random(2)  # sub-pixel start phase
  wobble = Wobble(rng)
  frames = len(path) * laps
  noisy = lambda t: sense(route_at(t, path)[None] + offset, surf, wobble()) + np.tile(
      SENSOR_NOISE * rng.standard_normal((n, VIEW, VIEW), np.float32), (reps, 1, 1))
  drift = np.zeros((math.ceil(frames / stride), reps * n), np.float32)
  err_sum, err_sq = np.zeros((frames, 2)), np.zeros(frames)
  hist = np.zeros((HEIGHT, WIDTH), np.float32) if task.get("density") else None
  tracker = Tracker(noisy(0), path[0], p)
  for k in range(1, (frames - 1) * SUBSTEPS + 1):
    pos = tracker.update(noisy(k / SUBSTEPS))
    f, sub = divmod(k, SUBSTEPS)
    if sub:
      continue
    if hist is not None:
      ix, iy = pos.astype(int).T
      ok = (ix >= 0) & (ix < WIDTH) & (iy >= 0) & (iy < HEIGHT)
      np.add.at(hist, (iy[ok], ix[ok]), 1)
    err = pos - path[f % len(path)]
    err_sum[f], err_sq[f] = err.sum(0), (err**2).sum()
    if f % stride == 0:
      drift[f // stride] = np.hypot(*err.T)
  return {"drift": drift, "final": tracker.pos - path[(frames - 1) % len(path)], "hist": hist,
          "err_sum": err_sum, "err_sq": err_sq}


def run_batch(runs=RUNS, laps=LAPS, chunk=CHUNK, seed=None, p=None):
  """Returns (sample frames, drift (samples, runs), final error (runs, 2), position histogram,
  (mean error (frames, 2), error spread (frames,)) across runs for attribution)."""
  p = p or PARAMS
  stride = max(1, len(PATH) * laps // 1000)  # keep ~1000 time samples
  seeds = np.random.default_rng(seed).integers(2**32, size=math.ceil(runs / chunk))
  tasks = [{"seed": int(s), "n": min(chunk, runs - i * chunk), "laps": laps, "stride": stride, "p": p,
            "density": True} for i, s in enumerate(seeds)]
  parts, t0 = [], time.time()
  with multiprocessing.Pool() as pool:
    for r in pool.imap(run_chunk, tasks):
      parts.append(r)
      print(f"  {sum(len(q['final']) for q in parts)}/{runs} runs  ({time.time() - t0:.0f}s)")
  drift = np.concatenate([r["drift"] for r in parts], 1)
  err_mean = sum(r["err_sum"] for r in parts) / runs
  spread = np.sqrt(np.maximum(sum(r["err_sq"] for r in parts) / runs - (err_mean**2).sum(1), 0))
  return (np.arange(len(drift)) * stride, drift, np.concatenate([r["final"] for r in parts]),
          sum(r["hist"] for r in parts), (err_mean, spread))


def make_cases(rng, fixed, random):
  """Test cases: `fixed` textures on the README route, plus `random` random routes, each on a
  surface with random grain (SURFACE_BLUR 0.5-3 px)."""
  cases = [{"seed": int(rng.integers(2**32))} for _ in range(fixed)]
  for _ in range(random):
    cases.append({"seed": int(rng.integers(2**32)), "path": random_route(rng)[1],
                  "blur": float(rng.uniform(0.5, 3.0))})
  return cases


def evaluate(pool, pop, names, cases, runs, per_case=False):
  """Score each candidate row of pop (values for `names`; other params default) on the cases:
  RMS of drift per 1000 px travelled, each case weighted equally. One task per case, all
  candidates inside it. per_case=True returns the (candidates, cases) matrix instead."""
  fixed = {k: v for k, v in DEFAULT_PARAMS.items() if k not in names}
  p = fixed | {k: np.repeat(pop[:, j], runs) for j, k in enumerate(names)}
  tasks = [{"n": runs, "laps": 1, "stride": 1, "p": p, "reps": len(pop)} | c for c in cases]
  out = np.zeros((len(pop), len(cases)))
  for i, r in enumerate(pool.imap(run_chunk, tasks)):
    path = cases[i].get("path", PATH)
    travelled = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    ok = travelled > 50  # the ratio is all noise over the first few px
    d = r["drift"].reshape(len(r["drift"]), len(pop), runs)[ok].astype(np.float64)
    out[:, i] = np.sqrt(((1000 * d / travelled[ok, None, None]) ** 2).mean((0, 2)))
  out = np.nan_to_num(out, nan=1e9, posinf=1e9)
  return out if per_case else np.sqrt((out**2).mean(1))


def optimize(seed=0):
  """Cross-entropy search over BOUNDS, then a batched pattern-search polish.
  Score = RMS drift per 1000 px travelled, over the README route on several textures plus
  random routes on random surfaces. Every candidate sees the same textures + noise
  (common random numbers), so score differences come from the parameters, not luck.
  A held-out set of textures is scored for the best candidate each generation."""
  names = list(BOUNDS)
  lo, hi = np.array([BOUNDS[k] for k in names]).T
  rng = np.random.default_rng(seed)
  cases = make_cases(rng, OPT_TEXTURES // 2, OPT_TEXTURES // 2)
  val_cases = make_cases(rng, OPT_TEXTURES // 4, OPT_TEXTURES // 4)  # never used for selection
  mean, std = (lo + hi) / 2, (hi - lo) / 4
  history, best, t0 = [], (np.inf, None), time.time()
  shown = lambda x: ", ".join(f"{k}={v:.3f}" for k, v in zip(names, x))
  with multiprocessing.Pool() as pool:
    for g in range(OPT_GENERATIONS):
      pop = np.clip(rng.normal(mean, std, (OPT_POP, len(names))), lo, hi)
      if g == 0:
        pop[0] = [DEFAULT_PARAMS[k] for k in names]  # baseline: the untuned tracker
      scores = evaluate(pool, pop, names, cases, OPT_RUNS)
      if g == 0:
        baseline = scores[0]
      if scores.min() < best[0]:
        best = (scores.min(), pop[scores.argmin()])
      elite = pop[np.argsort(scores)[:OPT_POP // 4]]
      mean, std = elite.mean(0), np.maximum(elite.std(0), (hi - lo) * 0.02)
      val = evaluate(pool, best[1][None], names, val_cases, OPT_RUNS)[0]
      history.append((pop, scores, val))
      print(f"  gen {g + 1}/{OPT_GENERATIONS}: best {best[0]:.2f}, held-out {val:.2f} "
            f"({shown(best[1])})  ({time.time() - t0:.0f}s)")

    # Polish: probe +-step on every axis in one batch; move to the best probe, else halve the step
    step = (hi - lo) * 0.05
    for i in range(OPT_POLISH):
      probes = np.clip(best[1] + np.vstack([np.diag(step), -np.diag(step)]), lo, hi)
      s = evaluate(pool, probes, names, cases, OPT_RUNS)
      if s.min() < best[0]:
        best = (s.min(), probes[s.argmin()])
      else:
        step = step / 2
      print(f"  polish {i + 1}/{OPT_POLISH}: best {best[0]:.2f}px ({shown(best[1])})  ({time.time() - t0:.0f}s)")
    val = evaluate(pool, best[1][None], names, val_cases, OPT_RUNS)[0]
  params = DEFAULT_PARAMS | dict(zip(names, map(float, best[1])))
  with open(PARAMS_FILE, "w") as f:
    json.dump(params, f, indent=2)
  print(f"Baseline {baseline:.2f} -> best {best[0]:.2f} px per 1000 px (held-out {val:.2f}). Saved {PARAMS_FILE}")
  np.savez(HISTORY_FILE, pops=np.stack([h[0] for h in history]), scores=np.stack([h[1] for h in history]),
           vals=np.array([h[2] for h in history]), final_val=val, baseline=baseline, names=names, lo=lo, hi=hi)
  return history, baseline, names, lo, hi


def robustness(candidates, seed=11):
  """Each candidate (dict of params) on ROBUST_ROUTES random routes, each on its own surface.
  Returns grain (routes,) and drift per 1000 px travelled (candidates, routes)."""
  cases = make_cases(np.random.default_rng(seed), 0, ROBUST_ROUTES)
  names = list(DEFAULT_PARAMS)
  pop = np.array([[c[k] for k in names] for c in candidates])
  with multiprocessing.Pool() as pool:
    out = evaluate(pool, pop, names, cases, ROBUST_RUNS, per_case=True)
  return np.array([c["blur"] for c in cases]), out


SWEEP_VIEWS = (16, 24, 32)  # sweep: sensor sizes, px
SWEEP_SUBSTEPS = (2, 4, 8)  # sweep: sensor frames per display frame


def score(seed=5):
  """PARAMS on a fixed mixed case set (README route + random routes), px per 1000 px.
  Used by --sweep, one process per sensor configuration."""
  cases = make_cases(np.random.default_rng(seed), 8, 8)
  names = list(DEFAULT_PARAMS)
  with multiprocessing.Pool() as pool:
    return float(evaluate(pool, np.array([[PARAMS[k] for k in names]]), names, cases, 5)[0])


def hardware_sweep():
  """Score the tuned tracker for every VIEW x SUBSTEPS pair, each in its own process
  (the knobs are read at import). Returns the (views, substeps) score grid."""
  import subprocess
  grid = np.zeros((len(SWEEP_VIEWS), len(SWEEP_SUBSTEPS)))
  for i, view in enumerate(SWEEP_VIEWS):
    for j, sub in enumerate(SWEEP_SUBSTEPS):
      env = os.environ | {"MOUSESIM_VIEW": str(view), "MOUSESIM_SUBSTEPS": str(sub)}
      out = subprocess.run([sys.executable, os.path.abspath(__file__), "--score"], env=env,
                           capture_output=True, text=True, check=True).stdout
      grid[i, j] = float(out.strip().splitlines()[-1])
      print(f"  VIEW {view:2}  SUBSTEPS {sub}: {grid[i, j]:.2f} px per 1000 px")
  np.save(os.path.join(HERE, "sweep.npy"), grid)
  return grid


def load_history():
  """optimize()'s return value, read back from HISTORY_FILE."""
  h = np.load(HISTORY_FILE)
  vals = h["vals"] if "vals" in h else [np.nan] * len(h["pops"])
  return list(zip(h["pops"], h["scores"], vals)), float(h["baseline"]), list(h["names"]), h["lo"], h["hi"]


# ==========================================
# 4. CHARTS (matplotlib, classic MATLAB look)
# ==========================================
MATLAB_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
# MATLAB's default colormap (parula), approximated from its published anchor colours
PARULA_ANCHORS = [(0.2422, 0.1504, 0.6603), (0.2810, 0.3228, 0.9579), (0.1786, 0.5289, 0.9682),
                  (0.0689, 0.6948, 0.8394), (0.2161, 0.7843, 0.5923), (0.6720, 0.7793, 0.2227),
                  (0.9970, 0.7659, 0.2199), (0.9769, 0.9839, 0.0805)]


def matlab_style():
  """pyplot with white background, black boxed axes, inward ticks, MATLAB line colours."""
  import matplotlib.pyplot as plt  # imported here so batch workers stay light
  plt.rcParams.update({
      "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
      "axes.edgecolor": "black", "axes.labelcolor": "black", "text.color": "black",
      "xtick.color": "black", "ytick.color": "black",
      "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
      "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5,
      "axes.prop_cycle": plt.cycler(color=MATLAB_COLORS), "lines.linewidth": 1.5,
      "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
      "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1,
  })
  return plt


def save(fig, name):
  os.makedirs(DOCS, exist_ok=True)
  fig.savefig(os.path.join(DOCS, name), dpi=150)
  print(f"Saved docs/{name}")


def plot_batch(samples, drift, final):
  """Drift-vs-time band chart + final error scatter. Saves drift.png."""
  plt = matlab_style()
  p5, p50, p95 = np.percentile(drift, [5, 50, 95], axis=1)
  fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=[2, 1])
  ax.fill_between(samples, p5, p95, color=MATLAB_COLORS[0], alpha=0.2, lw=0, label="5-95%")
  ax.plot(samples, p50, color=MATLAB_COLORS[0], label="median")
  ax.plot(samples, drift.mean(1), color=MATLAB_COLORS[1], ls="--", label="mean")
  for lap in range(1, samples[-1] // len(PATH) + 1):  # lap boundaries
    ax.axvline(lap * len(PATH), color="0.5", lw=0.5, ls=":")
  ax.set(xlabel="Frame", ylabel="Drift (px)", xlim=(0, samples[-1]), ylim=(0, None),
         title=f"Drift over time, {drift.shape[1]} runs  ({params_text()})")
  ax.legend(loc="upper left")

  # Final error vectors: cloud off-centre = systematic bias, wide = random walk
  mean = final.mean(0)
  ax2.scatter(final[:, 0], final[:, 1], s=3, color=MATLAB_COLORS[0], alpha=0.3, lw=0)
  ax2.plot(*mean, "x", color=MATLAB_COLORS[1], ms=12, mew=2.5,
           label=f"mean ({mean[0]:+.1f}, {mean[1]:+.1f})")
  ax2.axhline(0, color="black", lw=0.8)
  ax2.axvline(0, color="black", lw=0.8)
  ax2.set(xlabel="Error x (px)", ylabel="Error y (px)", aspect="equal",
          title=f"Final error: median {p50[-1]:.1f}px, p95 {p95[-1]:.1f}px")
  ax2.invert_yaxis()  # screen coordinates, y down
  ax2.legend(loc="upper right")
  fig.tight_layout()
  save(fig, "drift.png")


def plot_route():
  """Route diagram: each command coloured and numbered, arrows show direction. Saves route.png."""
  plt = matlab_style()
  fig, (ax, key) = plt.subplots(1, 2, figsize=(13, 6.5), width_ratios=[3, 1])
  labels = np.array(LABELS)
  starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
  for k, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
    seg = np.vstack([PATH[a - 1] if a else START, PATH[a:b]])
    color = MATLAB_COLORS[k % len(MATLAB_COLORS)]
    ax.plot(*seg.T, color=color, lw=2)
    mid = len(seg) // 2
    d = seg[min(mid + 1, len(seg) - 1)] - seg[max(mid - 1, 0)]
    if np.hypot(*d) > 1e-6:  # direction arrow (none for 'wait')
      ax.annotate("", seg[mid] + d * 0.01, seg[mid] - d / np.hypot(*d) * 12,
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    ax.annotate(labels[a].split(".")[0], seg[mid], xytext=(6, 6), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color)
  ax.plot(*START, "o", color="black", ms=7, label="start / end")
  ax.add_patch(plt.Rectangle((START[0] - VIEW / 2, START[1] - VIEW / 2), VIEW, VIEW,
                             fill=False, color="black", lw=1, label=f"sensor view {VIEW}px"))
  ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0), aspect="equal", xlabel="x (px)", ylabel="y (px)",
         title=f"Test route: {len(PATH)} frames per lap, {SUBSTEPS} sensor frames each")
  ax.legend(loc="lower left")
  key.axis("off")
  key.text(0, 1, "\n".join(f"{i}. {' '.join(map(str, c))}" for i, c in enumerate(ROUTE, 1)),
           va="top", family="monospace", fontsize=9, transform=key.transAxes)
  fig.tight_layout()
  save(fig, "route.png")


def params_text(p=None):
  return ", ".join(f"{k} {v:.2f}" for k, v in (p or PARAMS).items())


def plot_density(hist, p=None, name="density.png"):
  """Gaussian density (KDE via histogram + blur) of every run's believed position, over the true route."""
  plt = matlab_style()
  from matplotlib.colors import LinearSegmentedColormap, PowerNorm
  density = np.maximum(gaussian_blur(hist, DENSITY_BLUR), 0)
  density /= density.sum()
  cmap = LinearSegmentedColormap.from_list("white_blue", ["white", MATLAB_COLORS[5], MATLAB_COLORS[0], "#00294A"])
  fig, ax = plt.subplots(figsize=(10, 7.5))
  im = ax.imshow(density, cmap=cmap, norm=PowerNorm(0.35, vmin=0), extent=(0, WIDTH, HEIGHT, 0),
                 interpolation="bilinear")
  ax.plot(*np.vstack([START, PATH]).T, color=MATLAB_COLORS[1], lw=1, label="true route")
  ax.plot(*START, "o", color="black", ms=6, label="start / end")
  ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0), aspect="equal", xlabel="x (px)", ylabel="y (px)",
         title=f"Believed position density, all runs, every frame  ({params_text(p)})")
  ax.grid(False)
  ax.legend(loc="lower left")
  fig.colorbar(im, ax=ax, shrink=0.8, label="probability density (gamma 0.35)")
  fig.tight_layout()
  save(fig, name)


def route_segments():
  """(command label, first frame, end frame) for each moving route command."""
  labels = np.array(LABELS)
  starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
  return [(labels[a], a, b) for a, b in zip(starts[:-1], starts[1:])]


def plot_attribution(err_mean, spread, p=None, name="attribution.png"):
  """Where drift comes from: per-command change in mean error (shared by all runs: bias)
  and in spread across runs (texture + noise). Saves attribution.png."""
  plt = matlab_style()
  bias = np.hypot(*err_mean.T)
  segs = route_segments()
  fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[1, 1.1])
  for k, (_, a, b) in enumerate(segs):
    if k % 2:
      ax.axvspan(a, b, color="0.94", lw=0)
  ax.plot(bias, color=MATLAB_COLORS[1], label=r"bias: $\|\mathrm{mean\ error}\|$")
  ax.plot(spread, color=MATLAB_COLORS[0], label="spread: RMS distance from the mean")
  ax.set(xlim=(0, len(bias) - 1), ylim=(0, None), xlabel="Frame", ylabel="px",
         title=f"Error over one lap, split into shared bias and run-to-run spread  ({params_text(p)})")
  ax.legend(loc="upper left")

  idx = np.arange(len(segs))
  end = lambda arr, b: arr[min(b, len(arr)) - 1]
  start = lambda arr, a: arr[max(a - 1, 0)]
  d_bias = [end(bias, b) - start(bias, a) for _, a, b in segs]
  d_spread = [end(spread, b) - start(spread, a) for _, a, b in segs]
  ax2.bar(idx - 0.2, d_bias, 0.4, color=MATLAB_COLORS[1], label="bias added")
  ax2.bar(idx + 0.2, d_spread, 0.4, color=MATLAB_COLORS[0], label="spread added")
  ax2.axhline(0, color="black", lw=0.8)
  ax2.set_xticks(idx, [s[0] for s in segs], rotation=60, ha="right", fontsize=8)
  ax2.set(ylabel="px added during command", title="Contribution of each route command")
  ax2.legend(loc="upper left")
  fig.tight_layout()
  save(fig, name)


def plot_robustness(blurs, out, labels):
  """Drift per 1000 px vs surface grain for each candidate on random routes. Saves robustness.png."""
  plt = matlab_style()
  fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[2, 1])
  for j, (row, label) in enumerate(zip(out, labels)):
    ax.plot(blurs, row, "o", color=MATLAB_COLORS[j], ms=5, alpha=0.8,
            label=f"{label}: median {np.median(row):.2f} px / 1000 px")
  ax.set(xlabel=r"surface grain $\sigma_s$ (px)", ylabel="RMS drift per 1000 px travelled (px)", yscale="log",
         title=f"{out.shape[1]} random routes, each on its own surface")
  ax.legend(loc="upper right")
  ax2.boxplot(list(out), tick_labels=labels, widths=0.5, medianprops=dict(color=MATLAB_COLORS[1]))
  ax2.set(yscale="log", title="all routes")
  from matplotlib.ticker import ScalarFormatter
  for a in (ax, ax2):
    a.yaxis.set_major_formatter(ScalarFormatter())
  fig.tight_layout()
  save(fig, "robustness.png")


def plot_sweep(grid):
  """Heatmap of drift per 1000 px over sensor size x sensor rate. Saves sweep.png."""
  plt = matlab_style()
  from matplotlib.colors import LogNorm
  fig, ax = plt.subplots(figsize=(7, 5.5))
  from matplotlib.colors import LinearSegmentedColormap
  parula_r = LinearSegmentedColormap.from_list("parula_r", PARULA_ANCHORS[::-1])
  im = ax.imshow(grid, cmap=parula_r, norm=LogNorm(), origin="lower", aspect="auto")
  for (i, j), v in np.ndenumerate(grid):
    ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > np.sqrt(grid.min() * grid.max()) else "black")
  ax.set_xticks(range(len(SWEEP_SUBSTEPS)), [f"{s} ({60 * s} Hz)" for s in SWEEP_SUBSTEPS])
  ax.set_yticks(range(len(SWEEP_VIEWS)), [f"{v} px" for v in SWEEP_VIEWS])
  ax.set(xlabel="sensor frames per display frame (sensor rate)", ylabel="sensor size",
         title="Drift per 1000 px travelled, tuned tracker")
  ax.grid(False)
  fig.colorbar(im, ax=ax, label="px per 1000 px")
  fig.tight_layout()
  save(fig, "sweep.png")


def plot_optimize(history, baseline, names, lo, hi):
  """Score per candidate per generation + elite parameter trajectories. Saves optimize.png."""
  plt = matlab_style()
  fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
  gens = np.arange(1, len(history) + 1)
  for g, (_, scores, _) in zip(gens, history):
    ok = np.isfinite(scores)
    ax.plot(np.full(ok.sum(), g), scores[ok], ".", color="0.6", ms=4)
  best = np.minimum.accumulate([s.min() for _, s, _ in history])
  vals = np.array([v for _, _, v in history], float)
  ax.plot(gens, best, "o-", color=MATLAB_COLORS[0], label=f"best so far ({best[-1]:.2f})")
  if np.isfinite(vals).any():
    ax.plot(gens, vals, "s--", color=MATLAB_COLORS[4], ms=5, label=f"best, held-out cases ({vals[-1]:.2f})")
  ax.axhline(baseline, color=MATLAB_COLORS[1], ls="--", label=f"untuned baseline ({baseline:.2f})")
  ax.set(xlabel="Generation", ylabel="RMS drift per 1000 px travelled", yscale="log",
         title=f"Optimization: {OPT_POP} candidates x {OPT_TEXTURES * OPT_RUNS} runs per generation")
  ax.legend(loc="upper right")
  for j, name in enumerate(names):
    elite = [pop[np.argsort(s)[:OPT_POP // 4], j].mean() for pop, s, _ in history]
    ax2.plot(gens, (np.array(elite) - lo[j]) / (hi[j] - lo[j]), "o-" if j < len(MATLAB_COLORS) else "s--",
             color=MATLAB_COLORS[j % len(MATLAB_COLORS)],
             label=f"{name} [{lo[j]:g}, {hi[j]:g}] -> {elite[-1]:.3f}")
  ax2.set(xlabel="Generation", ylabel="Elite mean, scaled to bounds", ylim=(-0.62, 1.05),
          yticks=np.linspace(0, 1, 6), title="Parameter convergence")
  ax2.legend(loc="lower center", ncol=2, fontsize=8)  # lives in the reserved band below 0
  from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
  for a in (ax, ax2):
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.yaxis.set_major_formatter(ScalarFormatter())
  ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
  ax.yaxis.set_minor_formatter(NullFormatter())
  fig.tight_layout()
  save(fig, "optimize.png")


# ==========================================
# 5. LIVE VIEW (pygame)
# ==========================================
ACTUAL_COLOR = (0, 114, 189)  # MATLAB blue: ground truth path
BELIEVED_COLOR = (217, 83, 25)  # MATLAB orange: where the mouse thinks it is
TEXT_COLOR = (0, 0, 0)


def live(screen, font, clock):
  rng = np.random.default_rng(4316)
  gray = (175 + surface.T * 80).astype(np.uint8)  # texture kept light, white-ish page
  background = pygame.surfarray.make_surface(np.dstack([gray] * 3))
  noisy = lambda t: sense(route_at(t)[None]) + rng.normal(0, SENSOR_NOISE, (1, VIEW, VIEW))
  tracker = Tracker(noisy(0), PATH[0], PARAMS)
  actual_history, believed_history = [], []
  frame = 0
  while not any(e.type == pygame.QUIT for e in pygame.event.get()):
    for sub in range(1, SUBSTEPS + 1):
      img = noisy(frame + sub / SUBSTEPS)
      believed = tracker.update(img)[0]
    frame += 1
    true = PATH[frame % len(PATH)]

    actual_history = (actual_history + [tuple(true)])[-150:]
    believed_history = (believed_history + [tuple(believed)])[-150:]

    screen.blit(background, (0, 0))
    if len(actual_history) > 1:
      pygame.draw.lines(screen, ACTUAL_COLOR, False, actual_history, 3)
      pygame.draw.lines(screen, BELIEVED_COLOR, False, believed_history, 2)
    pygame.draw.rect(screen, TEXT_COLOR, (true[0] - VIEW / 2, true[1] - VIEW / 2, VIEW, VIEW), 1)
    pygame.draw.circle(screen, BELIEVED_COLOR, believed, 5)

    # Zoomed inset of what the sensor sees this frame
    patch = (np.clip(img[0].T, 0, 1) * 255).astype(np.uint8)
    inset = pygame.transform.scale(pygame.surfarray.make_surface(np.dstack([patch] * 3)), (120, 120))
    screen.blit(inset, (WIDTH - 130, 10))
    pygame.draw.rect(screen, TEXT_COLOR, (WIDTH - 130, 10, 120, 120), 1)

    drift = np.hypot(*(believed - true))
    lap = frame // len(PATH) + 1
    for i, s in enumerate([LABELS[frame % len(PATH)], f"lap {lap}  drift {drift:.2f}px"]):
      screen.blit(font.render(s, True, TEXT_COLOR), (10, 10 + 22 * i))
    for i, (color, s) in enumerate([(ACTUAL_COLOR, "true path"), (BELIEVED_COLOR, "believed path")]):
      y = HEIGHT - 50 + 22 * i
      pygame.draw.line(screen, color, (10, y + 8), (40, y + 8), 3)
      screen.blit(font.render(s, True, TEXT_COLOR), (48, y))

    pygame.display.flip()
    clock.tick(60)


if __name__ == "__main__":
  if "--route" in sys.argv:
    plot_route()
  elif "--optimize" in sys.argv and "--replot" in sys.argv:
    plot_optimize(*load_history())
  elif "--optimize" in sys.argv:
    print(f"Optimizing {list(BOUNDS)}: {OPT_GENERATIONS} gens x {OPT_POP} candidates "
          f"x {OPT_TEXTURES * OPT_RUNS} runs...")
    plot_optimize(*optimize())
  elif "--score" in sys.argv:
    print(score())
  elif "--sweep" in sys.argv and "--replot" in sys.argv:
    plot_sweep(np.load(os.path.join(HERE, "sweep.npy")))
  elif "--sweep" in sys.argv:
    print(f"Hardware sweep: VIEW {SWEEP_VIEWS} x SUBSTEPS {SWEEP_SUBSTEPS}")
    plot_sweep(hardware_sweep())
  elif "--robust" in sys.argv:
    print(f"Robustness: {ROBUST_ROUTES} random routes x {ROBUST_RUNS} runs, untuned vs tuned")
    plot_robustness(*robustness([DEFAULT_PARAMS, PARAMS]), ["untuned", "tuned"])
  elif "--batch" in sys.argv:
    print(f"Simulating {RUNS} runs x {LAPS} laps ({len(PATH) * LAPS} frames each), {params_text()}")
    samples, drift, final, hist, attribution = run_batch()
    plot_attribution(*attribution)
    plot_batch(samples, drift, final)
    plot_density(hist)
  if any(f in sys.argv for f in ("--route", "--optimize", "--batch", "--robust", "--sweep")):
    matlab_style().show()
  else:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Optical Mouse Tracking: Actual vs Believed")
    live(screen, pygame.font.SysFont(None, 24), pygame.time.Clock())
    pygame.quit()
