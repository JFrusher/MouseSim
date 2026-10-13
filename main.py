"""Optical mouse drift simulator.

python main.py             live view of one sensor tracking the route
python main.py --batch     RUNS x LAPS Monte Carlo: drift charts + density overlay
python main.py --route     diagram of the test route
python main.py --optimize  tune tracker parameters, saves best_params.json
python main.py --optimize --replot   redraw the optimizer chart from saved history
"""
import json
import math
import multiprocessing
import os
import sys
import time
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # quiet in batch worker processes
import pygame

# ==========================================
# 1. KNOBS
# ==========================================
WIDTH, HEIGHT = 800, 600
START = (100.0, 100.0)
VIEW = 24  # sensor viewport size, pixels per side
SUBSTEPS = 4  # sensor frames per display frame (sensor rate / display rate)
SURFACE_BLUR = 1.0  # texture grain (gaussian sigma, px); 0 = pure white noise
SENSOR_NOISE = 0.02  # per-frame read noise (surface values are 0..1)
SEED = 4316  # surface texture seed for live view
RUNS = 10000  # batch: independent repeats of the route
LAPS = 1  # batch: laps per run, drift keeps accumulating across laps
CHUNK = 500  # batch: runs per worker task; each chunk gets its own texture
OPT_GENERATIONS = 8  # optimize: cross-entropy generations
OPT_POP = 16  # optimize: candidate parameter sets per generation
OPT_TEXTURES = 32  # optimize: textures each candidate is scored on (same for all)
OPT_RUNS = 5  # optimize: runs per texture -> OPT_TEXTURES * OPT_RUNS runs per candidate
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


def route_at(t):
  """True position at fractional display frame t (route loops)."""
  i = int(t)
  a = t - i
  return PATH[i % len(PATH)] * (1 - a) + PATH[(i + 1) % len(PATH)] * a


# ==========================================
# 2. SURFACE + SENSOR + TRACKING ALGORITHM
# ==========================================
def gaussian_blur(img, sigma):
  """FFT gaussian blur (wraps at edges)."""
  fy = np.fft.fftfreq(img.shape[0])[:, None]
  fx = np.fft.fftfreq(img.shape[1])[None, :]
  return np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * (np.pi * sigma) ** 2 * (fx**2 + fy**2))).real


def make_surface(seed):
  """Every pixel random, lightly blurred like real surface grain."""
  s = gaussian_blur(np.random.default_rng(seed).random((HEIGHT, WIDTH)), SURFACE_BLUR)
  return ((s - s.min()) / np.ptp(s)).astype(np.float32)


surface = make_surface(SEED)
GRID = np.arange(VIEW) - VIEW / 2  # pixel offsets across the viewport
HANN = np.outer(np.hanning(VIEW), np.hanning(VIEW)).astype(np.float32)

# Tracker parameters, tuned by --optimize and saved to best_params.json:
#   whiten  correlation spectrum flattening: 0 = plain cross-correlation, 1 = phase correlation
#   window  edge taper on sensor image: 0 = none, 1 = full Hann
#   gain    scale correction on every measured shift (sensor "DPI" calibration)
#   rekey   keyframe threshold, px: hold a reference frame until the shift from it
#           exceeds this, so errors add up per keyframe not per frame (0 = every frame)
BOUNDS = {"whiten": (0.0, 1.0), "window": (0.0, 1.0), "gain": (0.8, 1.3), "rekey": (0.0, 8.0)}
DEFAULT_PARAMS = {"whiten": 0.0, "window": 1.0, "gain": 1.0, "rekey": 0.0}
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")  # all charts and README figures land here
PARAMS_FILE = os.path.join(HERE, "best_params.json")
HISTORY_FILE = os.path.join(HERE, "optimize_history.npz")  # lets --replot redraw without re-running
PARAMS = DEFAULT_PARAMS | (json.load(open(PARAMS_FILE)) if os.path.exists(PARAMS_FILE) else {})


def sense(pos, surf=surface):
  """Clean sensor images (n, VIEW, VIEW) centred at float positions (n, 2)."""
  x = pos[:, 0, None, None] + GRID[None, None, :]
  y = pos[:, 1, None, None] + GRID[None, :, None]
  x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
  fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
  return (surf[y0, x0] * (1 - fx) * (1 - fy) + surf[y0, x0 + 1] * fx * (1 - fy)
          + surf[y0 + 1, x0] * (1 - fx) * fy + surf[y0 + 1, x0 + 1] * fx * fy)


def _per_run(p, name, ndim):
  """Parameter as a column broadcastable over n runs: scalar or one value per run."""
  return np.asarray(p[name], np.float32).reshape((-1,) + (1,) * ndim)


def spectrum(img, p):
  """Windowed FFT of sensor images, computed once per frame and reused."""
  w = _per_run(p, "window", 2)
  window = 1 - w + w * HANN
  return np.fft.rfft2((img - img.mean((-2, -1), keepdims=True)) * window)


def _parabola(left, mid, right):
  """Sub-pixel offset of a peak from its two neighbours."""
  return 0.5 * (left - right) / np.minimum(left - 2 * mid + right, -1e-12)


def track(key, cur, p):
  """Estimate sensor motion (n, 2) from frame spectrum `key` to `cur` by correlation."""
  r = key * np.conj(cur)
  whiten = _per_run(p, "whiten", 2)
  if whiten.any():
    r = r / (np.abs(r) + 1e-12) ** whiten
  c = np.fft.irfft2(r, s=(VIEW, VIEW))
  rows = np.arange(len(c))
  iy, ix = np.divmod(c.reshape(len(c), -1).argmax(1), VIEW)
  mid = c[rows, iy, ix]
  ox = _parabola(c[rows, iy, (ix - 1) % VIEW], mid, c[rows, iy, (ix + 1) % VIEW])
  oy = _parabola(c[rows, (iy - 1) % VIEW, ix], mid, c[rows, (iy + 1) % VIEW, ix])
  # peak index wraps: VIEW-1 means -1
  dx = (ix + VIEW // 2) % VIEW - VIEW // 2 + ox
  dy = (iy + VIEW // 2) % VIEW - VIEW // 2 + oy
  return np.stack([dx, dy], 1) * _per_run(p, "gain", 1)


class Tracker:
  """Dead reckoning for n sensors: position = keyframe position + shift from keyframe."""

  def __init__(self, img, start, p):
    self.p = p
    self.key = spectrum(img, p)
    self.key_pos = np.repeat(np.asarray(start, float)[None], len(img), 0)
    self.pos = self.key_pos.copy()

  def update(self, img):
    cur = spectrum(img, self.p)
    shift = track(self.key, cur, self.p)
    self.pos = self.key_pos + shift
    rekey = np.hypot(*shift.T) >= np.asarray(self.p["rekey"])
    self.key[rekey], self.key_pos[rekey] = cur[rekey], self.pos[rekey]
    return self.pos


# ==========================================
# 3. BATCH: DRIFT OVER TIME, MANY RUNS
# ==========================================
def run_chunk(task):
  """n runs sharing one texture + start phase, differing only by sensor noise.
  reps > 1 repeats those n runs for reps parameter sets (p values given per run, length
  reps * n) with identical noise, so candidates are compared on exactly the same data.
  Returns drift samples (samples, reps * n), final error (reps * n, 2), position histogram or None."""
  seed, n, laps, stride, p, density, reps = task
  rng = np.random.default_rng(seed)
  surf = make_surface(seed)
  offset = rng.random(2)  # sub-pixel start phase
  frames = len(PATH) * laps
  noisy = lambda t: sense(route_at(t)[None] + offset, surf) + np.tile(
      SENSOR_NOISE * rng.standard_normal((n, VIEW, VIEW), np.float32), (reps, 1, 1))
  drift = np.zeros((math.ceil(frames / stride), reps * n), np.float32)
  hist = np.zeros((HEIGHT, WIDTH), np.float32) if density else None
  tracker = Tracker(noisy(0), PATH[0], p)
  for k in range(1, (frames - 1) * SUBSTEPS + 1):
    pos = tracker.update(noisy(k / SUBSTEPS))
    f, sub = divmod(k, SUBSTEPS)
    if sub:
      continue
    if density:
      ix, iy = pos.astype(int).T
      ok = (ix >= 0) & (ix < WIDTH) & (iy >= 0) & (iy < HEIGHT)
      np.add.at(hist, (iy[ok], ix[ok]), 1)
    if f % stride == 0:
      drift[f // stride] = np.hypot(*(pos - PATH[f % len(PATH)]).T)
  return drift, tracker.pos - PATH[(frames - 1) % len(PATH)], hist


def run_batch(runs=RUNS, laps=LAPS, chunk=CHUNK, seed=None, p=None):
  """Returns (sample frames, drift (samples, runs), final error (runs, 2), position histogram)."""
  p = p or PARAMS
  stride = max(1, len(PATH) * laps // 1000)  # keep ~1000 time samples
  seeds = np.random.default_rng(seed).integers(2**32, size=math.ceil(runs / chunk))
  tasks = [(int(s), min(chunk, runs - i * chunk), laps, stride, p, True, 1) for i, s in enumerate(seeds)]
  parts, hist, t0 = [], np.zeros((HEIGHT, WIDTH)), time.time()
  with multiprocessing.Pool() as pool:
    for drift, final, h in pool.imap(run_chunk, tasks):
      parts.append((drift, final))
      hist += h
      print(f"  {sum(len(f) for _, f in parts)}/{runs} runs  ({time.time() - t0:.0f}s)")
  drift = np.concatenate([d for d, _ in parts], 1)
  return np.arange(len(drift)) * stride, drift, np.concatenate([f for _, f in parts]), hist


def optimize(seed=0):
  """Cross-entropy search over BOUNDS. Score = RMS drift over a whole lap.
  Every candidate sees the same textures + noise (common random numbers), so
  score differences come from the parameters, not luck."""
  names = list(BOUNDS)
  lo, hi = np.array([BOUNDS[k] for k in names]).T
  rng = np.random.default_rng(seed)
  seeds = rng.integers(2**32, size=OPT_TEXTURES)
  mean, std = (lo + hi) / 2, (hi - lo) / 4
  history, best, t0 = [], (np.inf, None), time.time()
  with multiprocessing.Pool() as pool:
    for g in range(OPT_GENERATIONS):
      pop = np.clip(rng.normal(mean, std, (OPT_POP, len(names))), lo, hi)
      if g == 0:
        pop[0] = [DEFAULT_PARAMS[k] for k in names]  # baseline: the untuned tracker
      # one task per texture, all candidates inside it (OPT_RUNS rows each)
      per_run = {k: np.repeat(pop[:, j], OPT_RUNS) for j, k in enumerate(names)}
      tasks = [(int(s), OPT_RUNS, 1, 1, per_run, False, OPT_POP) for s in seeds]
      sq = sum((d.reshape(len(d), OPT_POP, OPT_RUNS).astype(np.float64) ** 2).sum((0, 2))
               for d, _, _ in pool.map(run_chunk, tasks))
      scores = np.sqrt(sq / (len(PATH) * OPT_TEXTURES * OPT_RUNS))
      scores = np.nan_to_num(scores, nan=np.inf)
      if g == 0:
        baseline = scores[0]
      if scores.min() < best[0]:
        best = (scores.min(), pop[scores.argmin()])
      elite = pop[np.argsort(scores)[:OPT_POP // 4]]
      mean, std = elite.mean(0), np.maximum(elite.std(0), (hi - lo) * 0.02)
      history.append((pop, scores))
      shown = ", ".join(f"{k}={v:.3f}" for k, v in zip(names, best[1]))
      print(f"  gen {g + 1}/{OPT_GENERATIONS}: best {best[0]:.2f}px ({shown})  ({time.time() - t0:.0f}s)")
  params = dict(zip(names, map(float, best[1])))
  with open(PARAMS_FILE, "w") as f:
    json.dump(params, f, indent=2)
  print(f"Baseline {baseline:.2f}px -> best {best[0]:.2f}px RMS drift. Saved {PARAMS_FILE}")
  np.savez(HISTORY_FILE, pops=np.stack([h[0] for h in history]), scores=np.stack([h[1] for h in history]),
           baseline=baseline, names=names, lo=lo, hi=hi)
  return history, baseline, names, lo, hi


def load_history():
  """optimize()'s return value, read back from HISTORY_FILE."""
  h = np.load(HISTORY_FILE)
  return list(zip(h["pops"], h["scores"])), float(h["baseline"]), list(h["names"]), h["lo"], h["hi"]


# ==========================================
# 4. CHARTS (matplotlib, classic MATLAB look)
# ==========================================
MATLAB_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]


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


def plot_optimize(history, baseline, names, lo, hi):
  """Score per candidate per generation + elite parameter trajectories. Saves optimize.png."""
  plt = matlab_style()
  fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
  gens = np.arange(1, len(history) + 1)
  for g, (_, scores) in zip(gens, history):
    ok = np.isfinite(scores)
    ax.plot(np.full(ok.sum(), g), scores[ok], ".", color="0.6", ms=4)
  best = np.minimum.accumulate([s.min() for _, s in history])
  ax.plot(gens, best, "o-", color=MATLAB_COLORS[0], label=f"best so far ({best[-1]:.2f}px)")
  ax.axhline(baseline, color=MATLAB_COLORS[1], ls="--", label=f"untuned baseline ({baseline:.2f}px)")
  ax.set(xlabel="Generation", ylabel="RMS drift over lap (px)", yscale="log",
         title=f"Optimization: {OPT_POP} candidates x {OPT_TEXTURES * OPT_RUNS} runs per generation")
  ax.legend(loc="upper right")
  for j, name in enumerate(names):
    elite = [pop[np.argsort(s)[:OPT_POP // 4], j].mean() for pop, s in history]
    ax2.plot(gens, (np.array(elite) - lo[j]) / (hi[j] - lo[j]), "o-",
             label=f"{name} [{lo[j]:g}, {hi[j]:g}] -> {elite[-1]:.3f}")
  ax2.set(xlabel="Generation", ylabel="Elite mean, scaled to bounds", ylim=(-0.05, 1.05),
          title="Parameter convergence")
  ax2.legend(loc="best")
  from matplotlib.ticker import MaxNLocator, ScalarFormatter
  for a in (ax, ax2):
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.yaxis.set_major_formatter(ScalarFormatter())
  ax.yaxis.set_minor_formatter(ScalarFormatter())
  fig.tight_layout()
  save(fig, "optimize.png")


# ==========================================
# 5. LIVE VIEW (pygame)
# ==========================================
ACTUAL_COLOR = (0, 114, 189)  # MATLAB blue: ground truth path
BELIEVED_COLOR = (217, 83, 25)  # MATLAB orange: where the mouse thinks it is
TEXT_COLOR = (0, 0, 0)


def live(screen, font, clock):
  rng = np.random.default_rng()
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
  elif "--batch" in sys.argv:
    print(f"Simulating {RUNS} runs x {LAPS} laps ({len(PATH) * LAPS} frames each), {params_text()}")
    samples, drift, final, hist = run_batch()
    plot_batch(samples, drift, final)
    plot_density(hist)
  if "--route" in sys.argv or "--optimize" in sys.argv or "--batch" in sys.argv:
    matlab_style().show()
  else:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Optical Mouse Tracking: Actual vs Believed")
    live(screen, pygame.font.SysFont(None, 24), pygame.time.Clock())
    pygame.quit()
