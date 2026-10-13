"""README figures and GIFs. python figures.py -> docs/*.png, docs/*.gif

Everything is drawn with the simulator's own sense / spectrum / track / Tracker,
so the pictures show the real algorithm, not an illustration of it.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # live view renders off-screen
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import pygame
import main as m

plt = m.matlab_style()
C = m.MATLAB_COLORS
V = m.VIEW
PARULA = LinearSegmentedColormap.from_list("parula", m.PARULA_ANCHORS)  # MATLAB's default colormap
MATLAB_VIEW = dict(elev=30, azim=-37.5)  # MATLAB's default 3-D camera
P = m.PATH[220]  # a sensor position on the zigzag
rng = np.random.default_rng(7)


def noisy(pos):
  pos = np.atleast_2d(pos)
  return m.sense(pos) + m.SENSOR_NOISE * rng.standard_normal((len(pos), V, V), np.float32)


def correlation(a, b, p):
  """Same maths as main.track, but returns the whole surface c(u, v), centred (fftshift)."""
  r = m.spectrum(a, p) * np.conj(m.spectrum(b, p))
  if p["whiten"]:
    r = r / (np.abs(r) + 1e-12) ** p["whiten"]
  c = np.fft.irfft2(r, s=(m.SIZE, m.SIZE))
  if m.CORRELATION == "linear":
    c = c / m.envelope(p, len(c))
  return np.fft.fftshift(c, axes=(-2, -1)), r


def clean_axes(ax):
  ax.set_xticks([])
  ax.set_yticks([])
  ax.grid(False)


# ------------------------------------------------------------------ figures
def fig_noise_viewport():
  fig = plt.figure(figsize=(13, 6.2))
  gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.6, 1.25])
  big = fig.add_subplot(gs[:, :2])
  big.imshow(m.surface, cmap="gray", extent=(0, m.WIDTH, m.HEIGHT, 0))
  big.plot(*np.vstack([m.START, m.PATH]).T, color=C[1], lw=1, alpha=0.8)
  big.add_patch(Rectangle(P - V / 2, V, V, fill=False, color=C[1], lw=2))
  big.set(title=r"Surface $S(x,y)$: 800$\times$600 pixels, each one random",
          xlabel="x (px)", ylabel="y (px)")
  big.grid(False)

  patch = fig.add_subplot(gs[0, 2])
  patch.imshow(noisy(P)[0], cmap="gray", interpolation="nearest", extent=(0, V, V, 0))
  patch.set_xticks(np.arange(0, V + 1, 4))
  patch.set_yticks(np.arange(0, V + 1, 4))
  patch.set_xticks(np.arange(V + 1), minor=True)
  patch.set_yticks(np.arange(V + 1), minor=True)
  patch.grid(True, which="minor", color="black", lw=0.2, alpha=0.4)
  patch.grid(False, which="major")
  patch.set_title(rf"Sensor image $I_k$: {V}$\times${V} + read noise")
  for corner_big, corner_patch in [((P[0] + V / 2, P[1] - V / 2), (0, 0)), ((P[0] + V / 2, P[1] + V / 2), (0, V))]:
    fig.add_artist(ConnectionPatch(corner_big, corner_patch, "data", "data", axesA=big, axesB=patch,
                                   color=C[1], lw=1))

  cmp = fig.add_subplot(gs[1, 2])
  y, x = int(P[1]) - 24, int(P[0]) - 24
  raw = np.random.default_rng(m.SEED).random((m.HEIGHT, m.WIDTH))[y:y + 48, x:x + 48]
  blur = m.surface[y:y + 48, x:x + 48]
  cmp.imshow(np.hstack([(raw - raw.min()) / np.ptp(raw), np.ones((48, 3)), (blur - blur.min()) / np.ptp(blur)]),
             cmap="gray", interpolation="nearest")
  cmp.set_title(rf"white noise vs grain $\sigma_s$={m.SURFACE_BLUR:g} px")
  clean_axes(cmp)
  fig.tight_layout()
  m.save(fig, "noise_viewport.png")


def fig_pipeline(p):
  d = np.array([2.3, -1.4])
  a, b = noisy(P), noisy(P + d)
  c = correlation(a, b, p)[0][0]
  est = m.track(m.spectrum(a, p), m.spectrum(b, p), p)[0]
  w = p["window"]
  window = 1 - w + w * m.HANN

  fig = plt.figure(figsize=(14, 8.4))
  gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.35])
  panels = [(a[0], r"$I_k$"), (b[0], r"$I_{k+1}$ (moved $\mathbf{d}$)"),
            ((a[0] - a[0].mean()) * window, r"$w\cdot(I_k-\bar I_k)$")]
  for j, (img, title) in enumerate(panels):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    clean_axes(ax)
  ax = fig.add_subplot(gs[0, 3])
  spec = np.abs(np.fft.fftshift(np.fft.fft2((a[0] - a[0].mean()) * window)))
  ax.imshow(np.log(spec + 1e-3), cmap=PARULA, interpolation="nearest")
  ax.set_title(r"$\log|\hat I_k(\mathbf{f})|$: grain is low-pass")
  clean_axes(ax)

  u = np.arange(m.SIZE) - m.SIZE // 2
  uu, vv = np.meshgrid(u, u)
  ax3 = fig.add_subplot(gs[1, :2], projection="3d")
  ax3.plot_surface(uu, vv, c, cmap=PARULA, edgecolor="black", lw=0.15, antialiased=True)
  ax3.view_init(**MATLAB_VIEW)
  ax3.set(xlabel="u (px)", ylabel="v (px)", zlabel=r"$c(u,v)$",
          title=r"correlation surface $c=\mathcal{F}^{-1}\{R\}$")

  iy, ix = np.unravel_index(c.argmax(), c.shape)
  ax = fig.add_subplot(gs[1, 2])
  k = 4
  ax.imshow(c[iy - k:iy + k + 1, ix - k:ix + k + 1], cmap=PARULA, interpolation="nearest",
            extent=(u[ix] - k - 0.5, u[ix] + k + 0.5, u[iy] + k + 0.5, u[iy] - k - 0.5))
  ax.plot(u[ix], u[iy], "s", mfc="none", mec="black", ms=12, mew=1.5, label="integer peak")
  ax.plot(*est, "o", color=C[1], ms=7, label="sub-pixel estimate")
  ax.plot(*d, "+", color="black", ms=14, mew=2, label="true shift")
  ax.set(title="peak neighbourhood", xlabel="u (px)", ylabel="v (px)")
  ax.grid(False)
  ax.legend(loc="lower left", fontsize=8)

  ax = fig.add_subplot(gs[1, 3])
  ys = c[iy, ix - 1:ix + 2]
  xs = u[ix - 1:ix + 2]
  denom = ys[0] - 2 * ys[1] + ys[2]
  vx = xs[1] + 0.5 * (ys[0] - ys[2]) / denom
  xx = np.linspace(xs[0] - 0.5, xs[2] + 0.5, 100)
  ax.plot(u, c[iy], ".-", color="0.6", lw=1, label=r"$c(u, v^*)$")
  ax.plot(xx, ys[1] + 0.5 * (ys[2] - ys[0]) * (xx - xs[1]) + 0.5 * denom * (xx - xs[1]) ** 2,
          color=C[0], label="parabola through 3 points")
  ax.plot(xs, ys, "o", color=C[0])
  ax.axvline(vx, color=C[1], label=f"vertex {vx:.2f}")
  ax.axvline(d[0], color="black", ls="--", label=f"true {d[0]:.2f}")
  ax.set(xlim=(xs[0] - 2, xs[2] + 2), xlabel="u (px)", title="sub-pixel fit along u")
  ax.legend(loc="lower center", fontsize=8)
  fig.tight_layout()
  m.save(fig, "pipeline.png")


def fig_params(p):
  fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))

  # (a) peak shape vs whitening
  ax = axs[0, 0]
  a, b = noisy(P), noisy(P + [1.3, 0])
  u = np.arange(m.SIZE) - m.SIZE // 2
  for j, wh in enumerate([0, 0.5, 1]):
    c = correlation(a, b, p | {"whiten": wh})[0][0]
    iy = np.unravel_index(c.argmax(), c.shape)[0]
    ax.plot(u, c[iy] / c.max(), ".-", color=C[j], label=f"whiten = {wh:g}")
  ax.set(xlim=(-8, 8), xlabel="u (px)", ylabel="c / max c", title="(a) whitening sharpens the peak")
  ax.legend()

  # (b) bias vs true shift for window strengths
  ax = axs[0, 1]
  shifts = np.linspace(0, 4, 17)
  starts = m.PATH[rng.integers(len(m.PATH), size=300)] + rng.random((300, 2))
  for j, wnd in enumerate([0, 0.5, 1]):
    q = p | {"window": wnd, "gain": 1.0, "gain3": 0.0}
    err = np.array([m.track(m.spectrum(noisy(starts), q), m.spectrum(noisy(starts + [s, 0]), q), q)[:, 0] - s
                    for s in shifts])
    ax.plot(shifts, err.mean(1), "o-", color=C[j], ms=4, label=f"window = {wnd:g}")
    ax.fill_between(shifts, err.mean(1) - err.std(1), err.mean(1) + err.std(1), color=C[j], alpha=0.12, lw=0)
  ax.axhline(0, color="black", lw=0.8)
  ax.set(xlabel=r"true shift $d_x$ (px)", ylabel=r"$\hat d_x - d_x$ (px)",
         title="(b) taper bias pulls estimates toward zero", ylim=(-0.6, 0.3))
  ax.legend(loc="lower left")

  # (c) error growth: per-frame vs keyframe
  ax = axs[1, 0]
  n, steps, v = 300, 400, np.array([0.375, 0.1])
  starts = np.c_[rng.uniform(150, 300, n), rng.uniform(150, 450, n)]
  for j, rk in enumerate([0, 2, 4, 6]):
    q = p | {"rekey": rk, "gain": 1.0, "gain3": 0.0, "loop": 0.0}
    tr = m.Tracker(noisy(starts), (0, 0), q)  # track displacement from each run's start
    rms = [np.sqrt(np.mean(np.sum((tr.update(noisy(starts + k * v)) - k * v) ** 2, 1)))
           for k in range(1, steps + 1)]
    ax.plot(np.arange(1, steps + 1), rms, color=C[j], label=f"rekey = {rk:g} px")
  ax.set(xlabel="sensor frame k", ylabel="RMS position error (px)", title="(c) keyframe threshold vs accumulated error")
  ax.legend(loc="upper left")

  # (d) gain: measured step vs true step
  ax = axs[1, 1]
  q = p | {"gain": 1.0, "gain3": 0.0}
  meas = np.array([m.track(m.spectrum(noisy(starts), q), m.spectrum(noisy(starts + [s, 0]), q), q)[:, 0].mean()
                   for s in shifts])
  ax.plot(shifts, shifts, color="black", lw=0.8, ls="--", label="ideal")
  ax.plot(shifts, meas, "o-", color=C[0], ms=4, label="gain = 1")
  ax.plot(shifts, meas * (p["gain"] + p["gain3"] * meas**2), "o-", color=C[1], ms=4,
          label=f"tuned: gain {p['gain']:.3f}, gain3 {p['gain3']:+.4f}")
  ax.set(xlabel=r"true shift $d_x$ (px)", ylabel=r"mean $\hat d_x$ (px)", title="(d) calibration undoes the scale error")
  ax.legend(loc="upper left")
  fig.tight_layout()
  m.save(fig, "params.png")


# ------------------------------------------------------------------ GIFs
def gif_live(frames=714, every=3, width=600):
  """Hero GIF: the pygame live view, one full lap, captured off-screen."""
  pygame.init()
  screen = pygame.display.set_mode((m.WIDTH, m.HEIGHT))
  font = pygame.font.SysFont(None, 24)
  shots = []

  class Capture:
    n = 0

    def tick(self, fps):
      Capture.n += 1
      if Capture.n % every == 0:
        img = Image.fromarray(pygame.surfarray.array3d(screen).swapaxes(0, 1))
        shots.append(img.resize((width, width * m.HEIGHT // m.WIDTH), Image.LANCZOS))
      if Capture.n >= frames:
        pygame.event.post(pygame.event.Event(pygame.QUIT))

  m.live(screen, font, Capture())
  pygame.quit()
  path = os.path.join(m.DOCS, "live.gif")
  # One shared palette, no dithering, and every pixel that did not change since the last
  # frame set to a transparent index. The noise texture compresses terribly, so the GIF
  # must only store what moves.
  clear = 255
  montage = Image.new("RGB", (shots[0].width, 3 * shots[0].height))
  for j, s in enumerate([shots[0], shots[len(shots) // 2], shots[-1]]):  # palette must see the trails
    montage.paste(s, (0, j * s.height))
  base = montage.quantize(colors=clear, dither=Image.Dither.NONE)
  idx = [np.array(s.quantize(palette=base, dither=Image.Dither.NONE)) for s in shots]
  frames_q = []
  for i, a in enumerate(idx):
    a = a.copy()
    if i:
      a[a == idx[i - 1]] = clear
    f = Image.fromarray(a, "P")
    f.putpalette(base.getpalette())
    frames_q.append(f)
  frames_q[0].save(path, save_all=True, append_images=frames_q[1:], duration=1000 * every // 60,
                   loop=0, transparency=clear, disposal=1, optimize=False)
  print(f"Saved docs/live.gif ({len(shots)} frames, {os.path.getsize(path) / 1e6:.1f} MB)")


def gif_correlation(p, steps=120, t0=150.0):
  """Correlation surface while the sensor moves, with keyframe swaps."""
  ts = t0 + np.arange(steps + 1) / m.SUBSTEPS
  imgs = [noisy(m.route_at(t))[0:1] for t in ts]
  tracker = m.Tracker(imgs[0], m.route_at(ts[0]), p)
  key_img = imgs[0][0]
  u = np.arange(m.SIZE) - m.SIZE // 2
  uu, vv = np.meshgrid(u, u)
  fig = plt.figure(figsize=(11, 4.2), dpi=80)
  gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.08)
  ax_key, ax_cur = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
  ax3 = fig.add_subplot(gs[2], projection="3d")
  zmax = correlation(imgs[0], imgs[0], p)[0].max()

  def frame(k):
    nonlocal key_img
    img = imgs[k + 1]
    c = correlation(key_img[None], img, p)[0][0]
    key_before = tracker.key_pos.copy()
    pos = tracker.update(img)[0]
    rekeyed = not np.allclose(key_before, tracker.key_pos)
    for ax in (ax_key, ax_cur, ax3):
      ax.cla()
    ax_key.imshow(key_img, cmap="gray", interpolation="nearest")
    ax_key.set_title("keyframe" + ("  (new)" if rekeyed else ""), color=C[1] if rekeyed else "black")
    ax_cur.imshow(img[0], cmap="gray", interpolation="nearest")
    err = np.hypot(*(pos - m.route_at(ts[k + 1])))
    ax_cur.set_title(f"current frame, error {err:.2f} px")
    for ax in (ax_key, ax_cur):
      clean_axes(ax)
    ax3.plot_surface(uu, vv, c, cmap=PARULA, edgecolor="black", lw=0.1, vmin=-0.2 * zmax, vmax=zmax)
    ax3.set(zlim=(-0.3 * zmax, zmax), xlabel="u", ylabel="v", zticklabels=[],
            title=r"$c(u,v)$ between keyframe and current frame")
    ax3.view_init(**MATLAB_VIEW)
    if rekeyed:
      key_img = img[0]

  anim = FuncAnimation(fig, frame, frames=steps)
  path = os.path.join(m.DOCS, "correlation.gif")
  anim.save(path, writer=PillowWriter(fps=12))
  plt.close(fig)
  print(f"Saved docs/correlation.gif ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
  p = m.PARAMS
  print("Tracker params:", m.params_text(p))
  which = sys.argv[1:] or ["noise", "pipeline", "params", "live", "correlation", "untuned"]
  if "noise" in which:
    fig_noise_viewport()
  if "pipeline" in which:
    fig_pipeline(p)
  if "params" in which:
    fig_params(p)
  if "live" in which:
    gif_live()
  if "correlation" in which:
    gif_correlation(p)
  if "untuned" in which:  # density of the untuned tracker, for the before/after pair
    hist = m.run_batch(runs=2000, p=m.DEFAULT_PARAMS, seed=1)[3]
    m.plot_density(hist, m.DEFAULT_PARAMS, "density_untuned.png")
