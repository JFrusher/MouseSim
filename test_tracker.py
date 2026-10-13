"""Regression checks for the tracker. python test_tracker.py (or pytest)."""
import numpy as np
import main as m

P = m.PATH[220]


def _frames(d, p, n=200, seed=0):
  rng = np.random.default_rng(seed)
  noise = lambda: m.SENSOR_NOISE * rng.standard_normal((n, m.VIEW, m.VIEW), np.float32)
  return m.spectrum(m.sense(P[None]) + noise(), p), m.spectrum(m.sense((P + d)[None]) + noise(), p)


def test_known_shift_recovered():
  for d in [(0.0, 0.0), (0.4, -0.2), (1.3, 0.7), (-2.0, 1.5)]:
    est = m.track(*_frames(np.array(d), m.DEFAULT_PARAMS), m.DEFAULT_PARAMS).mean(0)
    assert np.allclose(est, d, atol=0.15), (d, est)


def test_batched_candidates_match_separate_runs():
  a = dict(m.DEFAULT_PARAMS)
  b = dict(m.PARAMS)
  both = {k: np.repeat([a[k], b[k]], 3) for k in a}
  task = lambda p, reps: {"seed": 7, "n": 3, "laps": 1, "stride": 1, "p": p, "reps": reps}
  d2 = m.run_chunk(task(both, 2))["drift"]
  assert np.allclose(d2[:, :3], m.run_chunk(task(a, 1))["drift"], atol=1e-3)
  assert np.allclose(d2[:, 3:], m.run_chunk(task(b, 1))["drift"], atol=1e-3)


def test_route_closes_and_stays_on_surface():
  assert np.allclose(m.PATH[-1], m.START)
  assert (m.PATH.min(0) >= m.MARGIN).all()


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_"):
      fn()
      print("ok", name)
