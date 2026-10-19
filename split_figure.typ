// Leapfrog with the standard split H = K + V versus the split H = H_q + H_qp,
// drawn in the position space of a 2d, slightly non-normal adapted density.

#import "@preview/lilaq:0.6.0" as lq
#import "@preview/tiptoe:0.4.0"

// Adapted density: a mild banana, log p(y) = -y1^2/2 - u^2/2, u = y2 + b (y1^2 - 1).
// The shift in y2 has unit Jacobian, so this is a normalised density.
#let b = 0.25
#let u(y) = y.at(1) + b * (y.at(0) * y.at(0) - 1)
#let log-p(y) = -y.at(0) * y.at(0) / 2 - u(y) * u(y) / 2
#let log-q(y) = -(y.at(0) * y.at(0) + y.at(1) * y.at(1)) / 2

// Kick forces: the full score for H = K + V, and the score of log(p / q) for H_q + H_qp.
#let grad-log-p(y) = (-y.at(0) - 2 * b * y.at(0) * u(y), -u(y))
#let grad-log-ratio(y) = (-2 * b * y.at(0) * u(y), -b * (y.at(0) * y.at(0) - 1))

#let add(a, c, s: 1) = a.zip(c).map(((x, z)) => x + s * z)

// Exact flow of H_q for time t: a rotation in every (y_i, rho_i) plane.
#let rotate(y, rho, t) = (
  add(y.map(v => v * calc.cos(t)), rho.map(v => v * calc.sin(t))),
  add(rho.map(v => v * calc.cos(t)), y.map(v => -v * calc.sin(t))),
)

// Run n leapfrog steps of size eps. Consecutive half-kicks (or half-rotations) are merged.
// Returns the position path as a polyline, and the kicks as (position, delta-rho) pairs.
#let leapfrog(y, rho, eps, n, split: "standard", arc-points: 24) = {
  let path = (y,)
  let kicks = ()
  if split == "standard" {
    for i in range(n + 1) {
      let h = if i == 0 or i == n { eps / 2 } else { eps }
      let dr = grad-log-p(y).map(g => h * g)
      kicks.push((y, dr))
      rho = add(rho, dr)
      if i < n {
        y = add(y, rho, s: eps)
        path.push(y)
      }
    }
  } else {
    for i in range(n + 1) {
      let h = if i == 0 or i == n { eps / 2 } else { eps }
      for k in range(1, arc-points + 1) {
        path.push(rotate(y, rho, h * k / arc-points).first())
      }
      (y, rho) = rotate(y, rho, h)
      if i < n {
        let dr = grad-log-ratio(y).map(g => eps * g)
        kicks.push((y, dr))
        rho = add(rho, dr)
      }
    }
  }
  (path: path, kicks: kicks, end: y)
}

// Setup shared by both panels.
#let y0 = (-1.6, -0.2)
#let rho0 = (0.9, 1.3)
#let eps = 0.9
#let n-steps = 4
#let arrow-scale = 0.5
#let lim = 2.6

// Reference trajectory: the same integrator with a tiny step.
#let exact = leapfrog(y0, rho0, eps / 50, n-steps * 50, split: "rotation", arc-points: 1)

#let pts = lq.linspace(-lim, lim, num: 70)
#let levels = (0.5, 2, 4.5).map(r => -r) // log-density at 1, 2, 3 "sigmas"

#let panel(split, title) = {
  show: lq.set-grid(stroke: none)
  let run = leapfrog(y0, rho0, eps, n-steps, split: split)
  let accent = if split == "standard" { rgb("#c0392b") } else { rgb("#1f6fb2") }
  lq.diagram(
    width: 5.6cm,
    height: 5.6cm,
    title: title,
    xlim: (-lim, lim),
    ylim: (-lim, lim),
    xlabel: $x_1$,
    ylabel: $x_2$,
    lq.contour(pts, pts, (x, z) => log-q((x, z)), levels: levels,
      map: (luma(185), luma(185)), stroke: (dash: "dashed", thickness: 0.5pt)),
    lq.contour(pts, pts, (x, z) => log-p((x, z)), levels: levels,
      map: (luma(110), luma(110)), stroke: 0.6pt),
    lq.plot(exact.path.map(p => p.at(0)), exact.path.map(p => p.at(1)),
      mark: none, stroke: (paint: luma(60), thickness: 0.6pt, dash: "dotted")),
    lq.plot(run.path.map(p => p.at(0)), run.path.map(p => p.at(1)),
      mark: none, stroke: (paint: accent, thickness: 1.1pt)),
    ..run.kicks.map(((p, dr)) => lq.line(p, add(p, dr, s: arrow-scale),
      stroke: (paint: black, thickness: 0.8pt), tip: tiptoe.stealth.with(length: 1.5pt + 250%))),
    lq.scatter((y0.at(0),), (y0.at(1),), mark: "o", size: 4pt, color: black),
    lq.scatter((exact.end.at(0),), (exact.end.at(1),), mark: "x", size: 5pt, color: luma(60)),
    lq.scatter((run.end.at(0),), (run.end.at(1),), mark: "o", size: 4pt, color: accent),
  )
}

#let split-figure = grid(
  columns: 2,
  column-gutter: 1.2em,
  panel("standard", $H = K + V$),
  panel("rotation", $H = H_q + H_(q p)$),
)
