// Transformer and conditioner of one row of a triangular map.
//
// slice-figure: a 2d target with parent y_1 and child y_2. Each slice y_1 = c is mapped to a
// standard normal by its own 1d monotone transformer, whose coefficients the conditioner picks.
// conditioner-diagram: how the coefficients are computed from the parents.

#import "@preview/lilaq:0.6.0" as lq
#import "@preview/tiptoe:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

// The map is defined analytically, the target density is its pullback of N(0, I).
// Row 1: x_1 = y_1. Row 2: sinh-arcsinh transformer, with
//   location m(y_1) = 0.5 (y_1^2 - 1)  (nonlinear conditioner, banana),
//   log-scale l(y_1) = 0.35 y_1        (scale skip, funnel),
//   skewness eps = 0.4, tail weight delta = 1.
#let cond(y1) = (m: 0.5 * (y1 * y1 - 1), l: 0.35 * y1, eps: 0.4, delta: 1.0)

#let asinh(v) = calc.ln(v + calc.sqrt(v * v + 1))
#let sinh(v) = (calc.exp(v) - calc.exp(-v)) / 2
#let cosh(v) = (calc.exp(v) + calc.exp(-v)) / 2

// Transformer T(y; pi) and its derivative in y.
#let transformer(y, pi) = {
  let z = calc.exp(-pi.l) * (y - pi.m)
  sinh(pi.delta * asinh(z) - pi.eps)
}
#let transformer-dy(y, pi) = {
  let z = calc.exp(-pi.l) * (y - pi.m)
  cosh(pi.delta * asinh(z) - pi.eps) * pi.delta / calc.sqrt(1 + z * z) * calc.exp(-pi.l)
}

#let normal-pdf(v) = calc.exp(-v * v / 2) / calc.sqrt(2 * calc.pi)

// Conditional density p(y_2 | y_1) and the joint density.
#let cond-pdf(y2, y1) = {
  let pi = cond(y1)
  normal-pdf(transformer(y2, pi)) * transformer-dy(y2, pi)
}
#let joint-pdf(y2, y1) = normal-pdf(y1) * cond-pdf(y2, y1)

#let slices = (-1.5, 0, 1.5)
#let colors = (rgb("#1f6fb2"), rgb("#7a7a7a"), rgb("#c0392b"))
#let ridge-height = 0.8 // a density of 1 is drawn 0.8 units high

#let y2-range = (-3.5, 5)
#let x2-range = (-3.5, 3.5)
#let parent-range = (-2.6, 2.6)
#let ys2 = lq.linspace(..y2-range, num: 200)
#let xs2 = lq.linspace(..x2-range, num: 120)

// Joint density contours at the levels of a standard normal at 1, 2 and 3 "sigmas".
#let levels = (0.5, 2, 4.5).map(r => calc.exp(-r) / (2 * calc.pi))

#let ridges(xs, pdf) = slices.zip(colors).map(((c, col)) => {
  let top = xs.map(v => c + ridge-height * pdf(v, c))
  (
    lq.fill-between(xs, top, y2: xs.map(_ => c), fill: col.transparentize(80%), stroke: none),
    lq.plot(xs, top, mark: none, stroke: (paint: col, thickness: 0.9pt)),
    lq.plot((xs.first(), xs.last()), (c, c), mark: none, stroke: (paint: col, thickness: 0.5pt)),
  )
}).join()

#let panel-args = (width: 3.9cm, height: 3.9cm)

#let slice-figure = {
  show: lq.set-grid(stroke: none)
  let target = lq.diagram(
    ..panel-args,
    title: [target $cal(T)$],
    xlim: y2-range,
    ylim: parent-range,
    xlabel: $y_2$,
    ylabel: $y_1$,
    lq.contour(lq.linspace(..y2-range, num: 90), lq.linspace(..parent-range, num: 70),
      joint-pdf, levels: levels, map: (luma(185), luma(185)), stroke: 0.5pt),
    ..ridges(ys2, cond-pdf),
  )
  let maps = lq.diagram(
    ..panel-args,
    title: [transformers],
    xlim: y2-range,
    ylim: x2-range,
    xlabel: $y_2$,
    ylabel: $x_2$,
    ..slices.zip(colors).map(((c, col)) => {
      let pi = cond(c)
      lq.plot(ys2, ys2.map(v => transformer(v, pi)), mark: none,
        stroke: (paint: col, thickness: 1.1pt))
    }),
  )
  let adapted = lq.diagram(
    ..panel-args,
    title: [adapted $cal(A)$],
    xlim: x2-range,
    ylim: parent-range,
    xlabel: $x_2$,
    ylabel: $x_1$,
    lq.contour(lq.linspace(..x2-range, num: 70), lq.linspace(..parent-range, num: 70),
      (a, b) => normal-pdf(a) * normal-pdf(b), levels: levels,
      map: (luma(185), luma(185)), stroke: 0.5pt),
    ..ridges(xs2, (v, c) => normal-pdf(v)),
  )
  grid(columns: 3, column-gutter: 1.6em, target, maps, adapted)
}

// The conditioner sums the contributions of its parts, the transformer applies the result.
#let conditioner-diagram = diagram(
  spacing: (1.6em, 1.1em),
  node-stroke: 0.6pt,
  node-corner-radius: 3pt,
  edge-stroke: 0.6pt,
  mark-scale: 80%,

  node((0, 1), $y_(P(i))$, stroke: none, name: <par>),
  node((1, 0), [location skip \ #text(size: 0.85em)[$sum_j b_(i j) y_j$]], name: <loc>),
  node((1, 1), [scale skip \ #text(size: 0.85em)[$sum_j s_(i j) y_j$]], name: <scale>),
  node((1, 2), [MLP \ #text(size: 0.85em)[width `nn_width`]], name: <mlp>),
  node((2, 1), $plus.o$, stroke: none, inset: 2pt, name: <sum>),
  node((2, -0.5), [constants], stroke: none, name: <const>),
  node((3, 1), $pi_i = (m_i, ell_i, epsilon_i, delta_i)$, stroke: none, name: <pi>),
  node((4, 1), [transformer \ #text(size: 0.85em)[$T(dot; pi_i)$]], name: <T>),
  node((4, -0.5), $y_i$, stroke: none, name: <yi>),
  node((5, 1), $x_i$, stroke: none, name: <xi>),

  edge(<par>, <loc>, "-|>"),
  edge(<par>, <scale>, "-|>"),
  edge(<par>, <mlp>, "-|>"),
  edge(<loc>, <sum>, "-|>", $m_i$, label-side: left),
  edge(<scale>, <sum>, "-|>", $ell_i$, label-side: left, label-pos: 0.4),
  edge(<mlp>, <sum>, "-|>", [all], label-side: right),
  edge(<const>, <sum>, "-|>"),
  edge(<sum>, <pi>, "-|>"),
  edge(<pi>, <T>, "-|>"),
  edge(<yi>, <T>, "-|>"),
  edge(<T>, <xi>, "-|>"),

  // Everything left of the transformer is the conditioner.
  node(enclose: (<loc>, <scale>, <mlp>, <sum>, <const>), stroke: (dash: "dashed", paint: luma(150)),
    inset: 6pt, corner-radius: 5pt, name: <cond>),
  node((1.5, 2.75), text(size: 0.85em, fill: luma(100))[conditioner $C$], stroke: none),
)
