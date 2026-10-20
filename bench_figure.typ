// Benchmark of diagonal (`diag`) versus triangular flow (`flow`) adaptation, read from
// bench-big.csv. One row per model, sorted by dimension, with WALNUTS (maximum tree depth 6)
// and NUTS (maximum tree depth 10) as two sub-rows. Runs with max R-hat > 1.01 are drawn
// hollow, since their ESS estimates are not trustworthy.

#import "@preview/lilaq:0.6.0" as lq

#let rows = csv("bench-zen5.csv", row-type: dictionary)

// Display names; models not listed here show their posteriordb name.
#let labels = (
  "eight_schools-eight_schools_noncentered": [Eight schools, non-centred],
  "eight_schools-eight_schools_centered": [Eight schools, centred],
  "diamonds-diamonds": [Diamonds],
  "uk_drivers-state_space_stochastic_level_stochastic_seasonal": [UK drivers, state space],
  "Mth_data-Mth_model": [Mth],
  "stochastic_volatility": [Stochastic volatility],
  "arK-arK": [AR($K$)],
  "garch-garch11": [GARCH(1,1)],
  "butterfly-multi_occupancy": [Butterfly occupancy],
  "pilots-pilots": [Pilots],
  "GLMM_Poisson_data-GLMM_Poisson_model": [Poisson GLMM],
  "gp_pois_regr-gp_pois_regr": [GP Poisson regression],
  "mcycle_gp-accel_gp": [Motorcycle GP],
  "mcycle_splines-accel_splines": [Motorcycle splines],
  "normal_2-normal_mixture": [Normal mixture, 2],
  "normal_5-normal_mixture_k": [Normal mixture, 5],
  "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse": [Collapsing Gaussian mixture],
  "radon_mn-radon_hierarchical_intercept_centered": [Radon, hierarchical intercept],
  "radon_mn-radon_variable_intercept_slope_centered": [Radon, intercept and slope],
  "radon_mn-radon_variable_slope_centered": [Radon, varying slope],
  "surgical_data-surgical_model": [Surgical],
  "one_comp_mm_elim_abs-one_comp_mm_elim_abs": [One compartment PK],
  "soil-carbon-soil-incubation": [Soil incubation],
  "sat-hier_2pl": [SAT, hierarchical 2PL],
)

#let models = (
  rows
    .map(r => r.name)
    .dedup()
    .sorted(key: name => (int(rows.find(r => r.name == name).n_dim), name))
    .map(name => (name, labels.at(name, default: raw(name))))
)

#let colors = (diag: rgb("#eb6834"), flow: rgb("#2a78d6"))
#let samplers = (("True", "o", 0.17, [WALNUTS, depth 6]), ("False", "s", -0.17, [NUTS, depth 10]))
#let rhat-ok = 1.01

#let find(name, adaptation, walnuts) = rows.find(r => (
  r.name == name and r.adaptation == adaptation and r.walnuts == walnuts
))

// Metrics. `ess` is the minimum bulk ESS over parameters, gradients are summed over 4 chains.
#let ess-per-grad(r) = 1000 * float(r.ess) / (int(r.warmup_grads) + int(r.sampling_grads))
#let ess-per-sec(r) = float(r.ess) / float(r.time)

#let n-models = models.len()
#let ypos(m, offset) = n-models - 1 - m + offset

#let model-ticks = models.enumerate().map(((m, (name, label))) => {
  let n = find(name, "flow", "True").n_dim
  (ypos(m, 0), [#label #text(fill: luma(110))[(#n)]])
})

#let panel(metric, xlabel, show-labels: true) = {
  let items = ()
  // Faint separators between models.
  items.push(lq.hlines(..range(n-models - 1).map(m => m + 0.5), stroke: 0.4pt + luma(220)))
  for (m, (name, _)) in models.enumerate() {
    for (walnuts, _, offset, _) in samplers {
      let y = ypos(m, offset)
      let xs = ("diag", "flow").map(a => metric(find(name, a, walnuts)))
      items.push(lq.plot(xs, (y, y), stroke: 0.8pt + luma(170), mark: none))
    }
  }
  for adaptation in ("diag", "flow") {
    for (walnuts, mark, offset, _) in samplers {
      for converged in (true, false) {
        let pts = models.enumerate().map(((m, (name, _))) => (m, find(name, adaptation, walnuts)))
        pts = pts.filter(((m, r)) => (float(r.rhat) <= rhat-ok) == converged)
        if pts.len() == 0 { continue }
        items.push(lq.plot(
          pts.map(((m, r)) => metric(r)),
          pts.map(((m, r)) => ypos(m, offset)),
          stroke: none,
          mark: mark,
          mark-size: 5pt,
          color: colors.at(adaptation),
          mark-color: if converged { colors.at(adaptation) } else { white },
        ))
      }
    }
  }
  lq.diagram(
    width: 4.8cm,
    height: n-models * 0.7cm,
    xscale: "log",
    xlabel: xlabel,
    ylim: (-0.5, n-models - 0.5),
    yaxis: (
      ticks: if show-labels { model-ticks } else { model-ticks.map(((y, _)) => (y, none)) },
      subticks: none,
      mirror: false,
    ),
    grid: (stroke: luma(235)),
    legend: none,
    ..items,
  )
}

#let legend-entry(mark, fill, stroke, label) = box(baseline: -0.1em, {
  let shape = if mark == "o" { circle.with(radius: 2.5pt) } else { square.with(size: 5pt) }
  shape(fill: fill, stroke: 0.8pt + stroke)
}) + h(0.3em) + label

#let bench-figure = {
  grid(
    columns: 2,
    column-gutter: 0.8em,
    panel(ess-per-grad, [ESS per 1000 gradients]),
    panel(ess-per-sec, [ESS per second], show-labels: false),
  )
  v(0.4em)
  let gray = luma(90)
  (
    legend-entry("s", colors.diag, colors.diag, [diag]),
    legend-entry("s", colors.flow, colors.flow, [flow]),
    ..samplers.map(((_, mark, _, label)) => legend-entry(mark, gray, gray, label)),
    legend-entry("o", white, gray, [$hat(R) > #rhat-ok$]),
  ).join(h(1.2em))
}
