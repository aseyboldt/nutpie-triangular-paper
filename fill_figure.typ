// Spy plots of the triangular factor for a star graph, with the hub first and with the hub last.
// Black: diagonal and edges of G. Gray: fill created by integrating out later parameters.

// Symbolic elimination from last to first. `edges` are pairs of indices into `names`.
// Returns, for every row i, the set of parents as a dictionary j -> "edge" or "fill".
#let parents(n, edges) = {
  let adj = range(n).map(_ => (:))
  for (a, b) in edges {
    adj.at(a).insert(str(b), "edge")
    adj.at(b).insert(str(a), "edge")
  }
  let rows = range(n).map(_ => (:))
  for i in range(n - 1, -1, step: -1) {
    let earlier = adj.at(i).keys().map(int).filter(j => j < i)
    for j in earlier { rows.at(i).insert(str(j), adj.at(i).at(str(j))) }
    // Integrating out i connects all its earlier neighbours with each other.
    for a in earlier {
      for b in earlier {
        if a != b and str(b) not in adj.at(a) { adj.at(a).insert(str(b), "fill") }
      }
    }
  }
  rows
}

#let cell = 0.42cm
#let fill-color = luma(170)

#let spy(names, edges, title) = {
  let n = names.len()
  let rows = parents(n, edges)
  let n-fill = rows.map(r => r.values().filter(v => v == "fill").len()).sum()
  let label(body) = box(width: cell, height: cell, align(center + horizon, text(size: 0.8em, body)))
  let square(i, j) = {
    let kind = if i == j { "edge" } else if j < i { rows.at(i).at(str(j), default: none) } else { none }
    let paint = if kind == "edge" { black } else if kind == "fill" { fill-color } else { white }
    let stroke = if j <= i { 0.3pt + luma(200) } else { none }
    box(width: cell, height: cell, fill: paint, stroke: stroke)
  }
  stack(
    spacing: 0.5em,
    align(center, title),
    grid(
      columns: n + 1,
      gutter: 0pt,
      label[], ..names.map(label),
      ..range(n).map(i => (label(names.at(i)), ..range(n).map(j => square(i, j)))).join(),
    ),
    align(center, text(size: 0.85em)[#n-fill fill entries]),
  )
}

#let k = 6
#let leaves = range(1, k + 1).map(j => $ell_#j$)

#let fill-figure = grid(
  columns: 2,
  column-gutter: 3em,
  // Hub first: index 0 is the hub.
  spy(($h$, ..leaves), range(1, k + 1).map(j => (0, j)), [hub first]),
  // Hub last: index k is the hub.
  spy((..leaves, $h$), range(k).map(j => (j, k)), [hub last]),
)
