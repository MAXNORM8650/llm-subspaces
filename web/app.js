// LLM Subspace Atlas — Sankey of query subspace-usage flow across layers.
const COL = { attn: "#2b7bd6", mlp: "#e8823a", other: "#888" };
const tip = document.getElementById("tip");

async function boot() {
  const manifest = await fetch("data/manifest.json").then(r => r.json());
  const pick = document.getElementById("pick");
  manifest.forEach(m => {
    const o = document.createElement("option");
    o.value = m.file; o.textContent = m.tag; o.dataset.q = m.query; pick.appendChild(o);
  });
  pick.onchange = () => load(pick.value, pick.selectedOptions[0].dataset.q);
  load(manifest[0].file, manifest[0].query);
}

async function load(file, query) {
  document.getElementById("qtext").textContent = "“" + query + "”";
  const data = await fetch("data/" + file).then(r => r.json());
  render(data);
}

function render(data) {
  const chart = document.getElementById("chart");
  chart.innerHTML = "";
  const nById = new Map(data.nodes.map(d => [d.id, d]));
  const links = data.links.filter(l => nById.has(l.source) && nById.has(l.target));
  const nLayers = d3.max(data.nodes, d => d.layer) + 1;

  const W = Math.max(1100, nLayers * 34), H = 620, M = { t: 10, r: 40, b: 10, l: 40 };
  const svg = d3.select(chart).append("svg").attr("width", W).attr("height", H);

  const sankey = d3.sankey()
    .nodeId(d => d.id)
    .nodeWidth(9).nodePadding(6)
    .nodeSort(null)
    .extent([[M.l, M.t], [W - M.r, H - M.b]]);

  const graph = sankey({
    nodes: data.nodes.map(d => Object.assign({}, d)),
    links: links.map(d => Object.assign({}, d)),
  });

  // links
  svg.append("g").selectAll("path").data(graph.links).join("path")
    .attr("class", "link")
    .attr("d", d3.sankeyLinkHorizontal())
    .attr("stroke", d => COL[d.family] || COL.other)
    .attr("stroke-width", d => Math.max(1, d.width))
    .on("mousemove", (e, d) => showTip(e,
      `<b>${d.source.module}</b> L${d.source.layer}→L${d.target.layer}<br>usage ${d.value}`))
    .on("mouseleave", hideTip);

  // nodes
  const node = svg.append("g").selectAll("g").data(graph.nodes).join("g").attr("class", "node");
  node.append("rect")
    .attr("x", d => d.x0).attr("y", d => d.y0)
    .attr("height", d => Math.max(1, d.y1 - d.y0)).attr("width", d => d.x1 - d.x0)
    .attr("fill", d => COL[d.family] || COL.other)
    .on("mousemove", (e, d) => showTip(e,
      `<b>${d.module}</b> · layer ${d.layer}<br>usage <code>${d.usage}</code> · eff-rank ${d.eff_rank}<br>top subspace dirs: <code>[${d.top_dirs.slice(0,8).join(", ")}]</code>`))
    .on("mouseleave", hideTip);

  // layer ticks (every 6)
  svg.append("g").selectAll("text").data(graph.nodes.filter(d => d.module === "gate_proj" && d.layer % 6 === 0))
    .join("text").attr("x", d => (d.x0 + d.x1) / 2).attr("y", H - 2)
    .attr("text-anchor", "middle").attr("fill", "#8b98a9").attr("font-size", 10)
    .text(d => "L" + d.layer);
}

function showTip(e, html) { tip.innerHTML = html; tip.style.opacity = 1; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px"; }
function hideTip() { tip.style.opacity = 0; }

boot();
