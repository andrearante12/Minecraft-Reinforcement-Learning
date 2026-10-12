/* Live split-screen driver: SSE from visualization/live_server.py → camera + predictions. */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }
  function jpeg(b64) { return "data:image/jpeg;base64," + b64; }
  function fmt(v, d) { return v == null ? "—" : Number(v).toFixed(d == null ? 2 : d); }

  var api = window.NetViz.mount({ source: "push" });
  var kind = "vector";
  var lastFrameAt = 0;
  var phase = "";

  function setText(id, text) { var el = $(id); if (el) el.textContent = text; }

  function showPhase(text) {
    var el = $("phase");
    el.textContent = text || "";
    el.hidden = !text;
  }

  function buildStrip(n) {
    var strip = $("dream-strip");
    strip.innerHTML = "";
    for (var i = 0; i < n; i++) {
      var fig = document.createElement("figure");
      fig.className = "tile";
      fig.innerHTML = "<img alt=\"Imagined step " + (i + 1) + "\" />" +
        "<figcaption><b>t+" + (i + 1) + "</b> <span class=\"ra\"></span></figcaption>";
      strip.appendChild(fig);
    }
    setText("dream-h", String(n));
  }

  function onSchema(s) {
    kind = s.kind || "vector";
    document.body.dataset.kind = kind;
    api.setSchema(s);
    if (kind === "video") buildStrip(s.dream_horizon || 8);
    if (kind === "vector" && !s.has_world_model) document.body.dataset.noWm = "1";
    buildCharts();
    // Panels just changed visibility — let net-viz re-measure its canvases.
    window.dispatchEvent(new Event("resize"));
  }

  function onFrame(fr) {
    lastFrameAt = performance.now();
    if (fr.cam) {
      $("cam").src = jpeg(fr.cam);
      $("cam").hidden = false;
      $("cam-empty").hidden = true;
    }
    setText("mode", fr.mode === "train" ? "training" : (fr.mode === "eval" ? "playback" : "—"));
    setText("episode", "ep " + (fr.episode != null ? fr.episode : "—"));
    setText("ep-reward", "R " + fmt(fr.ep_reward));
    setText("reward", "r " + fmt(fr.reward));

    api.applyFrame(fr);
    tdRecord(fr);
    statsFromFrame(fr);
    // net-viz prints the gauge-normalized value; show the raw variance instead.
    if (fr.aleatoric_raw != null) setText("ale", fmt(fr.aleatoric_raw, 4));
    if (fr.epistemic_raw != null) setText("epi", fmt(fr.epistemic_raw, 4));

    if (kind === "video") {
      if (fr.real_small) $("real-small").src = jpeg(fr.real_small);
      if (fr.recon) $("recon").src = jpeg(fr.recon);
      var tiles = document.querySelectorAll("#dream-strip .tile");
      (fr.dream || []).forEach(function (d, i) {
        var t = tiles[i];
        if (!t) return;
        t.querySelector("img").src = jpeg(d.img);
        t.querySelector(".ra").textContent = "r̂ " + fmt(d.reward) + " · ĉ " + fmt(d.cont);
        t.title = d.action;
        t.style.opacity = String(0.35 + 0.65 * Math.max(0, Math.min(1, d.cont)));
      });
    }
    if (phase !== "update" && phase !== "reset") showPhase("");
  }

  var updateTimer = null;
  function onStatus(st) {
    phase = st.phase;
    clearTimeout(updateTimer);
    // updates can run after every step — only show the banner if one actually stalls the feed
    if (st.phase === "update") updateTimer = setTimeout(function () { showPhase(st.text || "training world model…"); }, 700);
    else if (st.phase === "reset" || st.phase === "done") showPhase(st.text);
    else showPhase("");
  }

  // ── top-down map, heading-up: the agent always faces up, like the camera behind it.
  // Shows: real agent + pig (short trails), the ensemble-MEAN imagined path of each,
  // and at the pig's predicted end point K small dots (one per model → epistemic
  // spread) inside one ring of the models' predicted pig noise (aleatoric σ).
  var TD = { data: null, trail: [], episode: null, view: null, yaw: null };
  var C_AGENT = "184,212,168", C_PIG = "232,154,171", C_GOLD = "196,163,90";

  function tdRecord(fr) {
    var td = fr.topdown;
    if (!td) return;
    if (TD.episode !== fr.episode) { TD.trail = []; TD.episode = fr.episode; TD.view = null; TD.yaw = null; }
    TD.trail.push({ agent: td.agent, pig: td.pig });
    if (TD.trail.length > 20) TD.trail.shift();
    TD.data = td;
    var h = td.members && td.members[0] ? td.members[0].agent.length - 1 : 0;
    setText("td-h", String(h));
    setText("td-plan", td.plan ? "plan: " + td.plan.slice(0, 4).join(" → ") + (td.plan.length > 4 ? " …" : "") : "");
  }

  function meanPath(paths) {
    if (!paths.length || !paths[0]) return null;
    return paths[0].map(function (_, i) {
      var x = 0, z = 0;
      paths.forEach(function (p) { x += p[i][0]; z += p[i][1]; });
      return [x / paths.length, z / paths.length];
    });
  }

  function sizeTd(c) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var r = c.getBoundingClientRect();
    if (c.width !== Math.floor(r.width * dpr) || c.height !== Math.floor(r.height * dpr)) {
      c.width = Math.floor(r.width * dpr);
      c.height = Math.floor(r.height * dpr);
    }
    var ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: r.width, h: r.height };
  }

  function drawTopdown() {
    requestAnimationFrame(drawTopdown);
    var c = $("topdown");
    if (!c || !c.offsetParent) return;
    var s = sizeTd(c), ctx = s.ctx, w = s.w, h = s.h;
    ctx.clearRect(0, 0, w, h);
    var td = TD.data;
    ctx.font = "11px \"IBM Plex Mono\", monospace";
    if (!td) {
      ctx.fillStyle = "#7a8a72";
      ctx.fillText("waiting for predictions…", 14, 24);
      return;
    }

    // smooth the heading (shortest way round) so the map turns, not snaps
    var yawT = td.yaw;
    if (TD.yaw == null) TD.yaw = yawT;
    var dy = ((yawT - TD.yaw + 540) % 360) - 180;
    TD.yaw += dy * 0.2;
    var yaw = TD.yaw * Math.PI / 180;
    // Malmo: yaw 0 faces +z; forward f = (-sin, cos), agent's right r = (-cos, -sin) in (x, z)
    var fx = -Math.sin(yaw), fz = Math.cos(yaw), rx = -Math.cos(yaw), rz = -Math.sin(yaw);
    var ox = td.agent[0], oz = td.agent[1];
    function local(p) {           // world → (right, ahead) in blocks, agent at origin
      var dx = p[0] - ox, dz = p[1] - oz;
      return [dx * rx + dz * rz, dx * fx + dz * fz];
    }

    var members = td.members || [];
    var agentPath = meanPath(members.map(function (m) { return m.agent; }));
    var pigPaths = members.filter(function (m) { return m.pig; }).map(function (m) { return m.pig; });
    var pigPath = pigPaths.length ? meanPath(pigPaths) : null;
    var pigEnds = pigPaths.map(function (p) { return p[p.length - 1]; });
    var sig = 0;
    members.forEach(function (m) { if (m.pig && m.sigma.length) sig += m.sigma[m.sigma.length - 1]; });
    sig = pigPaths.length ? sig / pigPaths.length : 0;

    // fit everything that matters (in agent-local coords), ≥ 8 blocks ahead/behind
    var pts = [[0, 0]];
    if (td.pig) pts.push(local(td.pig));
    (agentPath || []).forEach(function (p) { pts.push(local(p)); });
    (pigPath || []).forEach(function (p) { pts.push(local(p)); });
    pigEnds.forEach(function (p) { pts.push(local(p)); });
    // agent sits at 66% height: room ahead = 0.6h, behind = 0.28h, sides = 0.45w
    var ppbT = 0.6 * h / 6;                       // at most ~6 blocks ahead when idle
    pts.forEach(function (p) {
      if (p[1] > 0) ppbT = Math.min(ppbT, 0.6 * h / p[1]);
      if (p[1] < 0) ppbT = Math.min(ppbT, 0.28 * h / -p[1]);
      if (p[0] !== 0) ppbT = Math.min(ppbT, 0.45 * w / Math.abs(p[0]));
    });
    TD.view = TD.view == null ? ppbT : TD.view + (ppbT - TD.view) * 0.1;
    var ppb = TD.view, cx = w / 2, cy = h * 0.66;
    function P(p) { var l = local(p); return [cx + l[0] * ppb, cy - l[1] * ppb]; }

    function line(pts, rgb, alpha, width, dash) {
      if (!pts || pts.length < 2) return;
      ctx.strokeStyle = "rgba(" + rgb + "," + alpha + ")";
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      pts.forEach(function (p, i) { var q = P(p); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    function dot(p, rgb, r, alpha) {
      var q = P(p);
      ctx.fillStyle = "rgba(" + rgb + "," + (alpha == null ? 1 : alpha) + ")";
      ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, Math.PI * 2); ctx.fill();
      return q;
    }
    function label(q, text, rgb) {
      ctx.fillStyle = "rgba(" + rgb + ",0.95)";
      ctx.textAlign = "left";
      ctx.fillText(text, q[0] + 10, q[1] + 4);
    }

    // faint "ahead" axis so the heading-up frame reads at a glance
    ctx.strokeStyle = "rgba(184,212,168,0.10)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, 8); ctx.stroke();

    // real trails (short, faint)
    line(TD.trail.map(function (t) { return t.agent; }), C_AGENT, 0.3, 2);
    var pigTrail = TD.trail.map(function (t) { return t.pig; }).filter(Boolean);
    line(pigTrail, C_PIG, 0.3, 2);

    // imagined (ensemble mean), dashed
    line(agentPath, C_GOLD, 0.85, 1.6, [5, 5]);
    if (td.pig && pigPath) line([td.pig].concat(pigPath), C_PIG, 0.85, 1.6, [5, 5]);

    // predicted pig end point: aleatoric ring + one dot per model (epistemic spread)
    if (pigPath) {
      var end = pigPath[pigPath.length - 1];
      var q = P(end), rr = Math.max(4, sig * ppb);
      ctx.strokeStyle = "rgba(" + C_PIG + ",0.55)";
      ctx.fillStyle = "rgba(" + C_PIG + ",0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(q[0], q[1], rr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      pigEnds.forEach(function (p) { dot(p, C_PIG, 2, 0.9); });
    }
    if (agentPath) dot(agentPath[agentPath.length - 1], C_GOLD, 2.5, 0.9);

    // now
    if (td.pig) label(dot(td.pig, C_PIG, 6), "pig", C_PIG);
    var a = dot(td.agent, C_AGENT, 6);
    ctx.strokeStyle = "rgb(" + C_AGENT + ")";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a[0], a[1] - 6); ctx.lineTo(a[0], a[1] - 13); ctx.stroke();
    label(a, "agent", C_AGENT);

    // scale bar
    var bar = 5 * ppb;
    ctx.strokeStyle = "rgba(216,210,196,0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(w - 14 - bar, h - 12); ctx.lineTo(w - 14, h - 12); ctx.stroke();
    ctx.fillStyle = "#7a8a72";
    ctx.textAlign = "right";
    ctx.fillText("5 blocks", w - 14, h - 18);
    ctx.textAlign = "left";
    ctx.fillText("↑ facing", 10, 16);
  }
  requestAnimationFrame(drawTopdown);

  // ── tabs: Inference (what the model predicts now) | Training (is it learning?) ──
  var tabChosenByUser = false;
  function selectTab(name, byUser) {
    if (byUser) tabChosenByUser = true;
    ["inference", "training"].forEach(function (t) {
      $("tab-" + t).setAttribute("aria-selected", String(t === name));
      $("panel-" + t).hidden = t !== name;
    });
    window.dispatchEvent(new Event("resize"));   // canvases in a newly shown tab re-measure
    markCharts();
  }
  document.querySelectorAll(".pred-tabs [data-tab]").forEach(function (b) {
    b.addEventListener("click", function () { selectTab(b.dataset.tab, true); });
  });

  // ── training stats ──
  var ST = { episode: [], update: [], unc: [], mode: null, frames: 0 };
  var BLUE = "#3987e5", ORANGE = "#d95926";   // validated categorical slots 1–2 on the dark surface

  function addStats(items) {
    items.forEach(function (it) {
      if (it.kind === "episode") ST.episode.push(it);
      else if (it.kind === "update") ST.update.push(it);
    });
    if (ST.episode.length > 5000) ST.episode.splice(0, ST.episode.length - 5000);
    if (ST.update.length > 5000) ST.update.splice(0, ST.update.length - 5000);
    updateTiles();
    markCharts();
  }

  function statsFromFrame(fr) {
    ST.frames += 1;
    if (ST.mode !== fr.mode) {
      ST.mode = fr.mode;
      setText("tab-hint", fr.mode === "train" ? "training run" : "checkpoint playback");
      $("train-note").hidden = fr.mode === "train";
      if (!tabChosenByUser) selectTab(fr.mode === "train" ? "training" : "inference", false);
    }
    // playback has no update() stats: chart uncertainty from the live frames instead
    if (fr.aleatoric_raw != null && !ST.update.length) {
      ST.unc.push({ x: ST.frames, aleatoric: fr.aleatoric_raw, epistemic: fr.epistemic_raw });
      if (ST.unc.length > 3000) ST.unc.shift();
      markCharts();
    }
    updateTiles();
  }

  function lastN(arr, n) { return arr.slice(Math.max(0, arr.length - n)); }
  function fmtNum(v) {
    if (v == null || isNaN(v)) return "—";
    var a = Math.abs(v);
    if (a >= 10000) return (v / 1000).toFixed(1) + "k";
    if (Number.isInteger(v)) return String(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(a >= 0.01 ? 3 : 4);
  }

  function updateTiles() {
    var eps = ST.episode, ups = ST.update, last = ups[ups.length - 1] || {};
    var recent = lastN(eps, 20);
    setText("st-episodes", eps.length ? String(eps[eps.length - 1].episode) : "—");
    setText("st-success", recent.length ? Math.round(100 * recent.filter(function (e) { return e.outcome === "killed"; }).length / recent.length) + "%" : "—");
    setText("st-return", recent.length ? fmtNum(recent.reduce(function (a, e) { return a + e["return"]; }, 0) / recent.length) : "—");
    setText("st-steps", last.env_steps != null ? fmtNum(last.env_steps) : (ST.frames ? fmtNum(ST.frames) : "—"));
    setText("st-updates", last.update != null ? fmtNum(last.update) : "—");
    setText("st-buffer", last.buffer != null ? fmtNum(last.buffer) : "—");
    ["st-updates", "st-buffer"].forEach(function (id) { $(id).parentNode.hidden = playbackOnly(); });
  }

  function movingAvg(vals, k) {
    var out = [], sum = 0;
    for (var i = 0; i < vals.length; i++) {
      sum += vals[i];
      if (i >= k) sum -= vals[i - k];
      out.push(sum / Math.min(i + 1, k));
    }
    return out;
  }

  // chart specs: x/series pulled from the stats store at draw time
  function chartSpecs() {
    var vec = kind !== "video";
    function upd(key) {
      return function () { return ST.update.filter(function (u) { return u[key] != null; }).map(function (u) { return { x: u.update, y: u[key] }; }); };
    }
    var specs = [{
      id: "returns", title: "Episode return", sub: "dots = episodes · line = 10-ep average", xLabel: "episode",
      get: function () {
        var pts = ST.episode.map(function (e) { return { x: e.episode, y: e["return"] }; });
        var ma = movingAvg(pts.map(function (p) { return p.y; }), 10);
        return [{ label: "return", color: BLUE, pts: pts, dots: true, line: false },
                { label: "10-ep avg", color: BLUE, pts: pts.map(function (p, i) { return { x: p.x, y: ma[i] }; }) }];
      }, empty: "no finished episodes yet"
    }];
    if (vec) {
      specs.push({
        id: "unc", title: "Uncertainty", sub: "pig-position variance (blocks²)", xLabel: function () { return ST.update.length ? "update" : "step"; }, legend: true,
        get: function () {
          var src = ST.update.length
            ? ST.update.filter(function (u) { return u.aleatoric != null; }).map(function (u) { return { x: u.update, a: u.aleatoric, e: u.epistemic }; })
            : ST.unc.map(function (u) { return { x: u.x, a: u.aleatoric, e: u.epistemic }; });
          return [{ label: "aleatoric", color: BLUE, pts: src.map(function (p) { return { x: p.x, y: p.a }; }) },
                  { label: "epistemic", color: ORANGE, pts: src.map(function (p) { return { x: p.x, y: p.e }; }) }];
        }, empty: "waiting for predictions"
      });
      specs.push({ id: "horizon", title: "Effective imagination horizon", sub: "steps trusted after aleatoric gating", xLabel: "update", needsUpdates: true,
                   get: function () { return [{ label: "steps", color: BLUE, pts: upd("eff_horizon")() }]; }, empty: "appears after the first update" });
      specs.push({ id: "wm", title: "World-model loss", sub: "lower = better predictions (NLL, can go negative)", xLabel: "update", needsUpdates: true,
                   get: function () { return [{ label: "loss", color: BLUE, pts: upd("wm_loss")() }]; }, empty: "appears after the first update" });
    } else {
      specs.push({ id: "wm", title: "World-model loss", sub: "pixel RSSM total", xLabel: "update", needsUpdates: true,
                   get: function () { return [{ label: "loss", color: BLUE, pts: upd("vwm_loss")() }]; }, empty: "appears after the first update" });
      specs.push({ id: "kl", title: "RSSM KL", sub: "posterior vs prior", xLabel: "update", needsUpdates: true,
                   get: function () { return [{ label: "KL", color: BLUE, pts: upd("vwm_kl")() }]; }, empty: "appears after the first update" });
    }
    specs.push({ id: "imag", title: "Imagined return", sub: "λ-return inside the world model", xLabel: "update", needsUpdates: true,
                 get: function () { return [{ label: "return", color: BLUE, pts: upd("imag_return")() }]; }, empty: "appears after the first update" });
    specs.push({ id: "entropy", title: "Policy entropy", sub: "exploration; falls as the policy commits", xLabel: "update", needsUpdates: true,
                 get: function () { return [{ label: "entropy", color: BLUE, pts: upd("entropy")() }]; }, empty: "appears after the first update" });
    return specs;
  }

  var CHARTS = [];
  function buildCharts() {
    var host = $("charts");
    host.innerHTML = "";
    CHARTS = chartSpecs().map(function (spec) {
      var card = document.createElement("figure");
      card.className = "chart-card";
      card.innerHTML =
        "<figcaption><span class=\"t\">" + spec.title + "</span><span class=\"v\"></span></figcaption>" +
        "<div class=\"sub\">" + spec.sub + "</div>" +
        (spec.legend ? "<div class=\"chart-legend\"></div>" : "") +
        "<div class=\"plot\"><canvas></canvas><div class=\"tip\" hidden></div></div>";
      host.appendChild(card);
      var c = { spec: spec, card: card, canvas: card.querySelector("canvas"), tip: card.querySelector(".tip"), hover: null, geom: null };
      c.canvas.addEventListener("mousemove", function (ev) {
        var r = c.canvas.getBoundingClientRect();
        c.hover = ev.clientX - r.left;
        drawChart(c);
      });
      c.canvas.addEventListener("mouseleave", function () { c.hover = null; c.tip.hidden = true; drawChart(c); });
      return c;
    });
    markCharts();
  }

  var chartsDirty = false;
  function markCharts() { chartsDirty = true; }
  (function chartLoop() {
    requestAnimationFrame(chartLoop);
    if (!chartsDirty || $("panel-training").hidden) return;
    chartsDirty = false;
    CHARTS.forEach(drawChart);
  })();

  function downsample(pts, n) {
    if (pts.length <= n) return pts;
    var out = [], step = pts.length / n;
    for (var i = 0; i < n; i++) out.push(pts[Math.floor(i * step)]);
    out.push(pts[pts.length - 1]);
    return out;
  }

  function playbackOnly() { return ST.mode === "eval" && !ST.update.length; }

  function xLabel(spec) { return typeof spec.xLabel === "function" ? spec.xLabel() : spec.xLabel; }

  function drawChart(c) {
    // loss curves only exist in a training run — don't show empty cards during playback
    c.card.hidden = !!c.spec.needsUpdates && playbackOnly();
    if (c.card.hidden) return;
    var s = sizeTd(c.canvas), ctx = s.ctx, w = s.w, h = s.h;
    ctx.clearRect(0, 0, w, h);
    var series = c.spec.get();
    var all = [];
    series.forEach(function (se) { se.pts = downsample(se.pts, 600); all = all.concat(se.pts); });
    var head = series[series.length - 1].pts;
    c.card.querySelector(".v").textContent = head.length ? fmtNum(head[head.length - 1].y) : "";
    var lg = c.card.querySelector(".chart-legend");
    if (lg) {
      lg.innerHTML = series.map(function (se) {
        var last = se.pts.length ? fmtNum(se.pts[se.pts.length - 1].y) : "—";
        return "<span><i style=\"background:" + se.color + "\"></i>" + se.label + " <b>" + last + "</b></span>";
      }).join("");
    }
    ctx.font = "10px \"IBM Plex Mono\", monospace";
    if (!all.length) {
      ctx.fillStyle = "#898781";
      ctx.fillText(c.spec.empty, 10, h / 2);
      return;
    }
    var xs = all.map(function (p) { return p.x; }), ys = all.map(function (p) { return p.y; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; }
    var pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
    if (x1 === x0) x1 = x0 + 1;
    var L = 44, R = 8, T = 6, B = 16;
    function X(x) { return L + (x - x0) / (x1 - x0) * (w - L - R); }
    function Y(y) { return T + (1 - (y - y0) / (y1 - y0)) * (h - T - B); }

    // recessive grid + y labels (3 lines), x extent labels
    ctx.strokeStyle = "#2c2c2a";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#898781";
    ctx.textAlign = "right";
    for (var g = 0; g <= 2; g++) {
      var gy = y0 + (y1 - y0) * g / 2;
      ctx.beginPath(); ctx.moveTo(L, Y(gy)); ctx.lineTo(w - R, Y(gy)); ctx.stroke();
      ctx.fillText(fmtNum(gy), L - 6, Y(gy) + 3);
    }
    ctx.textAlign = "left";
    ctx.fillText(xLabel(c.spec) + " " + fmtNum(x0), L, h - 3);
    ctx.textAlign = "right";
    ctx.fillText(fmtNum(x1), w - R, h - 3);

    series.forEach(function (se) {
      if (se.dots) {
        ctx.fillStyle = se.color;
        ctx.globalAlpha = 0.35;
        se.pts.forEach(function (p) { ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 2, 0, Math.PI * 2); ctx.fill(); });
        ctx.globalAlpha = 1;
      }
      if (se.line === false || se.pts.length < 2) return;
      ctx.strokeStyle = se.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      se.pts.forEach(function (p, i) { if (i) ctx.lineTo(X(p.x), Y(p.y)); else ctx.moveTo(X(p.x), Y(p.y)); });
      ctx.stroke();
    });

    // hover: crosshair + tooltip on the nearest x
    if (c.hover == null) return;
    var ref = series[0].pts, best = null;
    ref.forEach(function (p, i) { if (best == null || Math.abs(X(p.x) - c.hover) < Math.abs(X(ref[best].x) - c.hover)) best = i; });
    if (best == null) return;
    var px = X(ref[best].x);
    ctx.strokeStyle = "#383835";
    ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, h - B); ctx.stroke();
    var rows = series.map(function (se) {
      var p = se.pts[Math.min(best, se.pts.length - 1)];
      if (!p) return "";
      ctx.fillStyle = se.color;
      ctx.strokeStyle = "#12151a";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      return "<div><i style=\"background:" + se.color + "\"></i>" + se.label + " <b>" + fmtNum(p.y) + "</b></div>";
    }).join("");
    c.tip.innerHTML = "<div class=\"tx\">" + xLabel(c.spec) + " " + ref[best].x + "</div>" + rows;
    c.tip.hidden = false;
    c.tip.style.left = Math.min(px + 10, w - 130) + "px";
  }

  function connect() {
    var es = new EventSource("/events");
    es.onopen = function () { $("conn").classList.add("on"); $("conn").title = "connected"; };
    es.onerror = function () { $("conn").classList.remove("on"); $("conn").title = "reconnecting…"; };
    es.addEventListener("schema", function (e) { onSchema(JSON.parse(e.data)); });
    es.addEventListener("frame", function (e) { onFrame(JSON.parse(e.data)); });
    es.addEventListener("status", function (e) { onStatus(JSON.parse(e.data)); });
    es.addEventListener("stats", function (e) { addStats([JSON.parse(e.data)]); });
    es.addEventListener("stats_history", function (e) {
      ST.episode = []; ST.update = [];
      addStats(JSON.parse(e.data));
    });
  }

  // Stale feed hint (env hung, or Malmo restarting a mission).
  setInterval(function () {
    if (!lastFrameAt || phase === "update" || phase === "reset") return;
    if (performance.now() - lastFrameAt > 4000) showPhase("waiting for env…");
  }, 1000);

  connect();
})();
