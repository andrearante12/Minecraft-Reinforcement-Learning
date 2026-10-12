/* Actor-critic replay: JSON dump or synthetic episode. Optional dream fan. */
(function (global) {
  "use strict";

  var COLORS = {
    cell: "#1a1e24",
    gold: "#c4a35a",
    hud: "#b8d4a8",
    dim: "#7a8a72",
    line: "#2a2d32",
    moss: "#3f6b3a",
    red: "#a63d3d",
    lapis: "#3d5a8a"
  };

  function $(id) { return document.getElementById(id); }
  function lerp(a, b, t) { return a + (b - a) * t; }

  function chevron(ctx, x, y0, y1, rgb, t, reduceMotion) {
    if (y1 - y0 < 12) return;
    ctx.strokeStyle = "rgba(" + rgb + ",0.55)";
    ctx.lineWidth = 1.6;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
    ctx.fillStyle = "rgba(" + rgb + ",0.95)";
    ctx.beginPath();
    ctx.moveTo(x, y1 + 1);
    ctx.lineTo(x - 5, y1 - 7);
    ctx.lineTo(x + 5, y1 - 7);
    ctx.closePath();
    ctx.fill();
    if (reduceMotion) return;
    var u = (t * 0.45) % 1;
    var y = y0 + (y1 - y0) * u;
    ctx.beginPath();
    ctx.arc(x, y, 2.3, 0, Math.PI * 2);
    ctx.fill();
  }

  function mount(opts) {
    opts = opts || {};
    var canvas = $("canvas");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var dreamCanvas = $("dream");
    var dctx = dreamCanvas ? dreamCanvas.getContext("2d") : null;
    var schema = null;
    var frame = null;
    var frames = [];
    var idx = 0;
    var playing = true;
    var prevActs = {};
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var timer = null;
    var dt = opts.dt || 150;
    var meta = {};
    var synced = false;
    var video = document.querySelector(".demo-stage .clip-slot video");

    function sizeCanvas(c, minH) {
      if (!c) return;
      var parent = c.parentElement;
      var r = parent.getBoundingClientRect();
      var title = parent.querySelector("h2");
      var trim = title ? title.getBoundingClientRect().height + 8 : 0;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.max(120, r.width);
      var h = Math.max(minH || 200, (r.height || minH || 200) - trim);
      c.width = Math.floor(w * dpr);
      c.height = Math.floor(h * dpr);
      c.style.width = w + "px";
      c.style.height = h + "px";
      var cx = c.getContext("2d");
      cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function initSchema(s) {
      schema = s;
      if ($("env")) $("env").textContent = "env " + (s.env || "—");
      var inputsEl = $("input-fields");
      inputsEl.innerHTML = "";
      (s.input_fields || []).forEach(function (f) {
        var d = document.createElement("div");
        d.className = "field";
        var extra = f.key === "vox" ? "<div class=\"vox\" id=\"vox\"></div>" : "<div class=\"vals\" data-key=\"" + f.key + "\"></div>";
        d.innerHTML = "<div class=\"lbl\">" + f.label + "</div>" + extra;
        inputsEl.appendChild(d);
      });
      var vox = $("vox");
      if (vox && s.vox) {
        vox.style.gridTemplateColumns = "repeat(" + s.vox.cols + ", 1fr)";
        var n = s.vox.cols * s.vox.rows;
        for (var i = 0; i < n; i++) vox.appendChild(document.createElement("span"));
      }
      var barsEl = $("action-bars");
      barsEl.innerHTML = "";
      (s.actions || []).forEach(function (name, i) {
        var row = document.createElement("div");
        row.className = "abar";
        row.dataset.i = String(i);
        row.innerHTML =
          "<span class=\"name\">" + name + "</span>" +
          "<div class=\"track\"><div class=\"fill\"></div></div>" +
          "<span class=\"pct\">0</span>";
        barsEl.appendChild(row);
      });
      var scrub = $("scrub");
      if (scrub) {
        scrub.max = String(Math.max(0, frames.length - 1));
        scrub.value = "0";
      }
    }

    function setInputs(inputs) {
      if (!inputs || !schema) return;
      (schema.input_fields || []).forEach(function (f) {
        if (f.key === "vox") {
          var cells = document.querySelectorAll("#vox span");
          var grid = inputs.vox || [];
          cells.forEach(function (el, i) {
            el.classList.toggle("on", !!grid[i]);
          });
          return;
        }
        var wrap = document.querySelector("#input-fields [data-key=\"" + f.key + "\"]");
        if (!wrap) return;
        wrap.innerHTML = "";
        var val = inputs[f.key];
        var isBool = f.kind === "bool" || (
          typeof val === "number" && (val === 0 || val === 1) &&
          /on_ground|ray|visible|in_range/.test(f.key)
        );
        if (isBool) {
          var chip = document.createElement("span");
          chip.className = "chip " + (val > 0.5 ? "bool-on" : "bool-off");
          chip.textContent = val > 0.5 ? "true" : "false";
          wrap.appendChild(chip);
          return;
        }
        var arr = Array.isArray(val) ? val : [val];
        arr.forEach(function (v) {
          var chip = document.createElement("span");
          chip.className = "chip";
          chip.textContent = typeof v === "number" ? v.toFixed(2) : String(v);
          wrap.appendChild(chip);
        });
      });
    }

    function setBars(fr) {
      if (!schema) return;
      document.querySelectorAll("#action-bars .abar").forEach(function (row, i) {
        var p = (fr.probs && fr.probs[i]) || 0;
        row.classList.toggle("chosen", i === fr.action);
        row.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%";
        row.querySelector(".pct").textContent = Math.round(p * 100) + "";
      });
      if ($("value-fill")) $("value-fill").style.width = ((fr.value_norm || 0) * 100).toFixed(1) + "%";
      if ($("value-num")) $("value-num").textContent = fr.value != null ? fr.value.toFixed(2) : "—";
      if ($("value-min")) $("value-min").textContent = fr.value_min != null ? fr.value_min.toFixed(2) : "—";
      if ($("value-max")) $("value-max").textContent = fr.value_max != null ? fr.value_max.toFixed(2) : "—";
      if ($("step")) $("step").textContent = "step " + (fr.step || 0);
      if ($("action")) $("action").textContent = fr.action_name || "—";
      if ($("scrub-label")) $("scrub-label").textContent = (idx + 1) + " / " + frames.length;
      if ($("ale")) $("ale").textContent = fr.aleatoric != null ? fr.aleatoric.toFixed(3) : "—";
      if ($("epi")) $("epi").textContent = fr.epistemic != null ? fr.epistemic.toFixed(3) : "—";
      if ($("ale-fill")) $("ale-fill").style.width = ((fr.aleatoric || 0) * 100).toFixed(1) + "%";
      if ($("epi-fill")) $("epi-fill").style.width = ((fr.epistemic || 0) * 100).toFixed(1) + "%";
    }

    function tween(id, vals) {
      var prev = prevActs[id] || vals;
      var k = reduce ? 1 : 0.35;
      var mixed = [];
      for (var i = 0; i < vals.length; i++) mixed.push(lerp(prev[i] || 0, vals[i] || 0, k));
      prevActs[id] = mixed;
      return mixed;
    }

    function sample(vals, n) {
      if (!vals || !vals.length) return Array(n).fill(0.15);
      if (vals.length === n) return vals;
      var out = [];
      for (var i = 0; i < n; i++) {
        var j = Math.round(i * (vals.length - 1) / Math.max(n - 1, 1));
        out.push(vals[j] || 0);
      }
      return out;
    }

    function stateCells() {
      var inputs = (frame && frame.inputs) || {};
      if (inputs.vox && inputs.vox.length) return sample(inputs.vox.map(function (v) { return v ? 1 : 0.08; }), 16);
      var acts = (frame && frame.activations) || {};
      var layers = schema.layers || [];
      var bag = [];
      layers.forEach(function (layer) {
        if (layer.group === "head") return;
        var vals = acts[layer.id] || [];
        for (var i = 0; i < vals.length; i++) bag.push(vals[i]);
      });
      if (!bag.length && layers[0]) bag = acts[layers[0].id] || [];
      return sample(bag, 16);
    }

    function hiddenCells() {
      var acts = (frame && frame.activations) || {};
      var layers = schema.layers || [];
      var layer = null;
      for (var i = 0; i < layers.length; i++) {
        if (layers[i].id === "actor_hidden" || /actor|policy/i.test(layers[i].label || "")) {
          layer = layers[i];
          break;
        }
      }
      if (!layer) layer = layers[0];
      if (!layer) return sample([], 12);
      return sample(tween(layer.id, acts[layer.id] || []), 12);
    }

    function drawGrid(vals, x, y, cols, cell, rgb) {
      var gap = 4;
      var rows = Math.max(1, Math.ceil(vals.length / cols));
      var gw = cols * (cell + gap) - gap;
      var gh = rows * (cell + gap) - gap;
      var gx = x - gw / 2;
      var gy = y;
      for (var i = 0; i < vals.length; i++) {
        ctx.fillStyle = "rgba(" + rgb + "," + (0.1 + 0.9 * (vals[i] || 0)) + ")";
        ctx.fillRect(gx + (i % cols) * (cell + gap), gy + Math.floor(i / cols) * (cell + gap), cell, cell);
      }
      return { x: gx, y: gy, w: gw, h: gh };
    }

    function cellFit(cols, rows, maxW, maxH, cap) {
      var gap = 4;
      var byW = Math.floor((maxW - (cols - 1) * gap) / cols);
      var byH = Math.floor((maxH - (rows - 1) * gap) / rows);
      return Math.max(6, Math.min(cap, byW, byH));
    }

    function drawNet() {
      if (!schema) return;
      var w = canvas.clientWidth;
      var h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      var t = performance.now() / 1000;
      var gold = "196,163,90";
      var hud = "184,212,168";
      var cx = w * 0.5;
      var name = (frame && frame.action_name) || "—";
      var pad = 10;
      var labelH = 16;
      var arrowGap = 26;
      var pillH = 38;
      var maxW = w - 28;

      var stateVals = stateCells();
      var hidVals = hiddenCells();
      var stateCols = 4;
      var hidCols = 6;
      var stateRows = Math.max(1, Math.ceil(stateVals.length / stateCols));
      var hidRows = Math.max(1, Math.ceil(hidVals.length / hidCols));

      var chrome = pad + labelH + arrowGap + labelH + arrowGap + labelH + pillH + pad;
      var gridSpace = Math.max(48, h - chrome);
      var stateH = gridSpace * (stateRows / (stateRows + hidRows));
      var hidH = gridSpace - stateH;
      var stateCell = cellFit(stateCols, stateRows, maxW, stateH, 18);
      var hidCell = cellFit(hidCols, hidRows, maxW, hidH, 16);

      function label(text, y) {
        ctx.font = "10px \"IBM Plex Mono\", monospace";
        ctx.textAlign = "center";
        ctx.fillStyle = COLORS.dim;
        ctx.fillText(text, cx, y);
      }

      var y = pad;
      label("state", y + 10);
      y += labelH;
      var stateBox = drawGrid(stateVals, cx, y, stateCols, stateCell, gold);
      y = stateBox.y + stateBox.h;
      chevron(ctx, cx, y + 4, y + arrowGap - 6, gold, t, reduce);
      y += arrowGap;

      label("net", y + 10);
      y += labelH;
      var hidBox = drawGrid(hidVals, cx, y, hidCols, hidCell, hud);
      y = hidBox.y + hidBox.h;
      chevron(ctx, cx, y + 4, y + arrowGap - 6, gold, t + 0.4, reduce);
      y += arrowGap;

      label("action", y + 10);
      y += labelH;
      var pillW = Math.min(maxW, Math.max(100, name.length * 8.5 + 24));
      var px = cx - pillW / 2;
      var py = y;
      ctx.fillStyle = "rgba(" + gold + ",0.16)";
      ctx.strokeStyle = "rgba(" + gold + ",0.85)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(px, py, pillW, pillH, 4);
      else ctx.rect(px, py, pillW, pillH);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = COLORS.gold;
      ctx.font = "13px \"IBM Plex Mono\", monospace";
      ctx.textAlign = "center";
      ctx.fillText(name, cx, py + 25);
    }

    function drawDream() {
      if (!dctx || !dreamCanvas || !frame || !frame.fan) return;
      var w = dreamCanvas.clientWidth;
      var h = dreamCanvas.clientHeight;
      dctx.clearRect(0, 0, w, h);
      var fan = frame.fan;
      var palette = [COLORS.red, COLORS.gold, COLORS.moss, COLORS.lapis, "#8a8680"];
      dctx.strokeStyle = COLORS.line;
      dctx.beginPath();
      dctx.moveTo(28, h - 24);
      dctx.lineTo(w - 16, h - 24);
      dctx.moveTo(28, 16);
      dctx.lineTo(28, h - 24);
      dctx.stroke();
      dctx.fillStyle = COLORS.dim;
      dctx.font = "11px \"IBM Plex Mono\", monospace";
      dctx.textAlign = "left";
      dctx.fillText("imagined pig Δ  ·  ensemble K=" + fan.length, 36, 18);
      fan.forEach(function (path, k) {
        dctx.strokeStyle = palette[k % palette.length];
        dctx.lineWidth = 1.6;
        dctx.beginPath();
        path.forEach(function (p, i) {
          var x = 28 + (i / Math.max(path.length - 1, 1)) * (w - 56);
          var y = (h - 24) - p * (h - 48);
          if (i === 0) dctx.moveTo(x, y);
          else dctx.lineTo(x, y);
        });
        dctx.stroke();
        var last = path[path.length - 1];
        dctx.fillStyle = palette[k % palette.length];
        dctx.beginPath();
        dctx.arc(28 + (w - 56), (h - 24) - last * (h - 48), 3.2, 0, Math.PI * 2);
        dctx.fill();
      });
    }

    function applyFrame(fr) {
      frame = fr;
      setInputs(fr.inputs);
      setBars(fr);
      var scrub = $("scrub");
      if (scrub && document.activeElement !== scrub) scrub.value = String(idx);
    }

    function goto(i, seekVideo) {
      if (!frames.length) return;
      idx = ((i % frames.length) + frames.length) % frames.length;
      applyFrame(frames[idx]);
      if (synced && video && seekVideo !== false) {
        var target = frames[idx].t_sec;
        if (target != null && Math.abs(video.currentTime - target) > 0.025) {
          video.currentTime = target;
        }
      }
    }

    function tick() {
      if (!playing || reduce) return;
      if (synced && video) {
        var stepSeconds = Number(meta.step_duration) || (dt / 1000);
        var videoIdx = Math.min(
          frames.length - 1,
          Math.max(0, Math.floor((video.currentTime + stepSeconds * 0.25) / stepSeconds))
        );
        if (videoIdx !== idx) goto(videoIdx, false);
        return;
      }
      goto(idx + 1, false);
    }

    function setPlaying(on) {
      playing = on;
      var btn = $("play");
      if (btn) btn.textContent = playing ? "pause" : "play";
      if (synced && video) {
        if (playing) {
          var promise = video.play();
          if (promise && promise.catch) {
            promise.catch(function () {
              playing = false;
              if (btn) btn.textContent = "play";
            });
          }
        } else {
          video.pause();
        }
      }
    }

    sizeCanvas(canvas, 240);
    sizeCanvas(dreamCanvas, 140);
    window.addEventListener("resize", function () {
      sizeCanvas(canvas, 240);
      sizeCanvas(dreamCanvas, 140);
      drawNet();
      drawDream();
    });

    (function loop() {
      drawNet();
      drawDream();
      requestAnimationFrame(loop);
    })();

    var playBtn = $("play");
    if (playBtn) {
      playBtn.addEventListener("click", function () { setPlaying(!playing); });
    }
    var scrub = $("scrub");
    if (scrub) {
      scrub.addEventListener("input", function () {
        setPlaying(false);
        goto(parseInt(scrub.value, 10) || 0, true);
      });
    }

    function start(data) {
      frames = data.frames || [];
      meta = data.meta || {};
      synced = !!(meta.synced && video && frames.length);
      if (synced) {
        dt = (Number(meta.step_duration) || 0.15) * 1000;
        video.loop = true;
        video.muted = true;
        video.removeAttribute("controls");
      }
      initSchema(data.schema);
      goto(0, true);
      if (timer) clearInterval(timer);
      if (!reduce) timer = setInterval(tick, synced ? 30 : dt);
      setPlaying(playing && !reduce);
    }

    if (opts.source === "push") {
      // An external driver (js/live.js) feeds schema + frames as they arrive.
      return {
        setSchema: initSchema,
        applyFrame: function (fr) { frames = [fr]; idx = 0; applyFrame(fr); }
      };
    }

    if (opts.source === "ws") {
      var proto = location.protocol === "https:" ? "wss" : "ws";
      var ws = new WebSocket(proto + "://" + location.host + "/ws");
      ws.onmessage = function (ev) {
        var msg = JSON.parse(ev.data);
        if (msg.type === "schema") initSchema(msg);
        else if (msg.type === "frame") applyFrame(msg);
      };
      return;
    }

    var url = opts.url;
    var synthName = opts.synth || "parkour";
    if (url) {
      fetch(url)
        .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
        .then(start)
        .catch(function () { start(synthesize(synthName)); });
    } else {
      start(synthesize(synthName));
    }
  }

  function layers() {
    return [
      { id: "proprio", label: "Proprio", group: "stream", size: 64, display: 16 },
      { id: "goal", label: "Goal / tgt", group: "stream", size: 64, display: 16 },
      { id: "voxel", label: "Voxel", group: "stream", size: 128, display: 16 },
      { id: "actor_hidden", label: "Actor hidden", group: "head", size: 64, display: 16 },
      { id: "critic_hidden", label: "Critic hidden", group: "head", size: 64, display: 16 }
    ];
  }

  function activations(s, L) {
    var out = {}, energy = {};
    L.forEach(function (layer) {
      var arr = [];
      for (var i = 0; i < layer.display; i++) {
        arr.push(Math.max(0, 0.28 + 0.5 * Math.sin((s + i * 1.7) * 0.22 + layer.size * 0.01)));
      }
      out[layer.id] = arr;
      energy[layer.id] = arr.reduce(function (a, b) { return a + b; }, 0) / arr.length;
    });
    return { activations: out, stream_energy: energy };
  }

  function peaked(n, act) {
    return Array.from({ length: n }, function (_, i) {
      return i === act ? 0.58 : 0.42 / (n - 1);
    });
  }

  function synthesize(kind) {
    if (kind === "bridging") return synthesizeBridging();
    if (kind === "hunting") return synthesizeHunting();
    return synthesizeParkour();
  }

  function synthesizeParkour() {
    var actions = [
      "move_forward", "move_backward", "strafe_left", "strafe_right",
      "sprint_forward", "jump", "sprint_jump", "jump_forward",
      "sprint_jump_left", "sprint_jump_right", "look_down", "look_up",
      "turn_left", "turn_right", "no_op"
    ];
    var L = layers();
    L[1].label = "Goal Δ";
    var cols = 5, rows = 6;
    var schema = {
      type: "schema", env: "simple_jump", actions: actions, layers: L,
      vox: { cols: cols, rows: rows },
      input_fields: [
        { key: "on_ground", label: "On ground", kind: "bool" },
        { key: "vel", label: "Velocity (Δy,Δx,Δz)" },
        { key: "goal", label: "Goal Δ (dx,dy,dz)" },
        { key: "vox", label: "Voxels at feet  5×6" }
      ]
    };
    var seq = [4, 4, 6, 6, 6, 0, 0, 0];
    var frames = [];
    for (var s = 0; s < 64; s++) {
      var act = seq[s % seq.length];
      var z = s % 8;
      var vox = [];
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          var occ = r <= 2 || r >= 5;
          if (r === 3 || r === 4) occ = false;
          vox.push(occ ? 1 : 0);
        }
      }
      var a = activations(s, L);
      frames.push({
        type: "frame", step: s + 1, action: act, action_name: actions[act],
        probs: peaked(actions.length, act),
        value: 1.8 + 0.5 * Math.sin(s * 0.2),
        value_norm: 0.4 + 0.25 * Math.sin(s * 0.2),
        value_min: 0.2, value_max: 4.0,
        activations: a.activations, stream_energy: a.stream_energy,
        inputs: {
          on_ground: z < 5 ? 1 : 0,
          vel: [z < 5 ? 0 : 0.45, 0.02, -0.28],
          goal: [0, z < 6 ? -1 : 0, -2.2 + s * 0.03],
          vox: vox
        }
      });
    }
    return { schema: schema, frames: frames };
  }

  function synthesizeBridging() {
    var actions = [
      "move_forward", "move_backward", "strafe_left", "strafe_right",
      "look_down", "look_up", "turn_left", "turn_right",
      "sneak_down", "sneak_up", "place_block", "no_op"
    ];
    var L = layers();
    L[1].label = "Goal Δ";
    var cols = 5, rows = 8;
    var schema = {
      type: "schema", env: "bridging", actions: actions, layers: L,
      vox: { cols: cols, rows: rows },
      input_fields: [
        { key: "on_ground", label: "On ground", kind: "bool" },
        { key: "inv", label: "Inventory" },
        { key: "ray", label: "Ray hit", kind: "bool" },
        { key: "goal", label: "Goal Δ (dx,dy,dz)" },
        { key: "vox", label: "Bridge voxels  5×8" }
      ]
    };
    var seq = [8, 4, 0, 10, 0, 10, 0, 10, 0, 9];
    var frames = [];
    for (var s = 0; s < 72; s++) {
      var act = seq[s % seq.length];
      var placed = Math.min(5, Math.floor(s / 8));
      var vox = [];
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          var occ = r === 0 || r === 7 || (r > 0 && r < 7 && r <= placed && c === 2);
          vox.push(occ ? 1 : 0);
        }
      }
      var a = activations(s, L);
      frames.push({
        type: "frame", step: s + 1, action: act, action_name: actions[act],
        probs: peaked(actions.length, act),
        value: 0.6 + placed * 0.4,
        value_norm: 0.2 + placed * 0.12,
        value_min: -2, value_max: 8,
        activations: a.activations, stream_energy: a.stream_energy,
        inputs: {
          on_ground: 1,
          inv: (64 - placed) / 64,
          ray: act === 10 ? 1 : 0,
          goal: [0.4, 0, 5.5 - placed],
          vox: vox
        }
      });
    }
    return { schema: schema, frames: frames };
  }

  function synthesizeHunting() {
    var actions = [
      "move_forward", "move_backward", "strafe_left", "strafe_right",
      "sprint_forward", "turn_left", "turn_right", "look_up", "look_down",
      "attack", "no_op"
    ];
    var L = layers();
    L[1].label = "Target";
    var schema = {
      type: "schema", env: "hunting", actions: actions, layers: L,
      input_fields: [
        { key: "visible", label: "Pig visible", kind: "bool" },
        { key: "dist", label: "Distance" },
        { key: "heading", label: "Heading err" },
        { key: "in_range", label: "In range", kind: "bool" }
      ]
    };
    var seq = [6, 6, 4, 4, 0, 6, 9, 9, 0];
    var frames = [];
    for (var s = 0; s < 80; s++) {
      var act = seq[s % seq.length];
      var t = s / 80;
      var a = activations(s, L);
      var fan = [];
      for (var k = 0; k < 5; k++) {
        var path = [];
        var y = 0.15;
        for (var h = 0; h < 12; h++) {
          y += 0.04 + 0.02 * k + 0.01 * Math.sin((s + h + k) * 0.4);
          path.push(Math.min(1, y));
        }
        fan.push(path);
      }
      frames.push({
        type: "frame", step: s + 1, action: act, action_name: actions[act],
        probs: peaked(actions.length, act),
        value: 4 + 8 * t,
        value_norm: t,
        value_min: 0, value_max: 14,
        activations: a.activations, stream_energy: a.stream_energy,
        aleatoric: 0.15 + 0.35 * Math.min(1, t * 1.4),
        epistemic: 0.45 * (1 - t) + 0.05,
        fan: fan,
        inputs: {
          visible: 1,
          dist: Math.max(0.05, 0.9 - t),
          heading: 0.25 * Math.sin(s * 0.3),
          in_range: t > 0.7 ? 1 : 0
        }
      });
    }
    return { schema: schema, frames: frames };
  }

  global.NetViz = { mount: mount, synthesize: synthesize };
})(window);
