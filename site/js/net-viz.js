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

    function sizeCanvas(c, minH) {
      if (!c) return;
      var parent = c.parentElement;
      var r = parent.getBoundingClientRect();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.max(120, r.width);
      var h = Math.max(minH || 200, r.height || minH || 200);
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

    function drawNet() {
      if (!schema) return;
      var w = canvas.clientWidth;
      var h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      var layers = schema.layers || [];
      var streams = layers.filter(function (l) { return l.group === "stream"; });
      var heads = layers.filter(function (l) { return l.group === "head"; });
      var acts = (frame && frame.activations) || {};
      var energy = (frame && frame.stream_energy) || {};
      var colW = w / Math.max(streams.length, 1);

      streams.forEach(function (layer, si) {
        var cx = colW * (si + 0.5);
        var vals = acts[layer.id] || [];
        var prev = prevActs[layer.id] || vals;
        var n = layer.display || vals.length || 16;
        var cols = 4;
        var rows = Math.ceil(n / cols);
        var cell = Math.min(16, (h * 0.34) / rows);
        var gridW = cols * (cell + 3);
        var x0 = cx - gridW / 2;
        var y0 = 32;
        ctx.fillStyle = COLORS.dim;
        ctx.font = "11px \"IBM Plex Mono\", monospace";
        ctx.textAlign = "center";
        ctx.fillText(layer.label, cx, 18);
        for (var i = 0; i < n; i++) {
          var v = lerp(prev[i] || 0, vals[i] || 0, reduce ? 1 : 0.35);
          ctx.fillStyle = "rgba(196,163,90," + (0.12 + 0.88 * v) + ")";
          ctx.fillRect(x0 + (i % cols) * (cell + 3), y0 + Math.floor(i / cols) * (cell + 3), cell, cell);
        }
        ctx.strokeStyle = "rgba(63,107,58," + (0.25 + (energy[layer.id] || 0)) + ")";
        ctx.strokeRect(x0 - 3, y0 - 3, gridW + 3, rows * (cell + 3) + 1);
        prevActs[layer.id] = vals.map(function (v, i) {
          return lerp(prev[i] || 0, v, reduce ? 1 : 0.35);
        });
      });

      ctx.strokeStyle = COLORS.line;
      ctx.beginPath();
      ctx.moveTo(w * 0.12, h * 0.48);
      ctx.lineTo(w * 0.5, h * 0.55);
      ctx.lineTo(w * 0.88, h * 0.48);
      ctx.stroke();
      ctx.fillStyle = COLORS.dim;
      ctx.fillText("concat", w * 0.5, h * 0.58);

      heads.forEach(function (layer, hi) {
        var cx = w * (hi === 0 ? 0.32 : 0.68);
        var y0 = h * 0.66;
        var vals = acts[layer.id] || [];
        var prev = prevActs[layer.id] || vals;
        var n = layer.display || vals.length || 16;
        ctx.fillStyle = COLORS.dim;
        ctx.fillText(layer.label, cx, y0 - 10);
        var cell = 9;
        var x0 = cx - (n * (cell + 2)) / 2;
        for (var i = 0; i < n; i++) {
          var v = lerp(prev[i] || 0, vals[i] || 0, reduce ? 1 : 0.35);
          ctx.fillStyle = "rgba(184,212,168," + (0.18 + 0.82 * v) + ")";
          ctx.fillRect(x0 + i * (cell + 2), y0, cell, 26);
        }
        prevActs[layer.id] = vals.map(function (v, i) {
          return lerp(prev[i] || 0, v, reduce ? 1 : 0.35);
        });
      });
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

    function goto(i) {
      if (!frames.length) return;
      idx = ((i % frames.length) + frames.length) % frames.length;
      applyFrame(frames[idx]);
    }

    function tick() {
      if (!playing || reduce) return;
      goto(idx + 1);
    }

    function setPlaying(on) {
      playing = on;
      var btn = $("play");
      if (btn) btn.textContent = playing ? "pause" : "play";
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
        goto(parseInt(scrub.value, 10) || 0);
      });
    }

    function start(data) {
      frames = data.frames || [];
      initSchema(data.schema);
      goto(0);
      if (timer) clearInterval(timer);
      if (!reduce) timer = setInterval(tick, dt);
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
