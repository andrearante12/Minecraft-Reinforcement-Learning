/* Shared ActorCritic schematic: JSON replay or live WebSocket. */
(function (global) {
  "use strict";

  var COLORS = {
    bg: "#12151a",
    cell: "#1a1e24",
    moss: "#3f6b3a",
    gold: "#c4a35a",
    hud: "#b8d4a8",
    dim: "#7a8a72",
    line: "#2a2d32",
    ink: "#d8d2c4"
  };

  function $(id) { return document.getElementById(id); }

  function lerp(a, b, t) { return a + (b - a) * t; }

  function mount(opts) {
    opts = opts || {};
    var canvas = $("canvas");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var schema = null;
    var frame = null;
    var prevActs = {};
    var barsEl = $("action-bars");
    var inputsEl = $("input-fields");
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var raf = 0;

    function sizeCanvas() {
      var parent = canvas.parentElement;
      var r = parent.getBoundingClientRect();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.max(120, r.width);
      var h = Math.max(200, r.height || 320);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function initSchema(s) {
      schema = s;
      if ($("env")) $("env").textContent = "env " + (s.env || "—");
      inputsEl.innerHTML = "";
      (s.input_fields || []).forEach(function (f) {
        var d = document.createElement("div");
        d.className = "field";
        d.innerHTML = "<div class=\"lbl\">" + f.label + "</div><div class=\"vals\" data-key=\"" + f.key + "\"></div>";
        inputsEl.appendChild(d);
      });
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
    }

    function setInputs(inputs) {
      if (!inputs || !schema) return;
      (schema.input_fields || []).forEach(function (f) {
        var wrap = inputsEl.querySelector("[data-key=\"" + f.key + "\"]");
        if (!wrap) return;
        wrap.innerHTML = "";
        var val = inputs[f.key];
        if (f.key === "on_ground") {
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
      var rows = barsEl.querySelectorAll(".abar");
      rows.forEach(function (row, i) {
        var p = (fr.probs && fr.probs[i]) || 0;
        row.classList.toggle("chosen", i === fr.action);
        row.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%";
        row.querySelector(".pct").textContent = Math.round(p * 100) + "";
      });
      if ($("value-fill")) $("value-fill").style.width = ((fr.value_norm || 0) * 100).toFixed(1) + "%";
      if ($("value-num")) $("value-num").textContent = (fr.value != null) ? fr.value.toFixed(2) : "—";
      if ($("value-min")) $("value-min").textContent = (fr.value_min != null) ? fr.value_min.toFixed(2) : "—";
      if ($("value-max")) $("value-max").textContent = (fr.value_max != null) ? fr.value_max.toFixed(2) : "—";
      if ($("step")) $("step").textContent = "step " + (fr.step || 0);
      if ($("action")) $("action").textContent = fr.action_name || "—";
    }

    function drawNet() {
      if (!schema) return;
      var w = canvas.clientWidth;
      var h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      var layers = schema.layers || [];
      var streams = layers.filter(function (l) { return l.group === "stream"; });
      var heads = layers.filter(function (l) { return l.group === "head"; });
      var colW = w / Math.max(streams.length + 1, 2);
      var acts = (frame && frame.activations) || {};
      var energy = (frame && frame.stream_energy) || {};

      streams.forEach(function (layer, si) {
        var cx = colW * (si + 0.5);
        var vals = acts[layer.id] || [];
        var prev = prevActs[layer.id] || vals;
        var n = layer.display || vals.length || 16;
        var cols = 4;
        var rows = Math.ceil(n / cols);
        var cell = Math.min(18, (h * 0.38) / rows);
        var gridW = cols * (cell + 3);
        var gridH = rows * (cell + 3);
        var x0 = cx - gridW / 2;
        var y0 = 36;
        ctx.fillStyle = COLORS.dim;
        ctx.font = "11px \"IBM Plex Mono\", monospace";
        ctx.textAlign = "center";
        ctx.fillText(layer.label, cx, 22);
        for (var i = 0; i < n; i++) {
          var v = lerp(prev[i] || 0, vals[i] || 0, reduce ? 1 : 0.35);
          var c = i % cols;
          var r = Math.floor(i / cols);
          var x = x0 + c * (cell + 3);
          var y = y0 + r * (cell + 3);
          ctx.fillStyle = COLORS.cell;
          ctx.fillRect(x, y, cell, cell);
          ctx.fillStyle = "rgba(196,163,90," + (0.15 + 0.85 * v) + ")";
          ctx.fillRect(x, y, cell, cell);
        }
        var glow = energy[layer.id] || 0;
        ctx.strokeStyle = "rgba(63,107,58," + (0.2 + glow) + ")";
        ctx.strokeRect(x0 - 4, y0 - 4, gridW + 5, gridH + 5);
        prevActs[layer.id] = vals.map(function (v, i) {
          return lerp(prev[i] || 0, v, reduce ? 1 : 0.35);
        });
      });

      heads.forEach(function (layer, hi) {
        var cx = w * (hi === 0 ? 0.35 : 0.65);
        var y0 = h * 0.62;
        var vals = acts[layer.id] || [];
        var prev = prevActs[layer.id] || vals;
        var n = layer.display || vals.length || 16;
        ctx.fillStyle = COLORS.dim;
        ctx.font = "11px \"IBM Plex Mono\", monospace";
        ctx.textAlign = "center";
        ctx.fillText(layer.label, cx, y0 - 12);
        var cell = 10;
        var x0 = cx - (n * (cell + 2)) / 2;
        for (var i = 0; i < n; i++) {
          var v = lerp(prev[i] || 0, vals[i] || 0, reduce ? 1 : 0.35);
          ctx.fillStyle = "rgba(184,212,168," + (0.2 + 0.8 * v) + ")";
          ctx.fillRect(x0 + i * (cell + 2), y0, cell, 28);
        }
        prevActs[layer.id] = vals.map(function (v, i) {
          return lerp(prev[i] || 0, v, reduce ? 1 : 0.35);
        });
      });

      ctx.strokeStyle = COLORS.line;
      ctx.beginPath();
      ctx.moveTo(w * 0.15, h * 0.52);
      ctx.lineTo(w * 0.5, h * 0.58);
      ctx.lineTo(w * 0.85, h * 0.52);
      ctx.stroke();
    }

    function applyFrame(fr) {
      frame = fr;
      setInputs(fr.inputs);
      setBars(fr);
    }

    function loop() {
      drawNet();
      raf = requestAnimationFrame(loop);
    }

    sizeCanvas();
    window.addEventListener("resize", function () {
      sizeCanvas();
      drawNet();
    });
    if (!reduce) raf = requestAnimationFrame(loop);

    if (opts.source === "ws") {
      var proto = location.protocol === "https:" ? "wss" : "ws";
      var ws = new WebSocket(proto + "://" + location.host + "/ws");
      ws.onmessage = function (ev) {
        var msg = JSON.parse(ev.data);
        if (msg.type === "schema") initSchema(msg);
        else if (msg.type === "frame") applyFrame(msg);
      };
      var status = $("status");
      ws.onopen = function () { if (status) status.classList.add("live"); };
      ws.onclose = function () { if (status) status.classList.remove("live"); };
      return;
    }

    var url = opts.url || "data/parkour_replay.json";
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        initSchema(data.schema);
        var frames = data.frames || [];
        var i = 0;
        function step() {
          if (!frames.length) return;
          applyFrame(frames[i]);
          i = (i + 1) % frames.length;
        }
        step();
        var dt = opts.dt || 150;
        if (!reduce) setInterval(step, dt);
      })
      .catch(function () {
        var synth = synthesizeParkour();
        initSchema(synth.schema);
        var frames = synth.frames;
        var i = 0;
        setInterval(function () {
          applyFrame(frames[i]);
          i = (i + 1) % frames.length;
        }, 150);
      });
  }

  function synthesizeParkour() {
    var actions = [
      "move_forward", "move_backward", "strafe_left", "strafe_right",
      "sprint_forward", "jump", "sprint_jump", "jump_forward",
      "sprint_jump_left", "sprint_jump_right", "look_down", "look_up",
      "turn_left", "turn_right", "no_op"
    ];
    var layers = [
      { id: "proprio", label: "Proprio", group: "stream", size: 64, display: 16 },
      { id: "goal", label: "Goal Δ", group: "stream", size: 64, display: 16 },
      { id: "voxel", label: "Voxel", group: "stream", size: 128, display: 16 },
      { id: "actor_hidden", label: "Actor hidden", group: "head", size: 64, display: 16 },
      { id: "critic_hidden", label: "Critic hidden", group: "head", size: 64, display: 16 }
    ];
    var schema = {
      type: "schema",
      env: "simple_jump",
      actions: actions,
      layers: layers,
      input_fields: [
        { key: "on_ground", label: "On ground" },
        { key: "vel", label: "Velocity (Δy,Δx,Δz)", dims: 3 },
        { key: "goal", label: "Goal Δ (dx,dy,dz)", dims: 3 }
      ]
    };
    var frames = [];
    var seq = [4, 4, 6, 6, 6, 0, 0, 0];
    for (var s = 0; s < 80; s++) {
      var act = seq[s % seq.length];
      var probs = actions.map(function (_, i) {
        return i === act ? 0.62 : 0.38 / (actions.length - 1);
      });
      var activations = {};
      var energy = {};
      layers.forEach(function (L) {
        var arr = [];
        for (var i = 0; i < L.display; i++) {
          arr.push(Math.max(0, 0.35 + 0.45 * Math.sin((s + i * 1.7) * 0.22 + L.size * 0.01)));
        }
        activations[L.id] = arr;
        energy[L.id] = arr.reduce(function (a, b) { return a + b; }, 0) / arr.length;
      });
      frames.push({
        type: "frame",
        step: s + 1,
        action: act,
        action_name: actions[act],
        probs: probs,
        value: 2.1 + 0.4 * Math.sin(s * 0.15),
        value_norm: 0.45 + 0.2 * Math.sin(s * 0.15),
        value_min: 0.2,
        value_max: 4.0,
        activations: activations,
        stream_energy: energy,
        inputs: {
          on_ground: s % 8 < 5 ? 1 : 0,
          vel: [s % 8 < 5 ? 0 : 0.4, 0.02, -0.3],
          goal: [0, s % 8 < 6 ? -1 : 0, -2.4 + s * 0.03]
        }
      });
    }
    return { schema: schema, frames: frames };
  }

  global.NetViz = { mount: mount, synthesizeParkour: synthesizeParkour };
})(window);
