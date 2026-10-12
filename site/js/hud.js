(function () {
  "use strict";

  var canvas = document.getElementById("voxel");
  var f3 = document.getElementById("f3");
  if (!canvas || !f3) return;

  var ctx = canvas.getContext("2d");
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var t = 0;

  var W = 5, D = 6, H = 5;
  // Parkour-ish occupancy: stone platforms with a 2-block gap at z=3,4
  function occ(x, y, z) {
    if (y !== 1) return 0;
    if (z <= 2 || z >= 5) return 1;
    return 0;
  }

  function resize() {
    var r = canvas.parentElement.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(r.width * dpr);
    canvas.height = Math.floor(r.height * dpr);
    canvas.style.width = r.width + "px";
    canvas.style.height = r.height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function iso(x, y, z, cx, cy, s) {
    return {
      x: cx + (x - z) * s,
      y: cy + (x + z) * s * 0.5 - y * s * 0.85
    };
  }

  function drawBlock(p, s, top, left, right) {
    var hx = s, hy = s * 0.5, hz = s * 0.85;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - hz);
    ctx.lineTo(p.x + hx, p.y - hy);
    ctx.lineTo(p.x, p.y);
    ctx.lineTo(p.x - hx, p.y - hy);
    ctx.closePath();
    ctx.fillStyle = top;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(p.x - hx, p.y - hy);
    ctx.lineTo(p.x, p.y);
    ctx.lineTo(p.x, p.y + hz);
    ctx.lineTo(p.x - hx, p.y + hz - hy);
    ctx.closePath();
    ctx.fillStyle = left;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(p.x + hx, p.y - hy);
    ctx.lineTo(p.x, p.y);
    ctx.lineTo(p.x, p.y + hz);
    ctx.lineTo(p.x + hx, p.y + hz - hy);
    ctx.closePath();
    ctx.fillStyle = right;
    ctx.fill();
  }

  function draw() {
    var w = canvas.clientWidth;
    var h = canvas.clientHeight;
    ctx.fillStyle = "#12151a";
    ctx.fillRect(0, 0, w, h);

    var s = Math.min(w, h) * 0.055;
    var cx = w * 0.62;
    var cy = h * 0.42;
    var agentZ = 2.2 + Math.sin(t * 0.018) * 0.15;
    var jumping = Math.max(0, Math.sin(t * 0.018));
    var agentY = 2.15 + jumping * 1.4;
    var agentX = 2;

    for (var y = 0; y < H; y++) {
      for (var z = D - 1; z >= 0; z--) {
        for (var x = 0; x < W; x++) {
          if (!occ(x, y, z)) continue;
          var p = iso(x, y, z, cx, cy, s);
          drawBlock(p, s, "#8a8680", "#5a5854", "#6e6b66");
        }
      }
    }

    var ap = iso(agentX, agentY, agentZ, cx, cy, s);
    drawBlock(ap, s * 0.72, "#c4a35a", "#3f6b3a", "#4a7a44");

    // faint gap ticks
    ctx.strokeStyle = "rgba(184,212,168,0.12)";
    ctx.setLineDash([3, 5]);
    var a = iso(2, 1, 3, cx, cy, s);
    var b = iso(2, 1, 4, cx, cy, s);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  var envs = [
    { env: "simple_jump", dim: 159, act: "sprint_jump", xyz: "0.50 / 46.00 / 3.50" },
    { env: "simple_jump", dim: 159, act: "sprint_forward", xyz: "0.50 / 46.12 / 4.10" },
    { env: "simple_jump", dim: 159, act: "move_forward", xyz: "0.50 / 45.00 / 6.40" }
  ];

  function hud() {
    var e = envs[Math.floor(t / 90) % envs.length];
    var step = (Math.floor(t / 6) % 30);
    f3.innerHTML =
      "MalmoRL debug<br>" +
      "<span class=\"dim\">env</span> " + e.env + "  <span class=\"dim\">obs</span> " + e.dim + "-d<br>" +
      "<span class=\"dim\">xyz</span> " + e.xyz + "<br>" +
      "<span class=\"dim\">action</span> <span class=\"hot\">" + e.act + "</span>  <span class=\"dim\">step</span> " + step + "/30<br>" +
      "<span class=\"dim\">registry</span> ENV[simple_jump] × ALGO[ppo]<br>" +
      "<span class=\"dim\">wire</span> Minecraft :10000 → env_server :10002 → train";
  }

  function tick() {
    t += 1;
    draw();
    if (t % 6 === 0) hud();
    if (!reduce) requestAnimationFrame(tick);
  }

  window.addEventListener("resize", function () {
    resize();
    draw();
  });

  resize();
  hud();
  draw();
  if (!reduce) requestAnimationFrame(tick);
})();
