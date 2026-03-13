#!/bin/bash
set -e

# =============================================================================
# MineRL Container Startup Script
# Starts: Xvfb virtual display, x11vnc, noVNC web viewer, Jupyter Lab
# =============================================================================

# ── 1. Virtual display (Xvfb) ─────────────────────────────────────────────────
echo "[1/4] Starting virtual display on :99..."
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99
sleep 1
echo "      Virtual display ready."

# ── 2. VNC server (x11vnc) ────────────────────────────────────────────────────
echo "[2/4] Starting VNC server on port 5900..."
x11vnc -display :99 -nopw -listen 0.0.0.0 -xkb -forever -shared -bg -o /var/log/x11vnc.log
echo "      VNC ready at localhost:5900"

# ── 3. noVNC web viewer ───────────────────────────────────────────────────────
echo "[3/4] Starting noVNC web viewer on port 6080..."
websockify --web /usr/share/novnc/ 6080 localhost:5900 &
echo "      noVNC ready at http://localhost:6080/vnc.html"

# ── 4. Jupyter Lab ────────────────────────────────────────────────────────────
echo "[4/4] Starting Jupyter Lab on port 8888..."
echo ""
echo "================================================"
echo "  Jupyter:  http://localhost:8888"
echo "  noVNC:    http://localhost:6080/vnc.html"
echo "  VNC:      localhost:5900 (no password)"
echo "================================================"
echo ""

exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token="" \
    --NotebookApp.password="" \
    --notebook-dir=/workspace