#!/bin/bash
# =============================================================================
# MineRL Docker Setup Script
# Run this ONCE before building the Docker image.
# It downloads the MixinGradle jar that is no longer available on public repos.
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR_NAME="MixinGradle-dcfaf61.jar"
JAR_PATH="$SCRIPT_DIR/$JAR_NAME"

echo -e "${GREEN}=== MineRL Docker Setup ===${NC}"
echo ""

# ── Step 1: Check Docker is installed ────────────────────────────────────────
echo -e "${YELLOW}[1/4] Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop/${NC}"
    exit 1
fi
echo "Docker found: $(docker --version)"

# ── Step 2: Check for MixinGradle jar ────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/4] Checking for MixinGradle jar...${NC}"

if [ -f "$JAR_PATH" ]; then
    echo -e "${GREEN}MixinGradle jar already present: $JAR_PATH${NC}"
else
    echo "MixinGradle-dcfaf61.jar not found."
    echo ""
    echo "This jar is required to build MineRL but is no longer on any public Maven repo."
    echo ""
    echo "Please download it manually:"
    echo ""
    echo "  https://drive.google.com/file/d/1z9i21_GQrewE0zIgrpHY5kKMZ5IzDt6U/view"
    echo ""
    echo "Save it as:  $JAR_PATH"
    echo ""
    read -p "Press ENTER once you have downloaded the jar, or Ctrl+C to abort..."

    if [ ! -f "$JAR_PATH" ]; then
        echo -e "${RED}Jar not found at $JAR_PATH. Aborting.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Jar found.${NC}"

# ── Step 3: Create workspace directory ───────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/4] Creating workspace directory...${NC}"
mkdir -p "$SCRIPT_DIR/workspace"
echo "Workspace ready at: $SCRIPT_DIR/workspace"

# ── Step 4: Detect platform and build ────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/4] Detecting platform...${NC}"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "OS: $OS | Arch: $ARCH"
echo ""

if [[ "$OS" == "Darwin" ]]; then
    echo -e "${GREEN}macOS detected → using CPU profile (linux/amd64 via Rosetta)${NC}"
    echo ""
    echo "Building image (this will take 15-30 mins the first time)..."
    docker compose --profile cpu build
    echo ""
    echo -e "${GREEN}Build complete!${NC}"
    echo ""
    echo "To start:"
    echo "  docker compose --profile cpu up"
    echo ""
    echo "Then open:  http://localhost:8888"
    echo "VNC viewer: localhost:5900 (optional, to see Minecraft render)"

elif [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "CYGWIN"* ]] || [[ "$OS" == "MSYS"* ]]; then
    echo -e "${GREEN}Windows detected → using GPU profile (requires NVIDIA Docker runtime)${NC}"
    echo ""
    echo "Checking for NVIDIA runtime..."
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo "NVIDIA runtime found."
    else
        echo -e "${YELLOW}Warning: NVIDIA runtime not detected. Install from:${NC}"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
        echo "Continuing build anyway..."
    fi
    echo ""
    echo "Building image (this will take 20-40 mins the first time)..."
    docker compose --profile gpu build
    echo ""
    echo -e "${GREEN}Build complete!${NC}"
    echo ""
    echo "To start:"
    echo "  docker compose --profile gpu up"
    echo ""
    echo "Then open:  http://localhost:8888"
    echo "VNC viewer: localhost:5900 (optional)"

else
    echo "Platform not auto-detected. Choose manually:"
    echo ""
    echo "  CPU (Mac):     docker compose --profile cpu build && docker compose --profile cpu up"
    echo "  GPU (Windows): docker compose --profile gpu build && docker compose --profile gpu up"
fi

echo ""
echo -e "${GREEN}Setup complete!${NC}"
