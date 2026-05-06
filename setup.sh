#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# VIBS — One-Command Setup Script
# Run: bash setup.sh
# ─────────────────────────────────────────────────────────────

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  VIBS — Voice Intelligence Setup             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Check Docker ─────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo -e "${RED}✗ Docker not found.${NC}"
  echo "  Install from: https://docs.docker.com/engine/install/"
  exit 1
fi
echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"

if ! docker info &>/dev/null; then
  echo -e "${RED}✗ Docker daemon not running. Start Docker and try again.${NC}"
  exit 1
fi

# ── Check docker-compose ─────────────────────────────────────
if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null 2>&1; then
  echo -e "${RED}✗ docker-compose not found.${NC}"
  exit 1
fi
COMPOSE="docker compose"
command -v docker-compose &>/dev/null && COMPOSE="docker-compose"
echo -e "${GREEN}✓ Docker Compose found${NC}"

# ── GPU detection ─────────────────────────────────────────────
GPU_AVAILABLE=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
  GPU_AVAILABLE=true
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo -e "${GREEN}✓ GPU detected: ${GPU_NAME}${NC}"
  echo "  Whisper and pyannote will use float16 on CUDA (fast)"
else
  echo -e "${YELLOW}⚠ No NVIDIA GPU detected — will use CPU (int8, slower)${NC}"
  echo "  Expected latency: 2–5s per 2s audio chunk"
fi

# ── Create .env if not exists ─────────────────────────────────
if [ ! -f backend/.env ]; then
  echo ""
  echo "Creating backend/.env from template..."
  cp backend/.env.example backend/.env
  echo -e "${YELLOW}⚠ ACTION REQUIRED: Edit backend/.env and add your tokens:${NC}"
  echo ""
  echo "  1. HF_TOKEN  → https://huggingface.co/settings/tokens"
  echo "     Then accept: https://huggingface.co/pyannote/speaker-diarization-3.1"
  echo ""
  echo "  2. GROQ_API_KEY (recommended, free) → https://console.groq.com"
  echo "     Or leave blank to skip LLM features (transcript still works)"
  echo ""
  read -p "Press Enter after editing backend/.env to continue..." _
else
  echo -e "${GREEN}✓ backend/.env already exists${NC}"
fi

# ── Strip GPU section if no GPU ───────────────────────────────
if [ "$GPU_AVAILABLE" = false ]; then
  echo ""
  echo "Patching docker-compose.yml to remove GPU requirements for CPU-only mode..."
  # Create a CPU-only override
  cat > docker-compose.override.yml << 'EOF'
version: '3.9'
services:
  backend:
    deploy: {}
  worker:
    deploy: {}
EOF
  echo -e "${GREEN}✓ CPU-only override created (docker-compose.override.yml)${NC}"
fi

# ── Create audio files directory ──────────────────────────────
mkdir -p audio_files
echo -e "${GREEN}✓ audio_files/ directory ready${NC}"

# ── Build and start ───────────────────────────────────────────
echo ""
echo "Building Docker images (first run takes 5–15 minutes, downloads ML models)..."
$COMPOSE build

echo ""
echo "Starting all services..."
$COMPOSE up -d

# ── Wait for healthy ──────────────────────────────────────────
echo ""
echo "Waiting for services to be ready..."
MAX_WAIT=60
COUNT=0
while [ $COUNT -lt $MAX_WAIT ]; do
  if $COMPOSE exec -T postgres pg_isready -U vibs_user -d vibs &>/dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL ready${NC}"
    break
  fi
  sleep 2
  COUNT=$((COUNT+2))
done

COUNT=0
while [ $COUNT -lt $MAX_WAIT ]; do
  if $COMPOSE exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo -e "${GREEN}✓ Redis ready${NC}"
    break
  fi
  sleep 2
  COUNT=$((COUNT+2))
done

sleep 3

# ── Health check ──────────────────────────────────────────────
HEALTH=$(curl -s http://localhost:8000/ 2>/dev/null || echo "")
if echo "$HEALTH" | grep -q '"name"'; then
  echo -e "${GREEN}✓ Backend API ready${NC}"
else
  echo -e "${YELLOW}⚠ Backend starting up, may take another 30s for model preload${NC}"
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  VIBS is running!                            ║"
echo "║                                              ║"
echo "║  Open:  http://localhost:5173                ║"
echo "║  API:   http://localhost:8000                ║"
echo "║  Docs:  http://localhost:8000/docs           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Commands:"
echo "  Stop:    docker-compose down"
echo "  Logs:    docker-compose logs -f backend"
echo "  Worker:  docker-compose logs -f worker"
echo ""
