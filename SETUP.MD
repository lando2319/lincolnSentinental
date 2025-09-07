# Setup
These are the setup for setting up a new machine

# Dev Setup for Mac

# 1) LLM runtime
```
brew install ollama
brew services start ollama
ollama pull gemma3:4b
```

# 2) Qdrant (vector DB)
```
brew install --cask docker
open -a Docker
mkdir -p ~/qdrant_storage
docker run -d --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

# 3) OCR tools (no canvas needed)
```
brew install poppler                    # gives: pdfinfo, pdftotext, pdftoppm
```

# Production Set for Jetson Orin Nano

## System packages
```
sudo apt update
sudo apt install -y curl git build-essential \
  poppler-utils                        # pdfinfo/pdftotext/pdftoppm for OCR
```

# Node.js (LTS) via nvm
```
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"
nvm install 20
nvm use 20
```

# Docker (for Qdrant)
```
sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl enable --now docker
```

# Qdrant container (ARM64 works)
```
mkdir -p ~/qdrant_storage
docker run -d --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

# Ollama (ARM64)
```
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull gemma3:4b
```



# BOTH DEV AND PRODUCTION

Clone Repo
npm i

cp .env .env.example

# Ingest docs
node ingest.mjs

# Run Server
node server.mjs


# Test
```
curl -s -X POST localhost:8010/ask \
  -H 'content-type: application/json' \
  -d '{"question":"How do I defog the windshield?"}' | jq .
```

# Health Checks
```
# Qdrant up?
curl -s http://localhost:6333/collections | jq .

# points for a specific file?
curl -s -X POST http://localhost:6333/collections/lincoln_docs/points/count \
  -H 'content-type: application/json' \
  -d '{"exact":true,"filter":{"must":[{"key":"filename","match":{"value":"67-OM32-33_large.pdf"}}]}}' | jq .

# server alive?
curl -s localhost:8010/healthz | jq .
```


## Optional for Producton: run the Node server as a service (auto-start)

Create `/etc/systemd/system/lincoln-rag.service`

```
[Unit]
Description=Lincoln RAG Server
After=network-online.target docker.service ollama.service
Wants=network-online.target

[Service]
User=%i
WorkingDirectory=/home/%i/lincolnSentinental
Environment=NODE_ENV=production
ExecStart=/usr/bin/env node server.mjs
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable (replace ubuntu if your user is different):

```
sudo systemctl enable lincoln-rag@ubuntu
sudo systemctl start lincoln-rag@ubuntu
```

