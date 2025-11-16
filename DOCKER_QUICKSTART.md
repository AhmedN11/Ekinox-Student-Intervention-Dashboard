# Docker Quick Start Guide

## For Users with Ollama (Local LLM)

### First Time Setup
```bash
# 1. (Optional) Copy environment template to pre-configure LLM settings
cp .env.example .env
# Note: DOCKER_ENV is automatically set to true by docker-compose
# Note: LLM configuration can be done from the dashboard homepage
# You can optionally pre-configure:
#    OLLAMA_MODEL_NAME=ollama/llama2
#    OLLAMA_BASE_URL=http://ollama:11434

# 2. Start all services
docker-compose up -d --build

# 3. Wait for services to start (30 seconds)
sleep 30

# 4. Pull an Ollama model (choose one)
docker exec -it ollama-service ollama pull llama2
# OR
docker exec -it ollama-service ollama pull mistral
# OR
docker exec -it ollama-service ollama pull codellama

# 5. Open browser and configure LLM from homepage
xdg-open http://localhost:8501  # Linux
# or
open http://localhost:8501      # macOS
```

### Daily Use
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f
```

---

## For Users with Cloud LLMs (Mistral, OpenAI, etc.)

### First Time Setup
```bash
# 1. (Optional) Copy environment template to pre-configure LLM settings
cp .env.example .env
# Note: DOCKER_ENV is automatically set to true by docker-compose
# Note: LLM configuration (API keys, models) can be done from the dashboard homepage
# You can optionally pre-configure:
#    MISTRAL_API_KEY=your_key_here
#    MISTRAL_MODEL_NAME=mistral/codestral-2508
#    # OR for OpenAI:
#    OPENAI_API_KEY=your_key_here
#    OPENAI_MODEL_NAME=openai/gpt-4

# 2. Start the application (no Ollama)
docker-compose -f docker-compose.cloud.yml up -d --build

# 3. Open browser and configure LLM from homepage
xdg-open http://localhost:8501  # Linux
# or
open http://localhost:8501      # macOS
```

### Daily Use
```bash
# Start
docker-compose -f docker-compose.cloud.yml up -d

# Stop
docker-compose -f docker-compose.cloud.yml down

# View logs
docker-compose -f docker-compose.cloud.yml logs -f
```

---

## Common Commands

### Viewing Logs
```bash
# All logs
docker-compose logs -f

# Application only
docker-compose logs -f streamlit-app

# Ollama only
docker-compose logs -f ollama
```

### Managing Ollama Models
```bash
# List installed models
docker exec -it ollama-service ollama list

# Pull a new model
docker exec -it ollama-service ollama pull <model-name>

# Remove a model
docker exec -it ollama-service ollama rm <model-name>
```

### Troubleshooting
```bash
# Restart everything
docker-compose restart

# Rebuild from scratch
docker-compose down -v
docker-compose up --build

# Check if containers are running
docker ps

# Access container shell
docker exec -it student-intervention-dashboard bash
docker exec -it ollama-service bash
```

### Updating the Application
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

---

## Port Reference

- **8501** - Streamlit Dashboard
- **11434** - Ollama API (if using Ollama)

## Volume Reference

Data is persisted in:
- `./data/` - Your CSV files
- `./Logs/` - Application logs
- Docker volume `ollama-data` - Ollama models (if using Ollama)

To completely remove everything including volumes:
```bash
docker-compose down -v
```
