# 🐳 RAG Backend – Docker Management Guide

This guide contains all useful Docker commands for building, running, managing, and debugging the **RAG Backend** project.

---

## 🧠 Services Overview

| Service Name | Container Name | Description | Ports |
|---------------|----------------|--------------|--------|
| `ai-server-rag` | `ai_server_rag_container` | Python AI Server (RAG logic, Google TTS, Coqui, etc.) | `11096 → 8250` |
| `node-backend-rag` | `node_backend_rag_container` | Node.js Backend API | `11095 → 8251` |
| `qdrant-rag` | `qdrant_rag_container` | Qdrant vector database | `11097 → 6333` |

---

## ⚙️ Basic Build and Run Commands

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python3 -m venv venv
source venv/bin/activate


### 🔧 Build a specific service
```bash
sudo docker compose build ai-server-rag
sudo docker compose build node-backend-rag
sudo docker compose build qdrant-rag

start the full code

docker compose up 
sudo docker compose build