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