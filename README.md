# 🚀 Advanced RAG & SQL Agent System

Welcome to the **RAG-Qdrant** project. This is an advanced, enterprise-grade AI system that combines **Retrieval-Augmented Generation (RAG)** with a powerful **Text-to-SQL Agent**. It features a modern React frontend, a dual-backend architecture (Node.js & Python), and integration with Qdrant for vector operations and MySQL for structured data management.

This project is designed to be a comprehensive solution for businesses needing to chat with their documents (RAG) and interact naturally with their SQL databases (SQL Agent).

## 🌟 Key Features

*   **Dual-Backend Architecture**: 
    *   **Node.js (Express)**: Handles user authentication, file management, and API orchestration.
    *   **Python (Flask)**: Powering the AI core, managing RAG logic, Vector DB interactions, and SQL generation.
*   **Advanced RAG**: Chat with your documents using state-of-the-art vector search (Qdrant) and LLMs.
*   **SQL Agent**: Natural language to SQL generation and execution on your custom databases.
*   **Voice Interface**: Full voice-to-voice interaction capabilities (STT/TTS).
*   **Role-Based Access Control (RBAC)**: Secure access for Super Admins, Admins, and Business Users.
*   **Modern Frontend**: Built with React, featuring a sleek, responsive design with dark mode aesthetics.

---

## 🛠️ Technology Stack

*   **Frontend**: React (Create React App), Lucide React, Axios.
*   **Backend A (API & Auth)**: Node.js, Express, MySQL2.
*   **Backend B (AI Core)**: Python 3.x, Flask, LangChain (implied), Qdrant Client.
*   **Database**: MySQL (master data), Qdrant (Vector DB).
*   **AI Services**: Integration with OpenAI, Groq, Gemini, Deepgram, ElevenLabs, etc.

---

## 🚀 Getting Started

### 1. Database Setup (MySQL)

You need a MySQL instance running. Run the following SQL commands to set up the required schema:

```sql
CREATE DATABASE IF NOT EXISTS `agent_db`;
USE `agent_db`;

-- 1. Firm Master Table
CREATE TABLE IF NOT EXISTS `firm_master` (
  `FIRM_ID` int unsigned NOT NULL AUTO_INCREMENT,
  `FIRM_NAME` varchar(200) CHARACTER SET latin1 COLLATE latin1_swedish_ci NOT NULL,
  `FIRM_CODE` varchar(50) CHARACTER SET latin1 COLLATE latin1_swedish_ci NOT NULL,
  `CONTACT_EMAIL` varchar(200) CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT NULL,
  `CONTACT_PHONE` varchar(20) CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT NULL,
  `ADDRESS` text CHARACTER SET latin1 COLLATE latin1_swedish_ci,
  `CREATED_BY` int unsigned NOT NULL COMMENT 'SUPER_ADMIN USER_ID',
  `CREATED_DATE` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `STATUS` enum('Active','Inactive') CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT 'Active',
  PRIMARY KEY (`FIRM_ID`),
  UNIQUE KEY `uk_firm_code` (`FIRM_CODE`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=latin1;

-- 2. LLM Details Table
CREATE TABLE IF NOT EXISTS `llm_details` (
  `ID` int NOT NULL AUTO_INCREMENT,
  `USERID` int NOT NULL,
  `FIRMID` int NOT NULL,
  `LLM_PROVIDER` enum('GROQ','OPENROUTER','HUGGINGFACE','DEEPSEEK','MISTRAL','GPT2','LLAMA3.2','HUGGINGFACE-TEXTTOIMAGE','GEMINI-2.0-FLASH-EXP-TEXTTOIMAGE','TOGETHER','OPENAI_DALLE3','MIDJOURNEY','ADOBE_FIREFLY','STABILITY_AI','LEONARDO_AI','FLUX_AI','IDEOGRAM','STABLE_DIFFUSION','DREAMSTUDIO','RUNWAY_ML','TOGETHER_STABLE_DIFFUSION','KANDINSKY','OPENAI_GPT4V','GEMINI_PRO_VISION','CLAUDE_3_OPUS','LLAVA_13B','MINIGPT4','INSTRUCTBLIP','PHI3_VISION','PLAYHT','CLAUDE','OPENAI_GPT4','GEMINI','OLLAMA','COHERE','TOGETHER_AI','REPLICATE','SUNO_AI','UDIO','MUSICGEN_META','AIVA','MUBERT','SOUNDRAW','AMAZON_POLLY','GOOGLE_TTS','IBM_WATSON','MURF_AI','WELLSAID','RUNWAY_GEN4','OPENAI_SORA','GOOGLE_VEO2','PIKA_2_2','KLING_AI_2','HAILUO_AI','WAN2_2','ELEVENLABS','DEEPGRAM','TWILIO_VOICE','TELNYX_VOICE') DEFAULT NULL,
  `LLM_PROVIDER_TYPE` enum('TEXT-TO-TEXT','TEXT-TO-IMAGE','IMAGE-TO-TEXT','TEXT-TO-MUSIC','TEXT-TO-SPEECH','TEXT-TO-VIDEO','VOICE-CALL','SPEECH-TO-TEXT') DEFAULT 'TEXT-TO-TEXT',
  `API_KEY` varchar(255) NOT NULL,
  `STATUS` enum('ACTIVE','INACTIVE') NOT NULL DEFAULT 'ACTIVE',
  `INSRT_DTM` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `UPD_DTM` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB AUTO_INCREMENT=492 DEFAULT CHARSET=latin1;

-- 3. RAG Metrics Table
CREATE TABLE IF NOT EXISTS `rag_metrics` (
  `id` int NOT NULL AUTO_INCREMENT,
  `firm_id` int NOT NULL,
  `user_id` int DEFAULT NULL,
  `query_text` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `query_type` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `response_text` text COLLATE utf8mb4_unicode_ci,
  `latency_ms` int DEFAULT NULL,
  `success` tinyint(1) NOT NULL DEFAULT '0',
  `error_message` text COLLATE utf8mb4_unicode_ci,
  `context_docs_count` int DEFAULT NULL,
  `sql_executed` text COLLATE utf8mb4_unicode_ci,
  `timestamp` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_firm` (`firm_id`),
  KEY `idx_ts` (`timestamp`)
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. User Registration Table
CREATE TABLE IF NOT EXISTS `user_registration` (
  `USER_ID` int unsigned NOT NULL AUTO_INCREMENT,
  `USER_NAME` varchar(50) CHARACTER SET latin1 COLLATE latin1_swedish_ci NOT NULL,
  `USERNAME` varchar(100) CHARACTER SET latin1 COLLATE latin1_swedish_ci NOT NULL,
  `PASSWORD` varchar(255) CHARACTER SET latin1 COLLATE latin1_swedish_ci NOT NULL,
  `EMAIL` varchar(200) CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT NULL,
  `ROLE` enum('SUPER_ADMIN','ADMIN','BUSINESS_USER') NOT NULL DEFAULT 'BUSINESS_USER',
  `DESIGNATION` varchar(200) CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT NULL,
  `COMPANY` varchar(200) CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT NULL,
  `FIRM_ID` int DEFAULT NULL COMMENT 'Company identifier; NULL for Super Admin',
  `CREATED_DATE` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `STATUS` enum('Active','Inactive') CHARACTER SET latin1 COLLATE latin1_swedish_ci DEFAULT 'Active',
  PRIMARY KEY (`USER_ID`),
  UNIQUE KEY `uk_username` (`USERNAME`)
) ENGINE=InnoDB AUTO_INCREMENT=1496 DEFAULT CHARSET=latin1;
```

---

### 2. Backend Setup & Running

This project uses **two** backend servers. You need to run both.

#### A. Node.js Backend (Auth & API)

1.  Navigate into the `backend` folder:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the Node server:
    ```bash
    npm start
    ```
    *Runs on Port: `8251` (Proxying to Python on failure or handling DB logic)*

#### B. Python AI Server (Flask)

1.  Navigate into the `backend` folder (or ensure you are in the root):
    ```bash
    cd backend
    ```
2.  (Optional but Recommended) Create and activate a Virtual Environment:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```
3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the AI Server:
    ```bash
    python ai_server.py
    ```
    *Runs on Port: `8250`*

---

### 3. Frontend Setup & Running

1.  Navigate into the `frontend` folder:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the React application:
    ```bash
    npm start
    ```
    *Runs on Port: `8350` (or 3000 by default if updated)*

---

## 🔒 Configuration

Ensure you have a `.env` file in your `backend` directory with the necessary keys:

```env
# Example .env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=agent_db
QDRANT_URL=http://localhost:6333
# Add other keys (GROQ_API_KEY, OPENAI_API_KEY, etc.) as needed.
```

---

## 📄 License

This project is licensed under the ISC License.

---
*Created by Allen Joseph*
