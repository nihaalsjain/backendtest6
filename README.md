# Allion.ai – Backend Voice Agent

### Features
- **Agent core** → Configurable assistant with Text & Voice (STT & TTS)
- **Vision capabilities** → Multi-modal live assistance with text / voice / video / screen sharing
- **RAG capabilities** → Retrieval-Augmented Generation with PDFs (User Manuals / Repair Guides / TSBs)

### Utilities
This project runs a LiveKit-based voice agent in two main modes:

- **Console Mode** → Local testing in your terminal  
- **Dev/Web Mode** → Connect to the hosted frontend: [AllionAI Web App](http://allion-ui.vercel.app/)  

Additionally:  
- **Python Test Mode** → Run unit tests or small harnesses under `scripts/`.

---

## 📂 Project structure
```bash
AllionAI/
├── agents/
│   ├── __init__.py
│   ├── automotive_agent.py
│   └── multilingual_agent.py
├── configs/
│   ├── __init__.py
│   ├── english_config.py
│   ├── hindi_config.py
│   └── kannada_config.py
├── prompts/
│   ├── prompts.json
│   └── automotive/
│       ├── instructions.txt
│       └── greetings.json
├── tools/
│   ├── __init__.py
│   └── vision_capabilities.py
├── docs/pdf_source/ (to hold TSBs, Repair guides, Owner manuals)
│   └── uml/ (architecture diagrams)
├── requirements.txt
├── requirements_rag.txt
└── .env
```

### Getting started
---
## 1. Clone the Repository
### Cloning with HTTPS + Personal Access Token (PAT)

> ⚠️ Treat your Personal Access Token (PAT) like a password. Never commit or share it publicly.

#### Who Can Clone
- You must be explicitly given access to this repository.
- For a **personal repo**: the owner must invite you as a **Collaborator** (with at least **Read** permission).


#### Generate a Personal Access Token (PAT)
1. Go to: **GitHub → Settings → Developer settings → Personal access tokens**
2. Click **Generate new token** (Fine-grained is preferred; Classic works too).
3. Fill in:
   - **Name** (e.g., `clone-access`)
   - **Expiration** (shorter is safer)
   - **Scopes / Permissions**:
     - **Fine-grained**: grant access to *this repository* with **Read** permissions.
     - **Classic**: check the `repo` scope.
4. Click **Generate token** and **copy it** (you won’t be able to see it again).

#### Clone the Repo
```bash
git clone https://<USERNAME>:<TOKEN>@github.com/nihaalsjain/Allion.ai.git
```




## 2. Create a Virtual Environment and Activate It
**Windows**
```bash
python -m venv allion
allion\Scripts\activate
```
**macOS / Linux**
```bash
python -m venv allion
source allion/bin/activate
```

---

## 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_rag.txt
```

---

## 4. Setup Environment Variables
Create a `.env` file in the project root:

```env
# LiveKit Connection
LIVEKIT_URL=wss://<your_project_id>.livekit.cloud
LIVEKIT_API_KEY=<your_api_key>
LIVEKIT_API_SECRET=<your_api_secret>

# Agent Settings
LIVEKIT_ROOM=my-test-room
AGENT_ID=agent1

# Model Provider
OPENAI_API_KEY=<your_openai_or_openrouter_key>
```

---

## 5. Download Required Files
```bash
python -m agents.multilingual_agent download-files
```

---

## 6a. Run in Console Mode (Local Testing)
```bash
python -m agents.multilingual_agent en console
```
- Use your microphone to talk to the agent in the terminal.  
- Ideal for quick debugging and testing.

---

## 6b. Run in Dev Mode (Connect to Web App)
```bash
python -m agents.multilingual_agent dev
```
- Starts the LiveKit agent worker and joins the specified room.  
- Go to the [frontend web app](http://allion-ui.vercel.app/)
- Enter the same room name as in `.env` (`LIVEKIT_ROOM`) to interact with your agent via browser.

---

## 7. Troubleshooting

### Cloning in GCP (Google Cloud Platform) to have the backend outside Bosch network to make it accessible from the frontend.

Since the repo is private, a plain HTTPS clone won’t work in Google Cloud Shell .

Use a GitHub Personal Access Token (PAT)
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a token with: `repo` (full control of private repos)
3. connect to VM in the cloud shell
   -In GCP shall 
   ```bash
   gcloud compute ssh instance-20250923-060518 --zone asia-south1-b
   ```
   Password can be found in Allion Dev Team channels pinned message.

4. In VM Shell will open, clone with: 
   ```bash
   git clone https://<USERNAME>:<TOKEN>@github.com/nihaalsjain/Allion.ai.git
   ```
