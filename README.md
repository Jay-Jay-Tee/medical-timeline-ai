# Meditrack AI
Semantic memory for medicine. Built to be useful, not pretty.

## Overview
Medical Timeline AI is a long-term semantic memory system for healthcare events. It ingests documents, doctor notes, and patient logs, then builds an intelligent, searchable timeline using vector embeddings and LLM summaries.

This isn’t a chatbot in a lab coat or another EHR trying to be Excel. It’s a memory architecture for clinicians, patients, and researchers who want clarity over chaos.

Live at: [https://medical-timeline-ai.onrender.com](https://medical-timeline-ai.onrender.com)

## Tech Stack
- Flask — Web framework
- Qdrant — Vector database for semantic search
- FastEmbed — Lightweight embedding generation
- Groq (LLaMA 3.3) — LLM for timeline analysis
- ReportLab — PDF generation
- Flask-Login — Session handling
- Render — Hosting platform

## Features
- Auth system with patient-linked memory
- Drag-and-drop document ingestion
- Automatic vector embedding + Qdrant upsert
- Timeline summarization using LLMs
- Semantic clustering of related events
- Downloadable PDF reports per patient
- Real-time API health monitoring
- Designed for clarity, not fluff

## Quick Start

```bash
git clone https://github.com/yourusername/medical-timeline-ai.git
cd medical-timeline-ai
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

Then create a .env file with:
```
SECRET_KEY=your_flask_secret_key
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_key
Run the server:
```
```
python app.py
```
## Team
Built by:
Siddharth Madhavan
Joshua Jacob Thomas

## License
MIT
