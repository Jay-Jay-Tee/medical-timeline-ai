# MediTrack-AI - AI Powered Medical Timeline System
### Deployed at: https://medical-timeline-ai.onrender.com/
---
## Revolutionary Healthcare Record Management

### What Makes This Game-Changing

1. **Voice-to-Medical-Note** - Real-time speech-to-text with Web Speech API
2. **Smart Document Upload** - Upload lab reports and medical documents with preview
3. **Document Download** - Download any uploaded document anytime from timeline
4. **Professional PDF Export** - Doctor-ready medical reports in seconds
5. **Beautiful Dark Mode** - Eye-friendly interface for all lighting conditions
5. **Shareable Patient Links** - Secure, read-only profile sharing for healthcare providers
6. **Secure Authentication** - Patient accounts with hospital associations
7. **AI-Powered Analysis** - Timeline analysis with Groq Llama 3.3 70B
8. **Data Quality Metrics** - Intelligent assessment of medical record completeness
9. **Multi-Hospital Support** - Perfect for coordinating care across facilities
10. **Modern UI/UX** - Professional and polished

---

## Quick Start

### Prerequisites

- **Python 3.8-3.13**
- **Qdrant Vector Database** (cloud or local)
- **Groq API Key** (free tier available)

### Installation

```bash
# 1. Clone repository
git clone <your-repo>
cd meditrack-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with your credentials
cat > .env << EOF
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
EOF

# 5. Run the application
python app.py
```

### First Time Setup

1. Open `http://localhost:5000` in your browser
2. Click **"Sign In"** → **"Create Account"**
3. Fill in:
   - Your full name
   - Hospital name (e.g., "City General Hospital")
   - Email address
   - Secure password
4. You'll receive a **unique Patient ID** (e.g., `MED-ABC12345`)
5. Start adding medical events!

---

### Backend Stack

- **Flask** - Python web framework (lightweight, production-ready)
- **Qdrant** - Vector database for semantic search
  - 384-dimensional embeddings
  - Cosine similarity for semantic matching
  - Scales to millions of records
- **FastEmbed** - Fast text embeddings (no GPU required)
- **Groq Llama 3.3 70B** - AI analysis (fastest inference available)
- **ReportLab** - Professional PDF generation
- **bcrypt** - Secure password hashing
- **Flask-Login** - Session management

### Frontend Stack

- **Vanilla JavaScript** - No framework bloat, maximum performance
- **Tailwind CSS** - Modern utility-first styling
- **Web Speech API** - Built-in browser voice recognition
- **Drag & Drop API** - Native file upload handling

### Why This Stack?

**Fast** - No heavy frameworks slowing things down  
**Scalable** - Vector DB handles massive datasets  
**Accurate** - Semantic search > keyword matching  
**Modern** - Latest AI models and web technologies  
**Secure** - Industry-standard auth and encryption  
**Cost-Effective** - Free tiers available for all services  

---

## Project Structure

```
medical-timeline-ai/
├── app.py                    # Main Flask backend
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this!)
├── static/
│   ├── index.html           # Main application UI
│   ├── patient_view.html    # Public patient timeline view
│   └── app.js               # Frontend JavaScript logic
├── README.md                # This file
└── TROUBLESHOOTING.md       # Debug guide
```

---

## Key Features Explained

### 1. Voice-to-Medical-Note

**How it works:**
- Uses browser's **Web Speech API** (Chrome/Edge recommended)
- Real-time transcription as you speak
- No backend processing needed - runs entirely in browser
- Automatically detects pauses and punctuation
- Perfect for hands-free data entry during patient consultations

**Browser Support:**
- Chrome/Edge: Full support
- Safari: Partial support
- Firefox: Not supported yet

**Usage Tips:**
- Speak clearly and at normal pace
- Browser will ask for microphone permission (allow it)
- Red pulsing button indicates active recording
- Click again to stop recording

### 2. Multi-Hospital Patient IDs

**Problem Solved:**
Traditional systems use hospital-specific IDs, making it impossible to track patients across facilities.

**Our Solution:**
- Every user gets a **globally unique ID** (format: `MED-XXXXXXXX`)
- Hospital name stored with each record
- Patients can visit multiple hospitals using same ID
- Easy coordination of care between facilities

**Example Workflow:**
```
Patient registers: MED-A1B2C3D4 at "City Hospital"
Visits emergency room at "County Hospital" → Uses same ID
Primary care at "Memorial Clinic" → Same ID again
All records unified in one timeline!
```

### 3. Semantic Timeline Analysis

**What is Semantic Search?**
Instead of keyword matching, we use AI embeddings to understand *meaning*.

**Example:**
```
Traditional Search: "heart attack" only finds exact phrase
Semantic Search: Also finds "myocardial infarction", 
                  "cardiac arrest", "MI", etc.
```

### 4. AI-Powered Insights

**What the AI Does:**
- Analyzes complete medical timeline
- Identifies patterns and trends
- Describes frequency of visits
- Notes temporal gaps or clusters
- Provides neutral, factual summary

**What the AI Does NOT Do:**
- Diagnose conditions
- Suggest treatments
- Make medical recommendations
- Replace healthcare professionals

**Powered by Groq:**
- Llama 3.3 70B model
- Fastest AI inference available (~500 tokens/second)
- Free tier: 14,400 requests/day
- No GPU required on your machine

### 5. Professional PDF Export

**What's Included:**
- Hospital letterhead-style header
- Patient ID and generation timestamp
- Complete timeline table with:
  - Dates formatted professionally
  - Event types color-coded
  - Full content of each event
- Print-optimized layout
- HIPAA-style formatting

**Perfect For:**
- Specialist referrals
- Insurance claims
- Medical record requests
- Patient handouts
- Legal documentation

### 6. Shareable Patient Links

**How It Works:**
- Every patient gets unique URL: `/patient/MED-XXXXXXXX`
- Read-only view - cannot edit or add records
- No login required to view
- Perfect for sharing with:
  - Doctors
  - Specialists
  - Family members
  - Insurance companies

**Security:**
- URL is the only "password" (security through obscurity)
- No sensitive data visible without the link
- Can be revoked by changing Patient ID (future feature)

---

## Security Features

### Password Security
- **bcrypt hashing** with automatic salt generation
- Passwords never stored in plain text
- Industry-standard algorithm (used by GitHub, Google, etc.)

### Session Management
- **Flask-Login** for secure sessions
- HTTPOnly cookies prevent XSS attacks
- CSRF protection enabled
- Automatic session expiration

### Data Protection
- All API calls require authentication
- Patient data isolated by Patient ID
- No SQL injection risk (using NoSQL vector database)
- Input sanitization on all fields

### HTTPS Ready
Works seamlessly with production WSGI servers (Gunicorn + nginx) for HTTPS encryption.

---

## API Endpoints

### Authentication
```
POST /register       # Create new account
POST /login          # Sign in
POST /logout         # Sign out
GET  /me             # Get current user info
```

### Medical Records
```
POST /ingest              # Add text-based medical event
POST /upload-document     # Upload document with notes
POST /timeline-summary    # Get complete timeline + AI analysis
POST /export-pdf          # Generate PDF report
```

### Public Access
```
GET /patient/<id>    # Public read-only timeline view
GET /health          # Health check
GET /api/status      # Detailed system status
```

---

## UI/UX Design Philosophy

### Accessibility

- **High contrast** - Text readable in all lighting
- **Keyboard navigation** - All features accessible via keyboard
- **Screen reader friendly** - Semantic HTML
- **Dark mode** - Reduces eye strain for night shifts

### Responsive Design

- **Mobile** - Full functionality on phones
- **Tablet** - Optimized for iPad/Android tablets
- **Desktop** - Takes advantage of large screens
- **Print** - Optimized for printing timelines

---

## Deployment

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:8000 --workers 4 --timeout 120
```

### Production with nginx (Recommended)
```nginx
server {
    listen 80;
    server_name medtimeline.example.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Environment Variables for Production
```bash
# Required
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your_production_key
GROQ_API_KEY=your_production_key
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Optional
FLASK_ENV=production
WORKERS=4
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4"]
```

---

## Testing

### Manual Testing Checklist

#### Registration & Login
- Can create new account
- Receives unique Patient ID
- Can log in with email/password
- Can log out
- Session persists across page refresh

#### Voice Input
- Microphone permission requested
- Real-time transcription works
- Can stop recording
- Text saves correctly

#### Document Upload
- Drag-and-drop works
- File preview displays
- Can add notes
- Document saves to timeline

#### Timeline Analysis
- Shows all events in table
- AI summary generates
- Data quality assessed correctly
- Semantic shift calculated

#### PDF Export
- PDF downloads successfully
- All events included
- Professional formatting
- Printable

#### Sharing
- Share link copies to clipboard
- Public view loads correctly
- Cannot edit from public view

---

## Future Enhancements

### Phase 2 (Next Sprint)
- Mobile app (React Native)
- Medication interaction checker
- Appointment scheduling
- Lab result visualization graphs
- Multi-language support (100+ languages)

### Phase 3 (Long-term)
- Wearable device integration (Fitbit, Apple Watch)
- Telemedicine video calls
- Family health history linking
- Insurance claim automation
- Blockchain for immutable records
- Machine learning for health predictions

---

## Credits & Acknowledgments

**Built with:**
- [Qdrant](https://qdrant.tech/) - Vector Database
- [Groq](https://groq.com/) - AI Inference
- [FastEmbed](https://github.com/qdrant/fastembed) - Text Embeddings
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [ReportLab](https://www.reportlab.com/) - PDF Generation

**Inspiration:**
This project addresses real pain points in healthcare record management, inspired by the challenges patients face coordinating care across multiple providers and institutions.

---

## Support & Contact

**Found a bug?**
- Check TROUBLESHOOTING.md first
- Open an issue on GitHub
- Include error messages and steps to reproduce

**Questions?**
- Read the inline code comments
- Check the API documentation above
- Review the troubleshooting guide

**Want to contribute?**
- Fork the repository
- Create a feature branch
- Submit a pull request
- We welcome contributions!

---

## Learning Resources

**New to Vector Databases?**
- [Qdrant Tutorial](https://qdrant.tech/documentation/tutorials/)
- [Understanding Embeddings](https://platform.openai.com/docs/guides/embeddings)

**Want to Learn More About AI?**
- [Groq Documentation](https://console.groq.com/docs)
- [Llama Model Details](https://www.llama.com/)

**Flask Development:**
- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [Flask-Login Docs](https://flask-login.readthedocs.io/)

---

## Scope & Medical Disclaimer

Medical Timeline AI is an **information retrieval and summarization system**, not a diagnostic or decision-making tool.  
All AI-generated outputs are derived strictly from stored patient events and are intended to **assist human understanding**, not replace professional medical judgment.

This system:
- Does **not** diagnose conditions
- Does **not** recommend treatments
- Does **not** perform causal or clinical inference

Clinicians and users must independently verify all information.

---

## Current Limitations

The following features are **intentionally out of scope** for the current implementation and represent future extensions:

- Explicit causal inference between events (e.g., medication → lab outcome)
- Formal citation tagging for individual AI-generated statements
- Role-based access control (RBAC) and audit logging
- Image/audio semantic embeddings (current support is text and documents)
- Regulatory compliance enforcement (HIPAA/GDPR layers)

These limitations do **not** affect core timeline storage, semantic search, or reporting functionality.

---

## License

MIT License - Free to use, modify, and distribute!

---

**Made with ❤️ for better healthcare**
