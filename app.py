from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from fastembed import TextEmbedding
import numpy as np
import os
import logging
import bcrypt
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import atexit

from dotenv import load_dotenv
load_dotenv()

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
COLLECTION_NAME = "medical_events"
VECTOR_DIM = 384

# ==================== FLASK APP ====================
app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
CORS(app, supports_credentials=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ==================== GLOBAL STATE ====================
qdrant_client = None
embedding_model = None
groq_client = None
users_db = {}
initialization_status = {
    "qdrant": False,
    "embedding": False,
    "collection": False,
    "groq": False,
    "initialized": False
}

# ==================== INITIALIZATION (RUNS ONCE) ====================

def initialize_app():
    """Initialize all components - runs ONCE on startup"""
    global qdrant_client, embedding_model, groq_client, initialization_status
    
    if initialization_status["initialized"]:
        logger.info("⚠️  Already initialized, skipping...")
        return True
    
    logger.info("=" * 60)
    logger.info("🩺 MEDICAL TIMELINE AI - INITIALIZATION")
    logger.info("=" * 60)
    
    # Step 1: Check environment variables
    logger.info("\n📋 Step 1/5: Checking environment variables...")
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Missing environment variables: {', '.join(missing)}")
        logger.error("💡 Create a .env file with:")
        for var in missing:
            logger.error(f"   {var}=your_value_here")
        return False
    
    for var in required_vars:
        logger.info(f"   ✅ {var}")
    
    # Step 2: Initialize Qdrant client
    logger.info("\n🔌 Step 2/5: Connecting to Qdrant...")
    try:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        collections = qdrant_client.get_collections()
        logger.info(f"   ✅ Connected to Qdrant ({len(collections.collections)} collections found)")
        initialization_status["qdrant"] = True
    except Exception as e:
        logger.error(f"   ❌ Qdrant connection failed: {e}")
        return False
    
    # Step 3: Initialize embedding model
    logger.info("\n🧠 Step 3/5: Loading embedding model...")
    try:
        embedding_model = TextEmbedding()
        logger.info("   ✅ Embedding model loaded")
        initialization_status["embedding"] = True
    except Exception as e:
        logger.error(f"   ❌ Embedding model failed: {e}")
        return False
    
    # Step 4: Initialize Groq client
    logger.info("\n🤖 Step 4/5: Initializing Groq AI...")
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Test with a simple call
        test_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'ready'"}],
            max_tokens=10
        )
        if test_response.choices:
            logger.info("   ✅ Groq AI connected and working")
            initialization_status["groq"] = True
        else:
            raise Exception("Groq returned empty response")
    except Exception as e:
        logger.error(f"   ❌ Groq initialization failed: {e}")
        logger.warning("   ⚠️  App will work but AI features disabled")
        groq_client = None
    
    # Step 5: Setup collection
    logger.info(f"\n📦 Step 5/5: Setting up collection '{COLLECTION_NAME}'...")
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if COLLECTION_NAME in collection_names:
            logger.info(f"   ℹ️  Collection '{COLLECTION_NAME}' already exists")
        else:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"   ✅ Collection '{COLLECTION_NAME}' created")
        
        initialization_status["collection"] = True
    except Exception as e:
        logger.error(f"   ❌ Collection setup failed: {e}")
        return False
    
    # Mark as fully initialized
    initialization_status["initialized"] = True
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ INITIALIZATION COMPLETE - SERVER READY")
    logger.info("=" * 60)
    logger.info(f"📍 Qdrant: {os.getenv('QDRANT_URL')}")
    logger.info(f"📦 Collection: {COLLECTION_NAME}")
    logger.info(f"🧠 Embedding: FastEmbed (dim={VECTOR_DIM})")
    logger.info(f"🤖 AI Model: Groq Llama 3.3 70B")
    logger.info("=" * 60 + "\n")
    
    return True

# Initialize on import
if not initialize_app():
    logger.error("🛑 Initialization failed - server will not start properly")
    raise RuntimeError("Failed to initialize application")

# Cleanup on shutdown
def cleanup():
    logger.info("🔚 Shutting down gracefully...")

atexit.register(cleanup)

# ==================== USER MANAGEMENT ====================

class User(UserMixin):
    def __init__(self, id, username, email, patient_id, profile_data=None):
        self.id = id
        self.username = username
        self.email = email
        self.patient_id = patient_id
        self.profile_data = profile_data or {
            "age": "",
            "gender": "",
            "blood_type": "",
            "allergies": "",
            "chronic_conditions": "",
            "emergency_contact": "",
            "phone": "",
            "address": ""
        }

@login_manager.user_loader
def load_user(user_id):
    user_data = users_db.get(user_id)
    return user_data["user"] if user_data else None

@dataclass
class MedicalEvent:
    event_id: str
    patient_id: str
    timestamp: str
    event_type: str
    modality: str
    content: str
    doctor_name: str = ""
    hospital_name: str = ""

# ==================== STATIC FILE SERVING ====================

@app.route('/')
def index():
    """Serve main application page"""
    return send_from_directory('static', 'index.html')

@app.route('/patient/<patient_id>')
def patient_view(patient_id):
    """Serve patient timeline view"""
    return send_from_directory('static', 'patient_view.html')

# ==================== HEALTH CHECK ====================

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if initialization_status["initialized"] else "initializing",
        "components": {
            "qdrant": initialization_status["qdrant"],
            "embedding": initialization_status["embedding"],
            "collection": initialization_status["collection"],
            "groq": initialization_status["groq"]
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route("/api/status")
def api_status():
    """Detailed API status"""
    try:
        collections = qdrant_client.get_collections()
        collection_exists = COLLECTION_NAME in [c.name for c in collections.collections]
        
        stats = None
        if collection_exists:
            try:
                info = qdrant_client.get_collection(COLLECTION_NAME)
                stats = {
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count
                }
            except:
                pass
        
        return jsonify({
            "status": "operational",
            "initialized": initialization_status["initialized"],
            "qdrant": {
                "connected": initialization_status["qdrant"],
                "url": os.getenv("QDRANT_URL"),
                "collections": len(collections.collections)
            },
            "collection": {
                "name": COLLECTION_NAME,
                "exists": collection_exists,
                "stats": stats
            },
            "embedding": {
                "loaded": initialization_status["embedding"],
                "dimension": VECTOR_DIM
            },
            "ai": {
                "provider": "Groq",
                "model": "llama-3.3-70b-versatile",
                "status": "connected" if initialization_status["groq"] else "disconnected"
            },
            "users": {
                "registered": len(users_db)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================== AUTH ROUTES ====================

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")
        
        if not all([username, email, password]):
            return jsonify({"error": "All fields required"}), 400
        
        for user_data in users_db.values():
            if user_data["user"].email == email:
                return jsonify({"error": "Email already registered"}), 400
        
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_id = str(uuid.uuid4())
        # Generate human-readable patient ID
        patient_id = f"MED-{user_id[:8].upper()}"
        
        user = User(user_id, username, email, patient_id)
        users_db[user_id] = {
            "user": user,
            "password": hashed,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        login_user(user)
        logger.info(f"👤 New user registered: {username} ({patient_id})")
        
        return jsonify({
            "status": "success",
            "patient_id": patient_id,
            "username": username
        })
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        
        user_data = None
        for ud in users_db.values():
            if ud["user"].email == email:
                user_data = ud
                break
        
        if not user_data:
            return jsonify({"error": "Invalid credentials"}), 401
        
        if not bcrypt.checkpw(password.encode('utf-8'), user_data["password"]):
            return jsonify({"error": "Invalid credentials"}), 401
        
        login_user(user_data["user"])
        logger.info(f"🔐 User logged in: {user_data['user'].username}")
        
        return jsonify({
            "status": "success",
            "patient_id": user_data["user"].patient_id,
            "username": user_data["user"].username
        })
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    username = current_user.username
    logout_user()
    logger.info(f"👋 User logged out: {username}")
    return jsonify({"status": "logged out"})

@app.route("/me")
@login_required
def get_current_user():
    return jsonify({
        "username": current_user.username,
        "email": current_user.email,
        "patient_id": current_user.patient_id,
        "profile_data": current_user.profile_data
    })

@app.route("/update-profile", methods=["POST"])
@login_required
def update_profile():
    try:
        data = request.json
        profile_data = data.get("profile_data", {})
        
        # Update user's profile data
        current_user.profile_data.update(profile_data)
        
        # Update in database
        users_db[current_user.id]["user"].profile_data = current_user.profile_data
        
        logger.info(f"📝 Profile updated for user: {current_user.username}")
        
        return jsonify({
            "status": "success",
            "profile_data": current_user.profile_data
        })
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== DOCUMENT UPLOAD WITH OCR ====================

@app.route("/upload-document", methods=["POST"])
def upload_document():
    """Upload document with OCR text extraction"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        patient_id = request.form.get('patient_id')
        doctor_name = request.form.get('doctor_name', 'Unknown')
        hospital_name = request.form.get('hospital_name', 'Unknown')
        
        if not patient_id:
            return jsonify({"error": "patient_id required"}), 400
        
        logger.info(f"📄 Processing document for patient: {patient_id}")
        
        # Read file as base64
        file_content = file.read()
        base64_image = base64.b64encode(file_content).decode('utf-8')
        
        # Use Groq for OCR (vision capabilities)
        if groq_client:
            try:
                # Note: Groq doesn't have vision API yet, so we'll use a workaround
                # For now, we'll just store metadata and manual notes
                logger.info("⚠️ Using metadata extraction (Groq vision not available)")
                extracted_text = f"Document: {file.filename}"
                
                # Get manual notes if provided
                additional_notes = request.form.get('notes', '')
                if additional_notes:
                    extracted_text += f"\n\nNotes: {additional_notes}"
                
            except Exception as e:
                logger.error(f"OCR error: {e}")
                extracted_text = f"Document uploaded: {file.filename}"
        else:
            extracted_text = f"Document uploaded: {file.filename}"
        
        # Create event with extracted text
        doc_date = datetime.now(timezone.utc).isoformat()
        event = create_medical_event(
            extracted_text, 
            patient_id, 
            "document", 
            doc_date,
            doctor_name=doctor_name,
            hospital_name=hospital_name
        )
        
        vector = list(embedding_model.embed(event.content))[0].tolist()
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=event.event_id,
                vector=vector,
                payload={
                    "patient_id": event.patient_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "modality": "document",
                    "content": event.content,
                    "filename": file.filename,
                    "doctor_name": event.doctor_name,
                    "hospital_name": event.hospital_name
                }
            )]
        )
        
        logger.info(f"✅ Document stored: {event.event_id[:8]}...")
        
        return jsonify({
            "status": "success",
            "event_id": event.event_id,
            "extracted_text": extracted_text,
            "document_type": "document",
            "note": "Document stored. Add manual notes for better context."
        })
        
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== DATA INGESTION ====================

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.json
        required = ["content", "patient_id", "event_type"]
        
        for field in required:
            if not data.get(field):
                return jsonify({"error": f"Missing: {field}"}), 400
        
        doctor_name = data.get("doctor_name", "Unknown")
        hospital_name = data.get("hospital_name", "Unknown")
        
        event = create_medical_event(
            data["content"],
            data["patient_id"],
            data["event_type"],
            data.get("timestamp"),
            doctor_name,
            hospital_name
        )
        
        vector = list(embedding_model.embed(event.content))[0].tolist()
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=event.event_id,
                vector=vector,
                payload={
                    "patient_id": event.patient_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "modality": "text",
                    "content": event.content,
                    "doctor_name": event.doctor_name,
                    "hospital_name": event.hospital_name
                }
            )]
        )
        
        logger.info(f"📝 Event ingested: {event.event_id[:8]}... ({event.event_type})")
        
        return jsonify({"status": "stored", "event_id": event.event_id})
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== TIMELINE & ANALYSIS ====================

@app.route("/timeline-summary", methods=["POST"])
def timeline_summary():
    try:
        patient_id = request.json.get("patient_id")
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        points = fetch_timeline_events(patient_id)
        
        if not points:
            return jsonify({"error": "No events found"}), 404
        
        if len(points) == 1:
            timeline = build_patient_timeline(points)
            return jsonify({
                "timeline": timeline,
                "semantic_shift": 0.0,
                "overall_summary": "Only one event recorded. More data needed for timeline analysis.",
                "data_quality": compute_data_quality(timeline)
            })
        
        timeline = build_patient_timeline(points)
        earliest = fetch_point_with_vector(points[0].id)
        latest = fetch_point_with_vector(points[-1].id)
        
        shift = cosine_distance(earliest.vector, latest.vector)
        summary = ai_explain(build_overview_prompt(timeline))
        
        logger.info(f"📊 Timeline generated for {patient_id}: {len(points)} events")
        
        return jsonify({
            "timeline": timeline,
            "semantic_shift": round(float(shift), 3),
            "overall_summary": summary,
            "data_quality": compute_data_quality(timeline)
        })
    except Exception as e:
        logger.error(f"Timeline error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== PDF EXPORT ====================

@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    try:
        patient_id = request.json.get("patient_id")
        if not patient_id:
            return jsonify({"error": "patient_id required"}), 400
        
        points = fetch_timeline_events(patient_id)
        if not points:
            return jsonify({"error": "No events found"}), 404
        
        timeline = build_patient_timeline(points)
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=20
        )
        
        # Get hospital name if available
        hospital_name = timeline[0].get('hospital_name', 'General Hospital') if timeline else 'General Hospital'
        
        story.append(Paragraph("Medical Timeline Report", title_style))
        story.append(Paragraph(f"Hospital: {hospital_name}", styles['Normal']))
        story.append(Paragraph(f"Patient ID: {patient_id}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 0.4*inch))
        
        table_data = [['Date', 'Type', 'Details']]
        for e in timeline:
            date = datetime.fromisoformat(e['timestamp']).strftime('%m/%d/%Y %I:%M %p')
            table_data.append([
                Paragraph(date, styles['Normal']),
                Paragraph(e['event_type'], styles['Normal']),
                Paragraph(e['content'][:200] + ('...' if len(e['content']) > 200 else ''), styles['Normal'])
            ])
        
        table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 11),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP')
        ]))
        
        story.append(table)
        doc.build(story)
        buffer.seek(0)
        
        logger.info(f"📄 PDF exported for {patient_id}")
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'medical_timeline_{patient_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== HELPER FUNCTIONS ====================

def create_medical_event(content, patient_id, event_type, timestamp=None, doctor_name="", hospital_name=""):
    return MedicalEvent(
        event_id=str(uuid.uuid4()),
        patient_id=patient_id,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        event_type=event_type,
        modality="text",
        content=content,
        doctor_name=doctor_name,
        hospital_name=hospital_name
    )

def fetch_timeline_events(patient_id):
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
            ]),
            limit=100,
            with_payload=True
        )
        points = results[0]
        return sorted(points, key=lambda p: datetime.fromisoformat(p.payload["timestamp"]))
    except Exception as e:
        logger.error(f"Fetch timeline error: {e}")
        return []

def fetch_point_with_vector(event_id):
    return qdrant_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[event_id],
        with_vectors=True
    )[0]

def cosine_distance(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_patient_timeline(points):
    return [{
        "timestamp": p.payload["timestamp"],
        "event_type": p.payload["event_type"],
        "content": p.payload["content"],
        "doctor_name": p.payload.get("doctor_name", "Unknown"),
        "hospital_name": p.payload.get("hospital_name", "Unknown")
    } for p in sorted(points, key=lambda x: datetime.fromisoformat(x.payload["timestamp"]))]

def compute_data_quality(timeline):
    count = len(timeline)
    if count == 0:
        return {"label": "No Data", "description": "No medical records available"}
    
    times = [datetime.fromisoformat(e["timestamp"]) for e in timeline]
    span = (max(times) - min(times)).days + 1
    avg_len = sum(len(e["content"]) for e in timeline) / count
    
    if count >= 6 and span >= 7 and avg_len >= 40:
        return {"label": "Rich", "description": "Sufficient records for comprehensive analysis"}
    if count >= 3 and span >= 2:
        return {"label": "Moderate", "description": "Some continuity present, insights may be limited"}
    return {"label": "Sparse", "description": "Limited data, interpretation constrained"}

def build_overview_prompt(timeline):
    entries = "\n".join([
        f"- {e['timestamp']}: {e['event_type']} → {e['content']}" 
        for e in timeline
    ])
    return f"""You are a medical timeline summarization assistant.

Analyze this patient's medical timeline and provide a concise overview.

STRICT RULES:
- Do NOT diagnose any conditions
- Do NOT suggest treatments
- Do NOT infer causes
- Only describe observable patterns

Focus on:
- Frequency and timing of medical events
- Any progression or changes over time
- Continuity of care
- Temporal gaps or clusters

Timeline:
{entries}

Provide a neutral, factual summary of the medical history."""

def ai_explain(prompt):
    """AI explanation using Groq"""
    try:
        if not groq_client:
            logger.warning("Groq client not initialized")
            return "AI analysis unavailable. Groq API not configured."
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return "AI returned empty response. Timeline data is still accessible."
            
    except Exception as e:
        error_msg = str(e).lower()
        
        if "rate limit" in error_msg or "quota" in error_msg:
            logger.error(f"Groq rate limit: {e}")
            return "⚠️ AI rate limit reached. Timeline data is available below."
        elif "auth" in error_msg or "invalid" in error_msg:
            logger.error(f"Groq auth failed: {e}")
            return "⚠️ AI authentication failed. Check GROQ_API_KEY."
        else:
            logger.error(f"Groq error: {e}")
            return "⚠️ AI analysis temporarily unavailable. Timeline data is still visible below."

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# ==================== MAIN ====================

if __name__ == "__main__":
    logger.info("🚀 Starting development server...")
    logger.info("📍 Access at: http://localhost:5000")
    logger.info("💡 Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5000)