from flask import Flask, jsonify, request
from flask import render_template
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PayloadSchemaType
import numpy as np
from datetime import datetime
from google import genai


app = Flask(__name__)

qdrant_client = QdrantClient(
    url="https://5d606d1a-e79e-4ea4-b9e0-4619e8d5f0c2.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ynU8qmDmY8_NsxChqp_SzZvrfU9VEyZ1hotwqXuJrTA"
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test", methods=["POST"])
def test():
    return jsonify({"status": "test works"})

@app.route("/qdrant-test")
def qdrant_test():
    collections = qdrant_client.get_collections()
    return jsonify({
        "status": "connected",
        "collections": [c.name for c in collections.collections]
    })

COLLECTION_NAME = "medical_events"
VECTOR_DIM = 384

@app.route("/timeline-summary", methods=["POST"])
def timeline_summary():
    data = request.json

    points = search_events(
        query_text=data["query"],
        patient_id=data["patient_id"],
        limit=10
    )

    if len(points) == 0:
        return jsonify({"error": "No events found"})

    timeline = build_patient_timeline(points)

    prompt = f"""
You are a medical record summarization assistant.

Rules:
- Do NOT diagnose
- Do NOT suggest treatment
- Only summarize what is explicitly stated
- Mention progression or stability if present
- Reference time order

Patient timeline:
"""

    for event in timeline:
        prompt += f"""
[{event['timestamp']}] ({event['event_type']}):
{event['content']}
"""

    prompt += "\nWrite a concise summary of how the patient's condition evolved."

    explanation = ai_explain(prompt)

    return jsonify({
        "timeline": timeline,
        "summary": explanation
    })

"""@app.route("/setup-collection")
def setup_collection():
    collections = qdrant_client.get_collections()
    names = [c.name for c in collections.collections]

    if COLLECTION_NAME not in names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE
            )
        )
        return jsonify({"status": "collection created"})
    
    return jsonify({"status": "collection already exists"})"""

"""@app.route("/setup-payload-index")
def setup_payload_index():
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="patient_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    return jsonify({"status": "payload index created for patient_id"})"""


embedding_model = TextEmbedding()

@dataclass
class MedicalEvent:
    event_id: str
    patient_id: str
    timestamp: str
    event_type: str
    modality: str
    content: str

def create_medical_event(content, patient_id, event_type, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return MedicalEvent(
        event_id=str(uuid.uuid4()),
        patient_id=patient_id,
        timestamp=timestamp,
        event_type=event_type,
        modality="text",
        content=content
    )

@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.json

    # sanitize inputs
    patient_name = data.get("patient_name")
    doctor_name = data.get("doctor_name")

    if not isinstance(patient_name, str) or not patient_name.strip():
        patient_name = "Unknown"

    if not isinstance(doctor_name, str) or not doctor_name.strip():
        doctor_name = "Self"

    event = create_medical_event(
        content=data["content"],
        patient_id=data["patient_id"],
        event_type=data["event_type"],
        timestamp=data.get("timestamp")
    )

    vector = list(embedding_model.embed(event.content))[0].tolist()

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=event.event_id,
                vector=vector,
                payload={
                    "patient_id": event.patient_id,
                    "patient_name": patient_name,
                    "doctor_name": doctor_name,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "modality": "text",
                    "content": event.content
                }
            )
        ]
    )

    return jsonify({
        "status": "stored",
        "event_id": event.event_id
    })

@app.route("/search", methods=["POST"])
def search():
    data = request.json

    points = search_events(
        query_text=data["query"],
        patient_id=data["patient_id"]
    )

    # Format results for frontend
    response = []
    for p in points:
        response.append({
            "event_id": p.id,
            "score": p.score,
            "content": p.payload["content"],
            "timestamp": p.payload["timestamp"],
            "event_type": p.payload["event_type"]
        })

    return jsonify(response)

def search_events(query_text: str, patient_id: str, limit: int = 5):
    # Convert query to vector
    query_vector = list(embedding_model.embed(query_text))[0].tolist()

    # Filter to only this patient
    search_filter = Filter(
        must=[
            FieldCondition(
                key="patient_id",
                match=MatchValue(value=patient_id)
            )
        ]
    )

    # Perform vector search
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=limit
    )

    return results.points

def fetch_point_with_vector(event_id: str):
    return qdrant_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[event_id],
        with_vectors=True
    )[0]

def cosine_distance(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sort_points_by_time(points):
    return sorted(
        points,
        key=lambda p: datetime.fromisoformat(p.payload["timestamp"])
    )
def build_patient_timeline(points):
    ordered = sort_points_by_time(points)
    timeline = []

    for p in ordered:
        timeline.append({
            "timestamp": p.payload["timestamp"],
            "event_type": p.payload["event_type"],
            "content": p.payload["content"]
        })

    return timeline

def compute_difference(points):
    if len(points) < 2:
        return {"error": "Not enough events to compute differences"}

    ordered = sort_points_by_time(points)

    earliest_meta = ordered[0]
    latest_meta = ordered[-1]

    earliest = fetch_point_with_vector(earliest_meta.id)
    latest = fetch_point_with_vector(latest_meta.id)

    semantic_shift = cosine_distance(
        earliest.vector,
        latest.vector
    )

    metadata_changes = {}
    for key in ["event_type", "modality"]:
        if earliest.payload.get(key) != latest.payload.get(key):
            metadata_changes[key] = (
                earliest.payload.get(key),
                latest.payload.get(key)
            )

    # Human-readable change label
    if semantic_shift < 0.2:
        change_level = "Low"
    elif semantic_shift < 0.5:
        change_level = "Moderate"
    else:
        change_level = "High"

    return {
        "time_range": {
            "from": earliest.payload["timestamp"],
            "to": latest.payload["timestamp"]
        },
        "events_compared": {
            "earliest_id": earliest.id,
            "latest_id": latest.id
        },
        "semantic_shift": round(float(semantic_shift), 3),
        "change_level": change_level,
        "metadata_changes": metadata_changes
    }

@app.route("/difference", methods=["POST"])
def difference():
    data = request.json

    points = search_events(
        query_text=data["query"],
        patient_id=data["patient_id"]
    )

    diff = compute_difference(points)

    return jsonify(diff)

def build_explanation_prompt(
    earliest_text: str,
    latest_text: str,
    diff_result: dict
):
    return f"""
You are a medical record comparison assistant.

Your task is to describe how two medical records differ over time.
You must follow these rules strictly:

- Do NOT diagnose any condition.
- Do NOT infer causes.
- Do NOT suggest treatments.
- Only describe differences that are explicit or strongly implied by the text.
- If differences are unclear, say so.

Context:
Time range: {diff_result["time_range"]["from"]} to {diff_result["time_range"]["to"]}
Semantic change score: {diff_result["semantic_shift"]}
Change level: {diff_result["change_level"]}

Earlier record:
\"\"\"
{earliest_text}
\"\"\"

Later record:
\"\"\"
{latest_text}
\"\"\"

Now write a short, neutral explanation of how the content changed over time.
"""

def ai_explain(prompt: str):
    client = genai.Client(api_key="AIzaSyAkixXheA9bAsjgbvKZ7MUYUyh5HOtyb7c")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text


@app.route("/explain", methods=["POST"])
def explain():
    data = request.json

    # Step 1: Get search results
    points = search_events(
        query_text=data["query"],
        patient_id=data["patient_id"]
    )

    # Step 2: Compute difference
    diff = compute_difference(points)

    if "error" in diff:
        return jsonify(diff)

    # Step 3: Fetch full records
    earliest = fetch_point_with_vector(diff["events_compared"]["earliest_id"])
    latest = fetch_point_with_vector(diff["events_compared"]["latest_id"])

    # Step 4: Build prompt
    prompt = build_explanation_prompt(
        earliest_text=earliest.payload["content"],
        latest_text=latest.payload["content"],
        diff_result=diff
    )

    # Step 5: Ask AI
    explanation = ai_explain(prompt)

    return jsonify({
        "difference": diff,
        "explanation": explanation
    })


print(app.url_map)

if __name__ == "__main__":
    app.run(debug=True)
