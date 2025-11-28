from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, numpy as np, cv2
from insightface.app import FaceAnalysis
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_NAME = os.environ.get("FACE_MODEL_NAME", "buffalo_s") 
PROVIDER = os.environ.get("FACE_PROVIDER", "CPUExecutionProvider")

face_app = FaceAnalysis(name=MODEL_NAME, providers=[PROVIDER])
face_app.prepare(ctx_id=0)

def decode_base64_image(b64):
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def get_embedding(img):
    faces = face_app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    return faces[0].embedding

@app.post("/api/embedding")
def embedding():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "image base64 required"}), 400

    img = decode_base64_image(payload["image"])
    if img is None:
        return jsonify({"error": "invalid image"}), 400

    emb = get_embedding(img)
    if emb is None:
        return jsonify({"error": "no face detected"}), 400

    return jsonify({"embedding": emb.tolist()}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
