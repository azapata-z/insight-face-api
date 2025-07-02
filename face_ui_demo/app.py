from flask import Flask, request, jsonify
from flask_cors import CORS
import os, base64, cv2
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

known_db = {}
SIMILARITY_THRESHOLD = 0.60  # You can tweak this based on accuracy needs

def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    return faces[0].embedding if faces else None

def find_best_match(test_embedding):
    best_name = None
    best_score = 0.0
    for name, embeddings in known_db.items():
        scores = cosine_similarity([test_embedding], embeddings)[0]
        max_score = np.max(scores)
        if max_score > best_score:
            best_score = max_score
            best_name = name
    return best_name, best_score

@app.route('/api/register', methods=['POST'])
def register():
    name = request.form.get('name')
    files = request.files.getlist('images')  # Multiple files expected under key 'images'

    if not name or not files:
        return jsonify({"error": "Name and at least one image required"}), 400

    if len(files) > 5:
        return jsonify({"error": "Maximum 5 images allowed"}), 400

    embeddings_added = 0
    for file in files:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        embedding = get_embedding(path)
        if embedding is not None:
            known_db.setdefault(name, []).append(embedding)
            embeddings_added += 1

    if embeddings_added == 0:
        return jsonify({"error": "No valid faces detected in any image"}), 400

    return jsonify({
        "message": f"{embeddings_added} face(s) registered for {name}"
    }), 200

@app.route('/api/verify', methods=['POST'])
def verify():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "Image required"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    embedding = get_embedding(path)
    if embedding is None:
        return jsonify({"error": "No face detected"}), 400

    if not known_db:
        return jsonify({"error": "No registered faces"}), 400

    best_name, best_score = find_best_match(embedding)
    if best_score >= SIMILARITY_THRESHOLD:
        return jsonify({"match": True, "name": best_name, "score": float(best_score)})
    else:
        return jsonify({"match": False, "score": float(best_score)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
