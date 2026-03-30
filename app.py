"""
API Flask — CNN CIFAR-10
"""

import io
import base64
import time
import logging
import numpy as np
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from functools import wraps

try:
    from models.cnn_model import CustomCNN  # noqa: F401
    print("✅ CustomCNN importé avec succès")
except Exception as e:
    print(f"❌ Erreur d'import CustomCNN : {e}")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
IMG_SIZE   = (32, 32)
MODEL_PATH = os.path.join(BASE_DIR, "best_cnn_model.keras")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model            = None
model_load_time  = None
model_load_error = None

EVAL_REPORT = {
    "dataset": "CIFAR-10",
    "test_samples": 10000,
    "global_metrics": {
        "accuracy": 0.7968,
        "loss": 0.595431,
        "top5_accuracy": 0.9856,
        "eval_time_s": 14.41,
    },
    "per_class_accuracy": {
        "airplane": 0.874, "automobile": 0.912, "bird": 0.707,
        "cat": 0.573, "deer": 0.747, "dog": 0.649,
        "frog": 0.922, "horse": 0.850, "ship": 0.869, "truck": 0.865,
    },
    "per_class_f1": {
        "airplane": 0.821043, "automobile": 0.881585, "bird": 0.756554,
        "cat": 0.619794, "deer": 0.772492, "dog": 0.701622,
        "frog": 0.825056, "horse": 0.839092, "ship": 0.895415, "truck": 0.824595,
    },
    "per_class_precision": {
        "airplane": 0.774136, "automobile": 0.853134, "bird": 0.813579,
        "cat": 0.674912, "deer": 0.799786, "dog": 0.763529,
        "frog": 0.746559, "horse": 0.828460, "ship": 0.923486, "truck": 0.787796,
    },
    "per_class_recall": {
        "airplane": 0.874, "automobile": 0.912, "bird": 0.707,
        "cat": 0.573, "deer": 0.747, "dog": 0.649,
        "frog": 0.922, "horse": 0.850, "ship": 0.869, "truck": 0.865,
    },
    "roc_auc_per_class": {
        "airplane": 0.9850, "automobile": 0.9937, "bird": 0.9654,
        "cat": 0.9458, "deer": 0.9762, "dog": 0.9645,
        "frog": 0.9874, "horse": 0.9864, "ship": 0.9919, "truck": 0.9873,
    },
    "macro_auc": 0.97836,
    "macro_avg": {"precision": 0.7965, "recall": 0.7968, "f1_score": 0.7937},
}


def load_model():
    global model, model_load_time, model_load_error
    try:
        if not os.path.exists(MODEL_PATH):
            model_load_error = f"Fichier modèle '{MODEL_PATH}' non trouvé"
            logger.error(model_load_error)
            return
        logger.info(f"Chargement du modèle depuis '{MODEL_PATH}' …")
        t0 = time.time()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.predict(np.zeros((1, 32, 32, 3), dtype="float32"), verbose=0)
        model_load_time = round(time.time() - t0, 2)
        logger.info(f"Modèle prêt en {model_load_time}s")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Erreur chargement modèle : {e}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def model_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if model is None:
            return jsonify({
                "error": "Modèle non disponible",
                "detail": model_load_error or "Non chargé."
            }), 503
        return f(*args, **kwargs)
    return decorated


def build_prediction(proba: np.ndarray) -> dict:
    idx = int(np.argmax(proba))
    sorted_idx = np.argsort(proba)[::-1]
    return {
        "predicted_class": CLASS_NAMES[idx],
        "predicted_index": idx,
        "confidence": round(float(proba[idx]), 6),
        "top5": [
            {"class": CLASS_NAMES[i], "confidence": round(float(proba[i]), 6)}
            for i in sorted_idx[:5]
        ],
    }


# ── Routes HTML ───────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


# ── Routes JSON ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    payload = {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }
    if model_load_time is not None:
        payload["model_load_time_s"] = model_load_time
    if model_load_error:
        payload["model_load_error"] = model_load_error
    return jsonify(payload), 200 if model else 503


@app.route("/model/info", methods=["GET"])
@model_required
def model_info():
    total = model.count_params()
    try:
        trainable = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    except Exception:
        trainable = None
    return jsonify({
        "architecture": "CustomCNN",
        "input_shape": [None, 32, 32, 3],
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable if trainable else None,
        "layers": [
            {"name": l.name, "type": l.__class__.__name__}
            for l in model.layers
        ],
    }), 200


@app.route("/model/stats", methods=["GET"])
def model_stats():
    return jsonify(EVAL_REPORT), 200


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify({
        "name": "CNN CIFAR-10 API",
        "version": "1.0.0",
        "classes": CLASS_NAMES,
        "endpoints": {
            "GET  /":              "Interface HTML — Classifier",
            "GET  /dashboard":     "Interface HTML — Dashboard",
            "GET  /health":        "État de santé de l'API",
            "GET  /model/info":    "Architecture et paramètres du modèle",
            "GET  /model/stats":   "Métriques d'évaluation complètes",
            "GET  /api/docs":      "Cette documentation",
            "POST /predict":       "Prédiction image unique (multipart ou JSON base64)",
            "POST /predict/batch": "Prédiction batch — JSON {'images': ['<b64>',…]}",
        },
    }), 200


@app.route("/predict", methods=["POST"])
@model_required
def predict():
    t0 = time.time()
    image_bytes = None

    if request.content_type and "multipart" in request.content_type:
        if "image" not in request.files:
            return jsonify({"error": "Champ 'image' manquant."}), 400
        image_bytes = request.files["image"].read()
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "Champ 'image' (base64) manquant."}), 400
        try:
            image_bytes = base64.b64decode(data["image"])
        except Exception as e:
            return jsonify({"error": f"Base64 invalide: {e}"}), 400
    else:
        return jsonify({"error": "Content-Type non supporté."}), 415

    try:
        tensor = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Image illisible : {e}"}), 422

    proba  = model.predict(tensor, verbose=0)[0]
    result = build_prediction(proba)
    result["inference_time_ms"] = round((time.time() - t0) * 1000, 2)
    return jsonify(result), 200


@app.route("/predict/batch", methods=["POST"])
@model_required
def predict_batch():
    MAX_BATCH = 32
    t0 = time.time()

    if not request.is_json:
        return jsonify({"error": "application/json requis."}), 415

    data = request.get_json(silent=True) or {}
    images_b64 = data.get("images", [])

    if not isinstance(images_b64, list) or not images_b64:
        return jsonify({"error": "'images' doit être une liste non vide."}), 400
    if len(images_b64) > MAX_BATCH:
        return jsonify({"error": f"Maximum {MAX_BATCH} images. Reçu : {len(images_b64)}."}), 400

    tensors, errors = [], []
    for i, b64 in enumerate(images_b64):
        try:
            tensors.append(preprocess_image(base64.b64decode(b64))[0])
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    if not tensors:
        return jsonify({"error": "Aucune image valide.", "details": errors}), 422

    probas      = model.predict(np.stack(tensors), verbose=0)
    predictions = [build_prediction(p) for p in probas]

    return jsonify({
        "count":             len(predictions),
        "predictions":       predictions,
        "inference_time_ms": round((time.time() - t0) * 1000, 2),
        "errors":            errors or None,
    }), 200


# ── Erreurs globales ──────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route introuvable.", "detail": str(e)}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Méthode non autorisée."}), 405

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Fichier trop volumineux (max 16 Mo)."}), 413

@app.errorhandler(500)
def internal(e):
    logger.exception("Erreur interne")
    return jsonify({"error": "Erreur interne.", "detail": str(e)}), 500


# ── Entrée principale ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DÉMARRAGE DE L'API CNN CIFAR-10")
    print("="*60)
    load_model()
    print(f"\n{'✅ Modèle chargé' if model else '❌ Modèle non chargé : ' + str(model_load_error)}")
    print("\n🚀 Serveur sur http://localhost:5000")
    print("="*60)
    app.run(host="0.0.0.0", port=5000, debug=False)
