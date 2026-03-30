"""
evaluate.py — Évaluation complète du CNN CIFAR-10
==================================================
Génère :
  - Accuracy / Loss globale
  - Rapport de classification (précision, rappel, F1 par classe)
  - Matrice de confusion (normalisée + brute)
  - Top-5 accuracy
  - Courbes ROC par classe (one-vs-rest)
  - Exemples d'images mal classifiées
  - Export JSON du rapport complet
"""
 
import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
 
# ── Import du modèle custom (nécessaire pour la désérialisation) ──────────────
from models.cnn_model import CustomCNN  # noqa: F401
from utils.data_loader import load_cifar10, make_datasets, CLASS_NAMES
 
# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "best_cnn_model.keras"
OUTPUT_DIR   = "evaluation_results"
BATCH_SIZE   = 128
NUM_CLASSES  = 10
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────
 
def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
 
 
def save_fig(fig, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Sauvegardé : {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement
# ─────────────────────────────────────────────────────────────────────────────
 
def load_everything():
    print("\n[1/7] Chargement du modèle et des données …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"      Modèle chargé depuis '{MODEL_PATH}'")
 
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    _, test_ds = make_datasets(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE)
 
    y_test_flat = y_test.flatten()
    return model, test_ds, x_test, y_test_flat
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Métriques globales
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_global(model, test_ds):
    print("\n[2/7] Calcul des métriques globales …")
    t0 = time.time()
    loss, acc = model.evaluate(test_ds, verbose=1)
    elapsed = time.time() - t0
    print(f"\n  Loss          : {loss:.4f}")
    print(f"  Accuracy      : {acc * 100:.2f}%")
    print(f"  Temps total   : {elapsed:.1f}s")
    return {"loss": round(loss, 6), "accuracy": round(acc, 6), "eval_time_s": round(elapsed, 2)}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Prédictions
# ─────────────────────────────────────────────────────────────────────────────
 
def get_predictions(model, test_ds, y_true):
    print("\n[3/7] Génération des prédictions …")
    probas = model.predict(test_ds, verbose=1)          # (N, 10)
    y_pred = np.argmax(probas, axis=1)
 
    # Top-5 accuracy
    top5_correct = 0
    for i, p in enumerate(probas):
        if y_true[i] in np.argsort(p)[-5:]:
            top5_correct += 1
    top5_acc = top5_correct / len(y_true)
    print(f"  Top-1 Accuracy : {np.mean(y_pred == y_true) * 100:.2f}%")
    print(f"  Top-5 Accuracy : {top5_acc * 100:.2f}%")
    return probas, y_pred, top5_acc
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Rapport de classification
# ─────────────────────────────────────────────────────────────────────────────
 
def print_classification_report(y_true, y_pred):
    print("\n[4/7] Rapport de classification …")
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    print(report_str)
    return report_str, report_dict
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Matrice de confusion
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_confusion_matrices(y_true, y_pred):
    print("\n[5/7] Génération des matrices de confusion …")
 
    cm_raw  = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1, keepdims=True)
 
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Matrices de Confusion — CNN CIFAR-10", fontsize=15, fontweight="bold", y=1.01)
 
    for ax, data, title, fmt, vmax in zip(
        axes,
        [cm_raw, cm_norm],
        ["Valeurs absolues", "Normalisée (par ligne)"],
        ["d", ".2f"],
        [None, 1.0],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Classe prédite", fontsize=11)
        ax.set_ylabel("Classe réelle", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
 
    plt.tight_layout()
    save_fig(fig, "confusion_matrices.png")
    return cm_raw, cm_norm
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Courbes ROC (one-vs-rest)
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_roc_curves(y_true, probas):
    print("\n[6/7] Courbes ROC …")
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
 
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    auc_scores = {}
 
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = round(roc_auc, 4)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{cls} (AUC = {roc_auc:.3f})")
 
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de Faux Positifs", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs", fontsize=12)
    ax.set_title("Courbes ROC par classe (One-vs-Rest) — CNN CIFAR-10", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
 
    macro_auc = np.mean(list(auc_scores.values()))
    ax.set_title(
        f"Courbes ROC par classe (One-vs-Rest) — CNN CIFAR-10\nMacro-AUC = {macro_auc:.4f}",
        fontsize=13, fontweight="bold"
    )
 
    plt.tight_layout()
    save_fig(fig, "roc_curves.png")
    print(f"  Macro-AUC : {macro_auc:.4f}")
    return auc_scores, macro_auc
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Exemples mal classifiés
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_misclassified(x_test, y_true, y_pred, probas, n=20):
    print("\n[7/7] Visualisation des erreurs …")
    errors = np.where(y_pred != y_true)[0]
    print(f"  {len(errors)} erreurs sur {len(y_true)} images ({len(errors)/len(y_true)*100:.1f}%)")
 
    # Trier par confiance décroissante (les plus sûres mais fausses)
    conf = probas[errors, y_pred[errors]]
    top_errors = errors[np.argsort(conf)[::-1][:n]]
 
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    fig.suptitle(
        f"Top {n} prédictions erronées (par confiance décroissante)",
        fontsize=13, fontweight="bold"
    )
 
    for ax_idx, img_idx in enumerate(top_errors):
        ax = axes[ax_idx // cols][ax_idx % cols]
        img = x_test[img_idx]
        ax.imshow(img)
        pred_cls  = CLASS_NAMES[y_pred[img_idx]]
        true_cls  = CLASS_NAMES[y_true[img_idx]]
        conf_val  = probas[img_idx, y_pred[img_idx]] * 100
        ax.set_title(
            f"Prédit : {pred_cls} ({conf_val:.0f}%)\nRéel : {true_cls}",
            fontsize=8,
            color="red",
        )
        ax.axis("off")
 
    # Masquer les axes vides
    for ax_idx in range(len(top_errors), rows * cols):
        axes[ax_idx // cols][ax_idx % cols].axis("off")
 
    plt.tight_layout()
    save_fig(fig, "misclassified_examples.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. Accuracy par classe
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_per_class_accuracy(y_true, y_pred):
    per_class_acc = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_true == i
        per_class_acc[cls] = round(float(np.mean(y_pred[mask] == y_true[mask])), 4)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        CLASS_NAMES,
        [per_class_acc[c] * 100 for c in CLASS_NAMES],
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, NUM_CLASSES)),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy par classe — CNN CIFAR-10", fontsize=13, fontweight="bold")
    ax.axhline(y=np.mean(list(per_class_acc.values())) * 100, color="navy",
               linestyle="--", linewidth=1.5, label="Moyenne globale")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    save_fig(fig, "per_class_accuracy.png")
    return per_class_acc
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. Export JSON
# ─────────────────────────────────────────────────────────────────────────────
 
def export_json(global_metrics, top5_acc, report_dict, auc_scores, macro_auc, per_class_acc, cm_raw):
    report = {
        "model_path": MODEL_PATH,
        "dataset": "CIFAR-10",
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "global_metrics": {
            **global_metrics,
            "top5_accuracy": round(top5_acc, 6),
        },
        "per_class_accuracy": per_class_acc,
        "per_class_f1": {
            cls: round(report_dict[cls]["f1-score"], 6)
            for cls in CLASS_NAMES
        },
        "per_class_precision": {
            cls: round(report_dict[cls]["precision"], 6)
            for cls in CLASS_NAMES
        },
        "per_class_recall": {
            cls: round(report_dict[cls]["recall"], 6)
            for cls in CLASS_NAMES
        },
        "roc_auc_per_class": auc_scores,
        "macro_auc": round(macro_auc, 6),
        "macro_avg": report_dict["macro avg"],
        "weighted_avg": report_dict["weighted avg"],
        "confusion_matrix": cm_raw.tolist(),
    }
 
    json_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  → Rapport JSON : {json_path}")
    return report
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    # 1. DÉCLARER GLOBAL EN PREMIER
    global MODEL_PATH, OUTPUT_DIR, BATCH_SIZE
    
    # 2. ENSUITE UTILISER LES VARIABLES (Initialisation du parser)
    parser = argparse.ArgumentParser(description="Évaluation CNN CIFAR-10")
    parser.add_argument("--model",  default=MODEL_PATH,  help="Chemin vers le .keras")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Dossier de sortie")
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE, help="Taille du batch")
    args = parser.parse_args()
 
    # 3. RÉASSIGNER LES VALEURS
    MODEL_PATH  = args.model
    OUTPUT_DIR  = args.output
    BATCH_SIZE  = args.batch
    
    # ... reste du code ...
 
    ensure_output_dir(OUTPUT_DIR)
 
    print("=" * 60)
    print("   ÉVALUATION CNN — CIFAR-10")
    print("=" * 60)
 
    # Pipeline complet
    model, test_ds, x_test, y_true    = load_everything()
    global_metrics                    = evaluate_global(model, test_ds)
    probas, y_pred, top5_acc          = get_predictions(model, test_ds, y_true)
    report_str, report_dict           = print_classification_report(y_true, y_pred)
    cm_raw, cm_norm                   = plot_confusion_matrices(y_true, y_pred)
    auc_scores, macro_auc             = plot_roc_curves(y_true, probas)
    per_class_acc                     = plot_per_class_accuracy(y_true, y_pred)
    plot_misclassified(x_test, y_true, y_pred, probas, n=20)
    report = export_json(
        global_metrics, top5_acc, report_dict,
        auc_scores, macro_auc, per_class_acc, cm_raw
    )
 
    print("\n" + "=" * 60)
    print("   RÉSUMÉ")
    print("=" * 60)
    print(f"  Test Loss          : {global_metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy     : {global_metrics['accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy     : {top5_acc*100:.2f}%")
    print(f"  Macro-AUC          : {macro_auc:.4f}")
    print(f"  Macro F1-Score     : {report_dict['macro avg']['f1-score']*100:.2f}%")
    print(f"\n  Fichiers dans : '{OUTPUT_DIR}/'")
    print("    - confusion_matrices.png")
    print("    - roc_curves.png")
    print("    - per_class_accuracy.png")
    print("    - misclassified_examples.png")
    print("    - evaluation_report.json")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()