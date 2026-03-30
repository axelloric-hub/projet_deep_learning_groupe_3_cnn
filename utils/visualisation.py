import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, save_path="courbes_entrainement.png"):
    """
    Trace les courbes Train Loss vs Val Loss et Train Acc vs Val Acc
    extraites de l'objet History retourné par model.fit().
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Courbes d'entraînement — CNN CIFAR-10", fontsize=13, fontweight='bold')

    epochs = range(1, len(history.history['loss']) + 1)

    # ── Loss ────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, history.history['loss'],     color='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history.history['val_loss'], color='#d62728', linewidth=2, linestyle='--', label='Val Loss')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train Loss vs Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.4)

    # ── Accuracy ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(epochs, history.history['accuracy'],     color='#2ca02c', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, history.history['val_accuracy'], color='#ff7f0e', linewidth=2, linestyle='--', label='Val Accuracy')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train Accuracy vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Courbes sauvegardées : {save_path}")


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Affiche une matrice de confusion normalisée avec les noms de classes CIFAR-10.
    """
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Valeurs dans chaque cellule
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Classe prédite', fontsize=11)
    ax.set_ylabel('Classe réelle', fontsize=11)
    ax.set_title('Matrice de confusion — CNN CIFAR-10', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Matrice sauvegardée : {save_path}")