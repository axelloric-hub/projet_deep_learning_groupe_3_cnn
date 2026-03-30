import tensorflow as tf
import tensorflow as tf
# Importation de ton modèle (dépendance 1)
from models.cnn_model import CustomCNN 
# Importation de tes fonctions de données (dépendance 2)
from utils.data_loader import load_cifar10, make_datasets 
# Importation de tes graphiques (dépendance 3)
from utils.visualisation import plot_history


def main():
    print("=" * 55)
    print("   Entraînement CNN — CIFAR-10")
    print("=" * 55)

    # ── 1. Chargement et préparation des données ─────────────────────
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    train_ds, test_ds = make_datasets(x_train, y_train, x_test, y_test, batch_size=64)

    # ── 2. Construction du modèle ────────────────────────────────────
    model = CustomCNN(num_classes=10)

    # ── 3. Compilation ───────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # ── 4. Callbacks ─────────────────────────────────────────────────
    callbacks = [
        # Arrête l'entraînement si val_loss ne s'améliore plus
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduit le lr si stagnation
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        # Sauvegarde le meilleur modèle automatiquement
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_cnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # ── 5. Entraînement ──────────────────────────────────────────────
    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=test_ds,
        callbacks=callbacks
    )

    # ── 6. Évaluation rapide ─────────────────────────────────────────
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Loss     : {loss:.4f}")
    print(f"Test Accuracy : {acc * 100:.2f}%")

    # ── 7. Visualisation des courbes ─────────────────────────────────
    plot_history(history)

    # ── 8. Résumé du modèle ──────────────────────────────────────────
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()


if __name__ == "__main__":
    main()