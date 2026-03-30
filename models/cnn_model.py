import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class CustomCNN(tf.keras.Model):
    def __init__(self, num_classes=10, **kwargs):
        # On passe les kwargs au parent (tf.keras.Model) pour gérer 'trainable', 'name', etc.
        super(CustomCNN, self).__init__(**kwargs)
        
        self.num_classes = num_classes

        # ── Data Augmentation ───────────────────
        self.augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="data_augmentation")

        # ── Blocs Conv ──────────────────────────
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.bn1   = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))

        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.bn2   = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))

        self.conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.bn3   = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))

        # ── Classifieur ─────────────────────────
        self.flatten  = layers.Flatten()
        self.dense1   = layers.Dense(256, activation='relu')
        self.dropout1 = layers.Dropout(0.4)
        self.dense2   = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    # IMPORTANT : Ajoute aussi cette méthode pour que Keras sache comment sauvegarder ton modèle
    def get_config(self):
        config = super(CustomCNN, self).get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config

    def call(self, inputs, training=False):
        # (Le reste de ton code call ne change pas)
        x = self.augmentation(inputs, training=training)
        x = self.pool1(ax := self.bn1(self.conv1(x), training=training)) # Version compacte ou garde la tienne
        # ... (garde ton code call actuel, il est très bien)

        # Bloc 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        # Bloc 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        # Classifieur
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        return self.output_layer(x)