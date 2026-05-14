# ============================================================
#  Tomato Disease Detection — CAE + CNN
#  Dataset: PlantVillage (Kaggle)
#  Author: RagulRajkummar

# ── 1. Imports ───────────────────────────────────────────────
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D,
    Reshape, UpSampling2D, Flatten, Dropout
)
from tensorflow.keras.models import Model

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ── 2. Config ────────────────────────────────────────────────
H, W, C = 128, 128, 3

# Update these paths based on what's inside your downloaded dataset folder
TRAIN_DIR = os.path.join(DATA_DIR, 'color')  # adjust if needed

CLASS_NAMES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

SOLUTIONS = {
    0: "Plowing down crop residue soon after harvest will facilitate decomposition "
       "and reduce spread. Fixed-copper based products are the primary tool for "
       "managing bacterial spot in tomato.",
    1: "Crush 1 aspirin into powder, mix with 4 cups of water. "
       "Spray on your plant every 2-3 weeks through the growing season "
       "to help prevent tomato blight effectively.",
    2: "Mix one heaping tablespoon of baking soda, one teaspoon of vegetable oil, "
       "and a small amount of mild soap. Spray the tomato plants and reapply regularly.",
    3: "An apple-cider and vinegar mix treats mold effectively. "
       "Corn and garlic spray can also prevent fungi outbreaks before they occur.",
    4: "Use raised beds and rotate crops yearly. Consider mulching to prevent "
       "water splash. Remove affected lower leaves early. Apply fungicides with "
       "chlorothalonil, copper, or mancozeb before disease appears.",
    5: "Bifenazate (Acramite) at 0.75–1 lb/acre is effective against two-spotted "
       "spider mites with lower toxicity to beneficial insects. "
       "Postharvest interval on tomatoes is seven days.",
    6: "Remove old plant debris after harvest. Rotate crops and avoid planting "
       "tomatoes where disease-prone plants grew last year. "
       "Ensure good air circulation and water in the morning.",
    7: "There is no cure for Tomato Yellow Leaf Curl Virus (TYLCV). "
       "Remove and destroy infected plants immediately to prevent spread.",
    8: "There is no cure for Mosaic Virus. Safely remove and destroy infected plants. "
       "Avoid planting in soil with root debris, as the virus thrives in root systems.",
    9: "Your tomato plant is healthy! Keep up the good care."
}

# ── 3. Load Data ─────────────────────────────────────────────
data = []
labels = []

for label_idx, class_name in enumerate(CLASS_NAMES):
    # Search for matching folder (case-insensitive friendly)
    pattern = os.path.join(TRAIN_DIR, f'*{class_name}*', '*.*')
    image_paths = glob.glob(pattern)

    if not image_paths:
        print(f"No images found for: {class_name}")
        continue

    print(f"Loading {len(image_paths)} images for {class_name}...")

    for img_path in image_paths:
        try:
            image = tf.keras.preprocessing.image.load_img(
                img_path, color_mode='rgb', target_size=(H, W)
            )
            image = np.array(image)
            data.append(image)
            labels.append(label_idx)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

data = np.array(data, dtype='float32') / 255.0   # normalize to [0, 1]
labels = np.array(labels)

print(f"\nTotal images loaded: {len(data)}")
print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# ── 4. Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, random_state=42
)

# ── 5. Visualise samples ─────────────────────────────────────
plt.figure(figsize=(15, 15))
for i in range(10):
    ax = plt.subplot(5, 2, i + 1)
    plt.imshow(X_train[i])
    plt.title(CLASS_NAMES[y_train[i]], fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig('sample_images.png')
plt.show()

# ── 6. Build Convolutional Autoencoder (CAE) ─────────────────
inputs = Input(shape=(H, W, C))

# Encoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

print("Encoder output shape before flatten:", x.shape)

# Bottleneck
flat_size = x.shape[1] * x.shape[2] * x.shape[3]
x = Flatten()(x)
encoded = Dense(10, activation='relu')(x)
encoded = Dropout(0.2)(encoded)

encoder_model = Model(inputs, encoded, name='encoder')
encoder_model.summary()

# Decoder
d = Dense(flat_size, activation='relu')(encoded)
d = Dropout(0.2)(d)
d = Reshape((x.shape[1] if hasattr(x, 'shape') else 2, 2, 16))(d)  # match encoder output

d = UpSampling2D((2, 2))(d)
d = Conv2D(16, (3, 3), activation='relu', padding='same')(d)

d = UpSampling2D((2, 2))(d)
d = Conv2D(32, (3, 3), activation='relu', padding='same')(d)

d = UpSampling2D((2, 2))(d)
d = Conv2D(32, (3, 3), activation='relu', padding='same')(d)

d = UpSampling2D((2, 2))(d)
d = Conv2D(64, (3, 3), activation='relu', padding='same')(d)

d = UpSampling2D((2, 2))(d)
d = Conv2D(128, (3, 3), activation='relu', padding='same')(d)

d = UpSampling2D((2, 2))(d)
d = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d)

autoencoder = Model(inputs, d, name='autoencoder')
autoencoder.summary()

# ── 7. Train CAE ─────────────────────────────────────────────
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
autoencoder.fit(
    X_train, X_train,
    batch_size=16,
    epochs=3,
    validation_data=(X_test, X_test)
)

os.makedirs('./models', exist_ok=True)
autoencoder.save('./models/cae.h5')
print("CAE saved to ./models/cae.h5")

# ── 8. Generate features using trained CAE ───────────────────
X_train_encoded = autoencoder.predict(X_train, verbose=1)
X_test_encoded  = autoencoder.predict(X_test,  verbose=1)

# ── 9. Build CNN Classifier ───────────────────────────────────
inputs2 = Input(shape=(H, W, C))

e = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs2)
e = MaxPooling2D((2, 2))(e)

e = Conv2D(64, (3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D((2, 2))(e)

e = Conv2D(64, (3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D((2, 2))(e)

e = Conv2D(32, (3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D((2, 2))(e)

e = Conv2D(32, (3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D((2, 2))(e)

e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D((2, 2))(e)

e = Flatten()(e)
e = Dense(10, activation='relu')(e)
e = Dropout(0.2)(e)

e = Dense(1000, activation='relu')(e)
e = Dropout(0.3)(e)
e = Dense(700, activation='relu')(e)
e = Dropout(0.5)(e)
e = Dense(350, activation='relu')(e)
output = Dense(10, activation='softmax')(e)

cnn_model = Model(inputs2, output, name='cnn_classifier')
cnn_model.summary()

# ── 10. Train CNN ─────────────────────────────────────────────
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = cnn_model.fit(
    X_train_encoded, y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.2
)

cnn_model.save('./models/cnn.h5')
print("CNN saved to ./models/cnn.h5")

# ── 11. Evaluate ──────────────────────────────────────────────
cnn_model.evaluate(X_test_encoded, y_test, verbose=2)

# ── 12. Plot Training History ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ── 13. Predict & Show Solution ───────────────────────────────
y_pred_probs = cnn_model.predict(X_test_encoded)
y_pred = np.argmax(y_pred_probs, axis=1)

# Show result for the first test image as a demo
sample_idx = 0
predicted_class = y_pred[sample_idx]

print(f"\nPredicted Disease : {CLASS_NAMES[predicted_class]}")
print(f"Suggested Solution: {SOLUTIONS[predicted_class]}")
