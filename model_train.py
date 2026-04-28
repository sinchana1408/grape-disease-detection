import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2, DenseNet121
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.densenet import preprocess_input as den_pre

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("saved_models", exist_ok=True)

# =========================
# PATHS
# =========================
TRAIN_DIR = r"grape_dataset/train"
TEST_DIR = r"grape_dataset/test"

RESULT_FILE = "model_comparison_results.txt"
open(RESULT_FILE, "w").close()

# =========================
# SETTINGS
# =========================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 4

# =========================
# LOG FUNCTION
# =========================
def log_results(text):
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(model, test_data, model_name):
    y_true = test_data.classes
    y_pred_probs = model.predict(test_data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(test_data.class_indices.keys())
    )

    result = f"""
=====================================================
MODEL : {model_name}

Accuracy  : {accuracy*100:.2f}%
Precision : {precision*100:.2f}%
Recall    : {recall*100:.2f}%
F1 Score  : {f1*100:.2f}%

Confusion Matrix :
{cm}

Classification Report :
{report}
=====================================================
"""
    print(result)
    log_results(result)

# =========================
# COMMON AUGMENTATION
# =========================
def get_generator(preprocess_func):
    return ImageDataGenerator(
        preprocessing_function=preprocess_func,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

# =========================
# DenseNet121
# =========================
train_gen_den = get_generator(den_pre)

train_data_den = train_gen_den.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='training', class_mode='categorical'
)

val_data_den = train_gen_den.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='validation', class_mode='categorical'
)

test_gen_den = ImageDataGenerator(preprocessing_function=den_pre)
test_data_den = test_gen_den.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

base = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model1 = models.Model(inputs=base.input, outputs=output)
model1.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

history1 = model1.fit(train_data_den, validation_data=val_data_den, epochs=EPOCHS)
model1.save("saved_models/grape_densenet121_model.keras")
evaluate_model(model1, test_data_den, "DenseNet121")

# =========================
# EfficientNetB0
# =========================
train_gen_eff = get_generator(eff_pre)

train_data_eff = train_gen_eff.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='training', class_mode='categorical'
)

val_data_eff = train_gen_eff.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='validation', class_mode='categorical'
)

test_gen_eff = ImageDataGenerator(preprocessing_function=eff_pre)
test_data_eff = test_gen_eff.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model2 = models.Model(inputs=base.input, outputs=output)
model2.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model2.fit(train_data_eff, validation_data=val_data_eff, epochs=EPOCHS)
model2.save("saved_models/grape_efficientnet_model.keras")
evaluate_model(model2, test_data_eff, "EfficientNetB0")

# =========================
# ResNet50
# =========================
train_gen_res = get_generator(res_pre)

train_data_res = train_gen_res.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='training', class_mode='categorical'
)

val_data_res = train_gen_res.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='validation', class_mode='categorical'
)

test_gen_res = ImageDataGenerator(preprocessing_function=res_pre)
test_data_res = test_gen_res.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model3 = models.Model(inputs=base.input, outputs=output)
model3.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

history3 = model3.fit(train_data_res, validation_data=val_data_res, epochs=EPOCHS)
model3.save("saved_models/grape_resnet50_model.keras")
evaluate_model(model3, test_data_res, "ResNet50")

# =========================
# MobileNetV2
# =========================
train_gen_mob = get_generator(mob_pre)

train_data_mob = train_gen_mob.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='training', class_mode='categorical'
)

val_data_mob = train_gen_mob.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    subset='validation', class_mode='categorical'
)

test_gen_mob = ImageDataGenerator(preprocessing_function=mob_pre)
test_data_mob = test_gen_mob.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model4 = models.Model(inputs=base.input, outputs=output)
model4.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

history4 = model4.fit(train_data_mob, validation_data=val_data_mob, epochs=EPOCHS)
model4.save("saved_models/grape_mobilenetv2_model.keras")
evaluate_model(model4, test_data_mob, "MobileNetV2")

# =========================
# ACCURACY GRAPH
# =========================
plt.plot(history1.history['val_accuracy'], label='DenseNet121')
plt.plot(history2.history['val_accuracy'], label='EfficientNetB0')
plt.plot(history3.history['val_accuracy'], label='ResNet50')
plt.plot(history4.history['val_accuracy'], label='MobileNetV2')
plt.legend()
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("accuracy_graph.png")
plt.show()

# =========================
# LOSS GRAPH
# =========================
plt.plot(history1.history['val_loss'], label='DenseNet121')
plt.plot(history2.history['val_loss'], label='EfficientNetB0')
plt.plot(history3.history['val_loss'], label='ResNet50')
plt.plot(history4.history['val_loss'], label='MobileNetV2')
plt.legend()
plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss_graph.png")
plt.show()

print("✅ All models trained, evaluated, and saved successfully.")