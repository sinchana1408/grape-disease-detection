import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# ✅ SAME PREPROCESS FUNCTIONS USED DURING TRAINING
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.densenet import preprocess_input as den_pre

IMG_SIZE = 128

# ✅ CORRECT CLASS ORDER (VERY IMPORTANT)
# Based on your training report
class_names = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy"
]

# ✅ CLEAN LABELS FOR FRONTEND
def clean_label(label):
    return (
        label.replace("Grape___", "")
             .replace("(Black_Measles)", "")
             .replace("(Isariopsis_Leaf_Spot)", "")
             .replace("_", " ")
    )

# ✅ MODEL-SPECIFIC PREPROCESSING
def preprocess_image(image, model_name):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype("float32")

    # ensure RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    if model_name == "resnet":
        img_array = res_pre(img_array)
    elif model_name == "mobilenet":
        img_array = mob_pre(img_array)
    elif model_name == "efficientnet":
        img_array = eff_pre(img_array)
    elif model_name == "densenet":
        img_array = den_pre(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ✅ SINGLE MODEL PREDICTION
def predict_with_model(model, image, model_name):
    processed = preprocess_image(image, model_name)

    preds = model.predict(processed)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    raw_label = class_names[class_index]
    prediction_label = clean_label(raw_label)

    # 🔍 DEBUG (remove later if needed)
    print(f"\nModel: {model_name}")
    print("Raw preds:", preds)
    print("Predicted index:", class_index)
    print("Raw label:", raw_label)
    print("Clean label:", prediction_label)

    return {
        "prediction": prediction_label,
        "confidence": confidence
    }


# ✅ MULTI-MODEL PREDICTION
def predict_all_models(image, models):
    results = {}

    def run_model(name, model):
        try:
            return name, predict_with_model(model, image, name)
        except Exception as e:
            print(f"{name} failed:", e)
            return name, {"prediction": "Error", "confidence": 0.0}

    # 🔥 RUN MODELS IN PARALLEL
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_model, name, model)
            for name, model in models.items()
        ]

        for future in futures:
            name, result = future.result()
            results[name] = result

    # ✅ SELECT BEST MODEL (highest confidence)
    best_model = max(results, key=lambda x: results[x]["confidence"])

    return {
        "models": results,
        "final_prediction": results[best_model]["prediction"],
        "final_confidence": float(results[best_model]["confidence"]),
        "best_model": best_model
    }