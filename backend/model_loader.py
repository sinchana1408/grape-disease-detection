import os
from tensorflow.keras.models import load_model

models = {}

def load_all_models():
    global models

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    models["resnet"] = load_model(os.path.join(BASE_DIR, "saved_models/grape_resnet50_model.keras"))
    models["mobilenet"] = load_model(os.path.join(BASE_DIR, "saved_models/grape_mobilenetv2_model.keras"))
    models["densenet"] = load_model(os.path.join(BASE_DIR, "saved_models/grape_densenet121_model.keras"))
    models["efficientnet"] = load_model(os.path.join(BASE_DIR, "saved_models/grape_efficientnet_model.keras"))

    return models