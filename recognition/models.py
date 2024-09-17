import tensorflow as tf # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # type: ignore
import numpy as np # type: ignore

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

def predict_food(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]  # Returns top 3 predictions
