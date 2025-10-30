import numpy as np
import joblib
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

T = 60
MODEL_PATH = "models/bilstm_attention.keras"
SCALER_PATH = "models/scaler_pipino.pkl"
ENCODER_PATH = "models/label_encoder_pipino.pkl"

@tf.keras.utils.register_keras_serializable()
class TFLiteAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TFLiteAttentionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(1, use_bias=True, activation='tanh')
        super(TFLiteAttentionPooling, self).build(input_shape)

    def call(self, x):
        e = self.dense(x)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "TFLiteAttentionPooling": TFLiteAttentionPooling,
        "AttentionPooling": TFLiteAttentionPooling
    }
)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("✅ Model, scaler, and label encoder loaded successfully!")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected ✅")

    try:
        while True:
            data = await ws.receive_json()

            if data.get("type") == "frame_sequence":
                seq = np.array(data["landmarks"], dtype=np.float32)

                if seq.shape[0] != T:
                    print(f"⚠️ Skipped: expected {T} frames, got {seq.shape[0]}")
                    continue

                X_input = seq.reshape(-1, seq.shape[-1])
                X_input = scaler.transform(X_input)
                X_input = X_input.reshape(1, T, -1)

                preds = model.predict(X_input, verbose=0)[0]
                pred_class = int(np.argmax(preds))
                pred_label = label_encoder.inverse_transform([pred_class])[0]
                confidence = float(preds[pred_class])

                if pred_label.lower() == "nothing":
                    continue  # skip sending "nothing"

                await ws.send_json({
                    "prediction": pred_label,
                    "confidence": confidence
                })

                print(f"Prediction sent: {pred_label} ({confidence:.2f})")

    except Exception as e:
        print("WebSocket disconnected ❌", e)
