from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("lstm_demand_forecasting.h5")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Demand Forecasting API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, 10, 8)  # Ensure correct shape
        
        prediction = model.predict(features)
        predicted_value = float(prediction[0][0])

        return jsonify({"predicted_sales": predicted_value})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is working fine!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)