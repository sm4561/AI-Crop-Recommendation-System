from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import requests

# --- Initialize App (THIS LINE IS UPDATED) ---
app = Flask(__name__) # Flask now automatically finds the 'templates' folder

CORS(app)

# --- Load Model and Data ---
try:
    model = joblib.load('crop_model.joblib')
    crop_data_df = pd.read_csv('Crop_recommendation.csv')
    print("✅ Model and crop data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or data: {e}")
    model = None
    crop_data_df = None


# --- Homepage Route ---
@app.route('/')
def home():
    # This will now correctly serve index.html from inside the 'templates' folder
    return render_template('index.html')


# --- Analysis and Tips Function ---
def get_analysis_and_tips(input_data, recommended_crop, df):
    try:
        ideal_conditions = df[df['label'] == recommended_crop].iloc[0]

        analysis_results = []
        params = {
            'N': {'name': 'Nitrogen', 'threshold': 10},
            'P': {'name': 'Phosphorus', 'threshold': 5},
            'K': {'name': 'Potassium', 'threshold': 5},
            'ph': {'name': 'Soil pH', 'threshold': 0.5},
        }

        for key, details in params.items():
            user_val = input_data[key]
            ideal_val = ideal_conditions[key]
            threshold = details['threshold']
            status = 'good'

            if user_val < ideal_val - threshold:
                status = 'low'
            elif user_val > ideal_val + threshold:
                status = 'high'

            analysis_results.append({
                "parameter": details['name'],
                "user_value": f"{user_val:.1f}",
                "ideal_value": f"~{ideal_val:.1f}",
                "status": status
            })

        fertilizer_tip = (
            f"For {recommended_crop}, if Nitrogen is low, consider Urea. "
            f"If Phosphorus is low, DAP is a good option. "
            f"Always test your soil for precise nutrient management."
        )
        irrigation_tip = (
            f"{recommended_crop.capitalize()} requires about "
            f"{ideal_conditions['rainfall']:.0f} mm of water over its growing season. "
            f"Adjust your irrigation schedule based on recent rainfall and soil moisture."
        )

        return {
            "analysis": analysis_results,
            "fertilizer": fertilizer_tip,
            "irrigation": irrigation_tip
        }
    except Exception as e:
        print(f"❌ Error in tip generation: {e}")
        return {
            "analysis": [],
            "fertilizer": "Use a balanced NPK fertilizer after soil testing.",
            "irrigation": "Ensure proper irrigation as per crop needs."
        }


# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or crop_data_df is None:
        return jsonify({"error": "Model or data not loaded"}), 500

    data = request.json
    try:
        features = np.array([[
            data['N'], data['P'], data['K'], data['temperature'],
            data['humidity'], data['ph'], data['rainfall']
        ]])

        probabilities = model.predict_proba(features)[0]
        crop_names = model.classes_

        recommendations = [{"name": crop_names[i], "suitability": round(prob * 100, 2)}
                           for i, prob in enumerate(probabilities)]
        recommendations.sort(key=lambda x: x['suitability'], reverse=True)

        top_4 = recommendations[:4]
        top_crop_name = top_4[0]['name']

        tips = get_analysis_and_tips(data, top_crop_name, crop_data_df)

        response = {
            "top_crop": top_4[0],
            "other_crops": top_4[1:],
            "tips": tips
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Failed to predict"}), 400


@app.route('/get_crop_names', methods=['GET'])
def get_crop_names():
    if crop_data_df is not None:
        crop_names = sorted(crop_data_df['label'].unique().tolist())
        return jsonify({"crop_names": crop_names})
    return jsonify({"error": "Crop data not available"}), 500


@app.route('/get_crop_info', methods=['POST'])
def get_crop_info():
    if crop_data_df is None:
        return jsonify({"error": "Crop data not loaded"}), 500

    data = request.json
    crop_name = data.get('crop_name')
    if not crop_name:
        return jsonify({"error": "Crop name not provided"}), 400

    try:
        crop_info = crop_data_df[crop_data_df['label'] == crop_name].mean(numeric_only=True)

        if crop_info.empty:
            return jsonify({"error": "Crop not found in dataset"}), 404

        guide = {
            "idealNitrogen": f"{crop_info['N']:.0f}",
            "idealPhosphorus": f"{crop_info['P']:.0f}",
            "idealPotassium": f"{crop_info['K']:.0f}",
            "idealPh": f"{crop_info['ph']:.1f}",
            "idealTemp": f"{crop_info['temperature']:.1f}",
            "idealHumidity": f"{crop_info['humidity']:.0f}",
            "idealRainfall": f"{crop_info['rainfall']:.0f}"
        }

        return jsonify({"crop_name": crop_name, "guide": guide})

    except Exception as e:
        print(f"❌ Error getting crop info: {e}")
        return jsonify({"error": "Could not retrieve crop info"}), 500


# --- NEW SECURE WEATHER ROUTE ---
@app.route('/get_weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    api_key = os.environ.get('WEATHER_API_KEY')
    if not api_key:
        print("❌ WEATHER_API_KEY environment variable not set!")
        return jsonify({"error": "Weather API key not configured on server"}), 500
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for 4XX/5XX errors
        weather_data = response.json()
        return jsonify(weather_data)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching weather data: {e}")
        return jsonify({"error": "Failed to fetch weather data"}), 500


# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

