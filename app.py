from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)   # ✅ Corrected

CORS(app)

# --- Step 1: Load Model and Data at Startup ---
try:
    model = joblib.load('crop_model.joblib')
    crop_data_df = pd.read_csv('Crop_recommendation.csv')
    print("✅ Model and crop data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or data: {e}")
    model = None
    crop_data_df = None


# --- Step 2: UPDATED Analysis and Tips Function ---
def get_analysis_and_tips(input_data, recommended_crop, df):
    """Compares user input to ideal crop conditions and provides structured analysis."""
    try:
        ideal_conditions = df[df['label'] == recommended_crop].iloc[0]

        # Create a list of structured analysis results
        analysis_results = []

        # Define parameters and their optimal thresholds for comparison
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

        # General tips remain as strings
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
            "analysis": analysis_results,  # This is now a list of objects
            "fertilizer": fertilizer_tip,
            "irrigation": irrigation_tip
        }
    except Exception as e:
        print(f"❌ Error in tip generation: {e}")
        return {
            "analysis": [],
            "fertilizer": "A standard NPK fertilizer is recommended. Test your soil for specific needs.",
            "irrigation": "Ensure good soil moisture according to the crop's specific needs."
        }


# --- Step 3: API Endpoints ---
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

        # Get the new structured tips
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


# --- Step 4: Run the Flask App ---
if __name__ == '__main__':   # ✅ Corrected
    app.run(debug=True)
