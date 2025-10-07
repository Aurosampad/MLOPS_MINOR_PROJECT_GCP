from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# --- Load the trained model and the scaler ---
# The scaler is essential for correct predictions as it preprocesses the new data
# in the same way as the training data.
with open('artifacts/models/rf_model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('artifacts/models/scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)


@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, preprocesses data, makes a prediction, and returns the result."""
    try:
        # --- 1. Explicitly extract form values and cast to correct types ---
        # This is safer and more readable than list comprehensions.
        input_features = {
            'AHU_Supply_Air_Temperature': float(request.form['AHU_Supply_Air_Temperature']),
            'AHU_Outdoor_Air_Temperature': float(request.form['AHU_Outdoor_Air_Temperature']),
            'AHU_Mixed_Air_Temperature': float(request.form['AHU_Mixed_Air_Temperature']),
            'AHU_Return_Air_Temperature': float(request.form['AHU_Return_Air_Temperature']),
            'AHU_Supply_Air_Fan_Status': int(request.form['AHU_Supply_Air_Fan_Status']),
            'AHU_Return_Air_Fan_Status': int(request.form['AHU_Return_Air_Fan_Status']),
            'AHU_Supply_Air_Fan_Speed_Control_Signal': float(request.form['AHU_Supply_Air_Fan_Speed_Control_Signal']),
            'AHU_Return_Air_Fan_Speed_Control_Signal': float(request.form['AHU_Return_Air_Fan_Speed_Control_Signal']),
            'AHU_Exhaust_Air_Damper_Control_Signal': float(request.form['AHU_Exhaust_Air_Damper_Control_Signal']),
            'AHU_Outdoor_Air_Damper_Control_Signal': float(request.form['AHU_Outdoor_Air_Damper_Control_Signal']),
            'AHU_Return_Air_Damper_Control_Signal': float(request.form['AHU_Return_Air_Damper_Control_Signal']),
            'AHU_Cooling_Coil_Valve_Control_Signal': float(request.form['AHU_Cooling_Coil_Valve_Control_Signal']),
            'AHU_Heating_Coil_Valve_Control_Signal': float(request.form['AHU_Heating_Coil_Valve_Control_Signal']),
            'AHU_Supply_Air_Duct_Static_Pressure': float(request.form['AHU_Supply_Air_Duct_Static_Pressure']),
            'Occupancy_Mode_Indicator': int(request.form['Occupancy_Mode_Indicator']),
            'TempResidual': float(request.form['TempResidual']),
            'PressResidual': float(request.form['PressResidual'])
        }

        # --- 2. Create a DataFrame from the extracted data ---
        input_df = pd.DataFrame([input_features])
        
        # --- 3. CRITICAL: Apply the saved scaler to the residual features ---
        # The model was trained on scaled residuals, so prediction data must also be scaled.
        input_df[['TempResidual', 'PressResidual']] = scaler.transform(input_df[['TempResidual', 'PressResidual']])

        # --- 4. Make prediction and get probability ---
        prediction = model.predict(input_df)[0]
        # The probability of the "fault" class (which is class 1)
        probability = model.predict_proba(input_df)[0][1]

        # --- 5. Format the result and status for the front-end ---
        if prediction == 1:
            result = f"🚨 Fault Detected (Confidence: {round(probability * 100, 2)}%)"
            status = "danger" # For CSS styling (e.g., red background)
        else:
            # Calculate confidence for "No Fault" class
            no_fault_prob = 1 - probability
            result = f"✅ No Fault Detected (Confidence: {round(no_fault_prob * 100, 2)}%)"
            status = "success" # For CSS styling (e.g., green background)
        
        return render_template('index.html', prediction_result=result, status=status)

    except Exception as e:
        # --- 6. Gracefully handle any errors ---
        return render_template('index.html', prediction_result=f"Error: {e}", status="warning")


# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
