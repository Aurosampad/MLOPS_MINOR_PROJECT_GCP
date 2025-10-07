import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Load data ---
df = pd.read_csv(r'D:\MLOps\JRF_Screening_Test_Assignemnt\MLOPS_MINOR_PROJECT_GCP\artifacts\raw\MZVAV-2-2.csv')

# --- Drop unwanted column ---
df = df.drop(columns=['Datetime'])

# --- Clean column names (remove colons, spaces, double underscores, etc.) ---
df.columns = (
    df.columns.str.replace(":", "", regex=False)
               .str.replace(" ", "_")
               .str.replace("__", "_")
               .str.strip("_")  # remove leading/trailing underscores
)

# ✅ Print renamed columns to verify
print("✅ Renamed Columns:\n", df.columns.tolist())

# --- Fix possible column name variations ---
pressure_col = None
for col in df.columns:
    if "AHU_Supply_Air_Duct_Static_Pressure" in col and "Set_Point" not in col:
        pressure_col = col
        break
if pressure_col is None:
    raise KeyError("❌ Could not find a valid 'AHU_Supply_Air_Duct_Static_Pressure' column!")

# --- Feature Engineering ---
df["TempResidual"] = df["AHU_Supply_Air_Temperature"] - df["AHU_Supply_Air_Temperature_Set_Point"]
df["PressResidual"] = df[pressure_col] - df["AHU_Supply_Air_Duct_Static_Pressure_Set_Point"]

# --- Normalize residuals ---
scaler = StandardScaler()
df[["TempResidual", "PressResidual"]] = scaler.fit_transform(df[["TempResidual", "PressResidual"]])

# --- Drop setpoint columns ---
df = df.drop(columns=["AHU_Supply_Air_Temperature_Set_Point",
                      "AHU_Supply_Air_Duct_Static_Pressure_Set_Point"])

# --- Ensure consistent feature order ---
feature_cols = [
    'AHU_Supply_Air_Temperature',
    'AHU_Outdoor_Air_Temperature',
    'AHU_Mixed_Air_Temperature',
    'AHU_Return_Air_Temperature',
    'AHU_Supply_Air_Fan_Status',
    'AHU_Return_Air_Fan_Status',
    'AHU_Supply_Air_Fan_Speed_Control_Signal',
    'AHU_Return_Air_Fan_Speed_Control_Signal',
    'AHU_Exhaust_Air_Damper_Control_Signal',
    'AHU_Outdoor_Air_Damper_Control_Signal',
    'AHU_Return_Air_Damper_Control_Signal',
    'AHU_Cooling_Coil_Valve_Control_Signal',
    'AHU_Heating_Coil_Valve_Control_Signal',
    'AHU_Supply_Air_Duct_Static_Pressure',
    'Occupancy_Mode_Indicator',
    'TempResidual',
    'PressResidual'
]

X = df[feature_cols]
y = df["Fault_Detection_Ground_Truth"]

# --- Train-test split ---
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# --- Random Forest Classifier ---
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# --- Train the model ---
rf.fit(X_train_rf, y_train_rf)

# --- Predictions ---
y_pred_rf = rf.predict(X_test_rf)

# --- Evaluation ---
print("✅ Random Forest Classifier Results")
print("Accuracy:", accuracy_score(y_test_rf, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_rf, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test_rf, y_pred_rf))

# --- Save the trained model and scaler ---
model_path = r"artifacts/models/rf_model.pkl"
scaler_path = r"artifacts/models/scaler.pkl"

joblib.dump(rf, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Model saved successfully at: {model_path}")
print(f"✅ Scaler saved successfully at: {scaler_path}")



