import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

# Initialize logger
logger = get_logger(__name__)

class DataProcessing:
    """
    A class to handle loading, preprocessing, and splitting of the fault detection dataset.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        # Define paths for output artifacts
        self.processed_data_path = "artifacts/processed"
        self.model_files_path = "trained_model_files"
        self.scaler_path = os.path.join(self.model_files_path, "scaler.pkl")
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.model_files_path, exist_ok=True)
    
    def load_data(self):
        """Loads data from the specified CSV file path into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully from {self.file_path}")
        except Exception as e:
            logger.error(f"Error while reading data: {e}")
            raise CustomException("Error loading data", e)
            
    def preprocess_data(self):
        """
        Applies all preprocessing steps to the loaded DataFrame:
        1. Drops unnecessary columns.
        2. Cleans and standardizes column names.
        3. Performs feature engineering to create residuals.
        4. Scales the new residual features and saves the scaler object.
        5. Drops original setpoint columns.
        """
        if self.df is None:
            raise CustomException("Data not loaded. Please call load_data() first.", sys.exc_info())

        try:
            logger.info("Starting data preprocessing...")

            # --- Drop unwanted column ---
            self.df = self.df.drop(columns=['Datetime'])
            logger.info("Dropped 'Datetime' column.")

            # --- Clean column names ---
            self.df.columns = (
                self.df.columns.str.replace(":", "", regex=False)
                                .str.replace(" ", "_")
                                .str.replace("__", "_")
                                .str.strip("_")
            )
            logger.info("Cleaned and standardized column names.")

            # --- Find the correct pressure column dynamically ---
            pressure_col = None
            for col in self.df.columns:
                if "AHU_Supply_Air_Duct_Static_Pressure" in col and "Set_Point" not in col:
                    pressure_col = col
                    break
            if pressure_col is None:
                raise KeyError("Could not find a valid 'AHU_Supply_Air_Duct_Static_Pressure' column!")
            logger.info(f"Identified pressure column as: {pressure_col}")

            # --- Feature Engineering ---
            self.df["TempResidual"] = self.df["AHU_Supply_Air_Temperature"] - self.df["AHU_Supply_Air_Temperature_Set_Point"]
            self.df["PressResidual"] = self.df[pressure_col] - self.df["AHU_Supply_Air_Duct_Static_Pressure_Set_Point"]
            logger.info("Created 'TempResidual' and 'PressResidual' features.")

            # --- Normalize residuals and save the scaler ---
            scaler = StandardScaler()
            self.df[["TempResidual", "PressResidual"]] = scaler.fit_transform(self.df[["TempResidual", "PressResidual"]])
            joblib.dump(scaler, self.scaler_path)
            logger.info(f"Normalized residuals and saved scaler to {self.scaler_path}")

            # --- Drop setpoint columns ---
            self.df = self.df.drop(columns=[
                "AHU_Supply_Air_Temperature_Set_Point",
                "AHU_Supply_Air_Duct_Static_Pressure_Set_Point"
            ])
            logger.info("Dropped setpoint columns after creating residuals.")
            logger.info("✅ Data preprocessing complete.")
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise CustomException("Error in preprocess_data", e)

    def split_data(self, test_size=0.4, random_state=42):
        """
        Splits the preprocessed data into training and testing sets (X_train, X_test, y_train, y_test)
        and saves them as individual .pkl files.

        Returns:
            tuple: A tuple containing the file paths for the four saved artifacts.
                   (X_train_path, X_test_path, y_train_path, y_test_path)
        """
        try:
            logger.info("Splitting data into training and testing sets...")
            
            # --- Define feature and target columns ---
            feature_cols = [
                'AHU_Supply_Air_Temperature', 'AHU_Outdoor_Air_Temperature', 'AHU_Mixed_Air_Temperature',
                'AHU_Return_Air_Temperature', 'AHU_Supply_Air_Fan_Status', 'AHU_Return_Air_Fan_Status',
                'AHU_Supply_Air_Fan_Speed_Control_Signal', 'AHU_Return_Air_Fan_Speed_Control_Signal',
                'AHU_Exhaust_Air_Damper_Control_Signal', 'AHU_Outdoor_Air_Damper_Control_Signal',
                'AHU_Return_Air_Damper_Control_Signal', 'AHU_Cooling_Coil_Valve_Control_Signal',
                'AHU_Heating_Coil_Valve_Control_Signal', 'AHU_Supply_Air_Duct_Static_Pressure',
                'Occupancy_Mode_Indicator', 'TempResidual', 'PressResidual'
            ]
            target_col = "Fault_Detection_Ground_Truth"

            X = self.df[feature_cols]
            y = self.df[target_col]

            # --- Perform train-test split with stratification ---
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # --- Define paths for the .pkl files ---
            X_train_path = os.path.join(self.processed_data_path, "X_train.pkl")
            X_test_path = os.path.join(self.processed_data_path, "X_test.pkl")
            y_train_path = os.path.join(self.processed_data_path, "y_train.pkl")
            y_test_path = os.path.join(self.processed_data_path, "y_test.pkl")

            # --- Save each split component using joblib ---
            joblib.dump(X_train, X_train_path)
            joblib.dump(X_test, X_test_path)
            joblib.dump(y_train, y_train_path)
            joblib.dump(y_test, y_test_path)
            
            logger.info(f"X_train saved to {X_train_path}")
            logger.info(f"X_test saved to {X_test_path}")
            logger.info(f"y_train saved to {y_train_path}")
            logger.info(f"y_test saved to {y_test_path}")

            return X_train_path, X_test_path, y_train_path, y_test_path

        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise CustomException("Error in split_data", e)

    def run(self):
        """Executes the full data processing pipeline."""
        self.load_data()
        self.preprocess_data()
        return self.split_data()
    
if __name__=="__main__":
    data_processor = DataProcessing("artifacts/raw/MZVAV-2-2.csv")
    data_processor.run()