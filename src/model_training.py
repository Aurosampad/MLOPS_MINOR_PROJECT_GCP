import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger=get_logger(__name__)

class ModelTraining:
    def __init__(self):
        self.processed_data_path="artifacts/processed"
        self.model_path="artifacts/models"
        os.makedirs(self.model_path, exist_ok=True)
        # The model is consistently named self.model
        self.model = RandomForestClassifier( n_estimators=200,max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=42,n_jobs=-1)
        logger.info("RandomForestClassifier instance created")
        
    def load_data(self):
        try:
            X_train=joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            X_test=joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            y_train=joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            y_test=joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            
            logger.info("Processed data loaded successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise CustomException("Error loading processed data", e)
        
    def train_model(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, os.path.join(self.model_path, "rf_model.pkl"))
            logger.info("Model training completed and saved successfully...")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Error during model training", e)
            
    def evaluate_model(self,X_test,y_test):
        try:
            # FIX #2: Use self.model for predictions, not self.rf_model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            cm = confusion_matrix(y_test,y_pred)
            # Handle cases for binary classification where average might be needed
            precision = precision_score(y_test,y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'macro')
            recall = recall_score(y_test,y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'macro')
            f1 = f1_score(y_test,y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'macro')

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")

            plt.figure(figsize=(8,6))
            # FIX #1: Corrected parameter names xticklabels and yticklabels
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            confusion_matrix_path=os.path.join(self.model_path, "confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close()

            logger.info(f"Confusion matrix saved at {confusion_matrix_path}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Error during model evaluation", e)
        
    def run(self):
        try:
            X_train, X_test, y_train, y_test = self.load_data()
            self.train_model(X_train, y_train)
            # This is simpler and more correct. No need to reload the model from the file.
            self.evaluate_model(X_test, y_test)
        except Exception as e:
            logger.error(f"Error in ModelTraining run method: {e}")
            raise CustomException("Error in ModelTraining run method", e)
        
if __name__=="__main__":
    trainer = ModelTraining()
    trainer.run()