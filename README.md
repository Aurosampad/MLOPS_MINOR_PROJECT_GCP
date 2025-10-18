## 🧠 Fault Detection API — End-to-End MLOps Project

### 🚀 Overview

The **Fault Detection API** is a scalable and production-ready machine learning system built to detect HVAC air handling unit (AHU) faults in real-time.
The pipeline automates every stage — from data ingestion and model training to containerization, continuous integration & deployment (CI/CD), monitoring, and auto-scaling orchestration.

---

### 🏗️ Project Architecture

```
📦 Fault_Detection_API
├── app/
│   ├── server.py                 # FastAPI backend serving predictions
│   ├── __init__.py
│   └── model/                    # Trained models & scalers
│       ├── fault_rf_model.pkl
│       └── scaler.pkl
├── client.py                     # Client to test the API
├── train_model.py                # Model training script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Local development Docker build
├── Dockerfile.prod               # Optimized production Docker build
├── dvc.yaml                      # DVC pipeline for data & model tracking
├── .github/workflows/ci-cd.yml   # GitHub Actions CI/CD pipeline
└── README.md
```

---

### ⚙️ Features

✅ **End-to-End MLOps Integration**

* Fully automated ML lifecycle — data versioning, model training, evaluation, and deployment.

✅ **FastAPI REST Service**

* Exposes `/predict` endpoint for real-time HVAC fault prediction.
* Returns both class label and fault probability.

✅ **Containerized via Docker**

* Ensures reproducibility across environments using lightweight Python 3.12.1 base.
* Separate Dockerfiles for development and production.

✅ **Deployed on Google Cloud Platform**

* Flask/FastAPI container deployed on **GCP Cloud Run** or **GKE (Kubernetes)** via GitHub Actions CI/CD pipeline.

✅ **Continuous Integration & Deployment**

* **GitHub Actions** automates:

  * Docker image build & push to GCP Artifact Registry
  * Kubernetes deployment updates on merge to main branch
  * MLflow tracking server health verification

✅ **MLflow Tracking**

* Centralized tracking of:

  * Experiment metrics (accuracy, F1-score, confusion matrix)
  * Model artifacts (Random Forest weights, scaler)
  * Versioned runs for performance comparison
* MLflow UI hosted on GCP VM instance for experiment visualization.

✅ **Kubernetes Orchestration**

* Containerized API automatically scales pods based on incoming traffic.
* Uses **Horizontal Pod Autoscaler (HPA)** and **LoadBalancer Service** for seamless high-availability scaling.

✅ **DVC (Data Version Control) Integration**

* Version-controlled datasets and trained models.
* Remote storage configured on **Google Cloud Storage (GCS)**.
* DVC pipeline automates retraining when new data is pushed.

✅ **Monitoring & Logging**

* **Prometheus** + **Grafana** dashboards for API latency, request counts, and CPU/memory metrics.
* **MLflow + DVC hooks** for retriggering pipelines on model drift or new data.

---

### 🧩 Tech Stack

| Category             | Tools/Frameworks                      |
| -------------------- | ------------------------------------- |
| **Language**         | Python 3.12.1                         |
| **Modeling**         | scikit-learn (RandomForestClassifier) |
| **API Framework**    | FastAPI / Flask                       |
| **Containerization** | Docker, Docker Compose                |
| **Cloud**            | Google Cloud Platform (GCP)           |
| **CI/CD**            | GitHub Actions, GCP Cloud Build       |
| **Tracking**         | MLflow                                |
| **Data Versioning**  | DVC (Google Cloud Storage remote)     |
| **Orchestration**    | Kubernetes (GKE)                      |
| **Monitoring**       | Prometheus, Grafana                   |

---

### 🧠 Model Workflow

1. **Data Preparation**

   * Cleans and preprocesses AHU operational data.
   * Feature engineering using temperature & pressure residuals.

2. **Model Training**

   * Random Forest Classifier trained and validated.
   * Artifacts (`fault_rf_model.pkl`, `scaler.pkl`) exported for inference.

3. **Model Serving**

   * FastAPI app serves predictions via REST endpoint.
   * Input: JSON payload of AHU sensor readings.
   * Output: Fault classification & probability.

4. **Versioning**

   * DVC tracks data, models, and metrics.
   * MLflow tracks experiments & versions.

5. **Deployment**

   * CI/CD builds Docker image and deploys it to Kubernetes cluster.

6. **Auto-Scaling**

   * Kubernetes monitors CPU/memory usage and scales pods automatically.

---

### 🧪 API Usage Example

**POST Request**

```bash
curl -X POST "http://<GCP_DEPLOYMENT_URL>/predict" \
-H "Content-Type: application/json" \
-d '{
  "AHU_Supply_Air_Temperature": 77.13,
  "AHU_Outdoor_Air_Temperature": 80.61,
  "AHU_Mixed_Air_Temperature": 75.86,
  "AHU_Return_Air_Temperature": 75.19,
  "AHU_Supply_Air_Fan_Status": 1,
  "AHU_Return_Air_Fan_Status": 0,
  "AHU_Supply_Air_Fan_Speed_Control_Signal": 1,
  "AHU_Return_Air_Fan_Speed_Control_Signal": 0,
  "AHU_Exhaust_Air_Damper_Control_Signal": 0,
  "AHU_Outdoor_Air_Damper_Control_Signal": 0,
  "AHU_Return_Air_Damper_Control_Signal": 1,
  "AHU_Cooling_Coil_Valve_Control_Signal": 0,
  "AHU_Heating_Coil_Valve_Control_Signal": 0,
  "AHU_Supply_Air_Duct_Static_Pressure": 0.06,
  "Occupancy_Mode_Indicator": 0,
  "TempResidual": 1.14,
  "PressResidual": -0.91
}'
```

**Response**

```json
{
  "prediction": "1",
  "probability_of_fault": 0.87
}
```

---

### ⚙️ Running Locally

```bash
# Build Docker image
docker build -t fault-detection-api .

# Run container
docker run -p 8000:8000 fault-detection-api

# Access locally
http://127.0.0.1:8000
```

---

### 🌩️ Deployment via CI/CD (GitHub Actions → GCP)

* On push to `main`:

  1. Build and test Docker image.
  2. Push image to GCP Artifact Registry.
  3. Deploy latest container to GKE cluster.
  4. Run smoke tests for `/predict` endpoint.
  5. Update MLflow tracking entry for deployment version.

---

### 🧬 DVC & MLflow Workflow

```bash
# Initialize DVC
dvc init
dvc remote add -d gcsremote gs://<your-bucket-name>

# Track dataset & model
dvc add data/raw/MZVAV-2-2.csv
dvc add fault_rf_model.pkl
git add data/.gitignore fault_rf_model.pkl.dvc
git commit -m "Track data & model versions with DVC"

# Push data to GCS
dvc push
```

In MLflow UI (deployed on GCP):

* Track experiment name: `fault_detection_rf`
* Compare accuracy, F1-score, and confusion matrix across runs.

---

### ⚡ Scalability via Kubernetes (GKE)

Kubernetes configuration (`k8s-deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fault-detection-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fault-detection-api
  template:
    metadata:
      labels:
        app: fault-detection-api
    spec:
      containers:
      - name: fault-detection-api
        image: gcr.io/<your-project-id>/fault-detection-api:latest
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: fault-detection-service
spec:
  type: LoadBalancer
  selector:
    app: fault-detection-api
  ports:
  - port: 80
    targetPort: 8000
```

The service is exposed via a public load balancer with automatic scaling managed by HPA.

---

### 🧾 License

MIT License © 2025 Aurosampad Mohanty

---

