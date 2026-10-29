# 🍎 Pomegranate Disease & Edibility AI Detector

An end-to-end Deep Learning & Machine Learning web application that analyzes pomegranate images in real-time, detects plant diseases, predicts confidence levels, and determines edibility and health risk warnings.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C?logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3-38B2AC?logo=tailwindcss)
![Netlify](https://img.shields.io/badge/Frontend-Netlify%20Ready-00C7B7?logo=netlify)
![Render](https://img.shields.io/badge/Backend-Render%20Ready-46E3B7?logo=render)

---

## 🌟 Key Features

- **Transfer Learning with PyTorch**: Uses pre-trained MobileNetV2 as a deep feature extractor for high-dimensional botanical pattern recognition.
- **Support Vector Classifier (SVC)**: Trained model for swift inference and disease classification with probability estimation.
- **5 Pathology Categories**:
  - `Healthy` (Safe for consumption, high in antioxidants)
  - `Alternaria` (Heart rot / fungal decay inside fruit)
  - `Anthracnose` (Dark sunken lesions, rotting pulp)
  - `Bacterial Blight` (Oily spots with distinctive rind cracking)
  - `Cercospora` (Dark brown spotting on outer rind)
- **Dietary Safety & Health Guidance**: Detailed toxicity analysis, edible vs. non-edible advice, and health precautions for each condition.
- **Modern Glassmorphic Web UI**: Responsive drag-and-drop file upload, live image preview, animated confidence meter, and diagnosis report card.
- **Dual Hosting Architecture**:
  - Deploy **Frontend on Netlify** (ultra-fast CDN).
  - Deploy **Backend on Render / Railway / Koyeb** (free cloud Python environment with PyTorch support).
  - Or run the full application together on **Render** or locally on **Flask**.

---

## 📁 Repository Structure

```text
├── app.py                 # Flask REST API with CORS & PyTorch inference
├── disease_info.py        # Pathological descriptions & health toxicity details
├── best_model.pkl         # Trained Scikit-Learn Support Vector Classifier
├── class_names.pkl        # Serialized class labels list
├── train_models.py        # Model training and feature extraction pipeline
├── requirements.txt       # Production Python dependencies
├── Procfile               # Gunicorn process definition for cloud hosting
├── render.yaml            # Render Blueprint for 1-click cloud deployment
├── netlify.toml           # Netlify publish and redirect configuration
├── .gitignore             # Ignored environments, cache, and temp files
├── templates/
│   └── index.html         # Full web frontend template
└── static_site/           # Standalone static site folder ready for Netlify Drop
    └── index.html
```

---

## 🚀 Step 1: Run Locally

### 1. Clone or Download Repository
```bash
git clone https://github.com/YOUR_USERNAME/Pomegranate-Disease-Detector.git
cd Pomegranate-Disease-Detector
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for CPU-only systems (faster & lighter installation):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4. Launch Application
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 🐙 Step 2: Push to GitHub

### Method A: Using Git CLI (Recommended)

1. Open a terminal in this folder:
```bash
git init
git add .
git commit -m "Initial commit: Pomegranate Disease Detector AI"
```

2. Create a new repository on [GitHub](https://github.com/new) named `Pomegranate-Disease-Detector`. Do **not** initialize with README (we already have one).

3. Link and push to GitHub:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Pomegranate-Disease-Detector.git
git push -u origin main
```

### Method B: Upload via Web Browser
1. Go to [github.com/new](https://github.com/new) and create a repository.
2. Click **uploading an existing file**.
3. Drag and drop all files from this folder (excluding `venv/` and `__pycache__/`).
4. Click **Commit changes**.

---

## 🌐 Step 3: Hosting on Netlify & Free Cloud Backend

### Why is a backend service required?
Netlify is an ultra-fast host for static HTML/CSS/JavaScript websites. However, the disease detection requires **PyTorch** and machine learning inference, which run in Python. 

The industry standard solution:
1. **Deploy Python Backend** on **Render.com** (100% Free).
2. **Deploy Frontend** on **Netlify** (100% Free) and connect it to your Render backend URL.
3. *(Alternative)* Deploy the whole app directly to **Render** in 1 click!

---

### Step-by-Step Deployment Instructions

#### 1. Deploy the Backend to Render (Free)
1. Go to **[Render.com](https://render.com)** and sign in with GitHub.
2. Click **New +** > **Web Service**.
3. Select your `Pomegranate-Disease-Detector` GitHub repository.
4. Fill in these settings:
   - **Name**: `pomegranate-disease-ai`
   - **Language**: `Python 3`
   - **Region**: Closest to you (e.g. Frankfurt, Oregon, Singapore)
   - **Branch**: `main`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`
5. Click **Create Web Service**.
6. When the build finishes, copy your live Render URL (e.g., `https://pomegranate-disease-ai.onrender.com`).

---

#### 2. Deploy the Frontend to Netlify (Free)

#### Option A: 1-Click Drag-and-Drop (Netlify Drop)
1. Log into **[Netlify](https://app.netlify.com/)**.
2. Go to **Sites** > drag and drop the `static_site` folder onto the page.
3. Netlify will deploy your site in 5 seconds and give you a URL like `https://funny-pomegranate-12345.netlify.app`.
4. Open the deployed Netlify URL in your browser.
5. Click the **API Settings** button (gear icon in the top right).
6. Paste your Render backend URL (e.g. `https://pomegranate-disease-ai.onrender.com`) and click **Save Settings**.
7. Done! Your site is live and running with full AI inference.

#### Option B: Connect Netlify to GitHub
1. In Netlify dashboard, click **Add new site** > **Import an existing project**.
2. Choose **GitHub** and authorize your repository.
3. Set:
   - **Publish directory**: `templates` (or `static_site`)
4. Click **Deploy site**.
5. Once live, open your site, click **API Settings** at the top right, and enter your Render URL.

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description | Sample Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web application user interface | HTML document |
| `GET` | `/health` | Health check and model status | `{"status": "healthy", "model_loaded": true}` |
| `POST` | `/predict` | Multipart image upload for AI diagnosis | `{"class": "Healthy", "confidence": "98.5%", "edible": true, ...}` |

---

## 🧪 Training Your Own Model

If you have a dataset of pomegranate disease images:
1. Place disease image folders inside `./dataset/<Disease_Name>/*.jpg`.
2. Run the training pipeline:
```bash
python train_models.py
```
3. The script evaluates 5 classifiers (SVC, Random Forest, Logistic Regression, KNN, MLP) and automatically saves the best performing model to `best_model.pkl`.

---

## 📄 License
This project is open-source and available under the MIT License.
