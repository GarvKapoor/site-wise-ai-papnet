# SiteWise AI

A Flask web interface for the EuroSAT satellite land-classification + site-suitability and multi-criteria site recommendation Deep Learning project.

---

## 1. Supported Model Architectures

SiteWise AI supports two primary Land Classification backbones (Model 1):
1. **PAPNet (Pattern-Affinitive Propagation Network)** *(New)*:
   - Uses non-local spatial pattern affinity matrix computations ($A = \text{Softmax}(Q K^T / \sqrt{d})$) and affinity propagation ($P = A \cdot V$) to capture disjoint structural features (canopy textures, agricultural crop geometries, river curves, and road networks).
   - Located in [`training_papnet/`](training_papnet/).
2. **EfficientNetB0**:
   - Pretrained ImageNet transfer learning CNN backbone.
   - Located in [`training/`](training/).

---

## 2. Folder Structure

```text
SiteWise-AI/
├── app.py                         # Flask Web Backend (auto-detects PAPNet or EfficientNet)
├── requirements.txt               # Dependencies
├── templates/
│   └── index.html                 # Frontend Web Interface
├── static/
│   ├── style.css                  # Styling
│   └── script.js                  # Frontend interactions
├── models/                        # Trained model artifacts
│   ├── model1_papnet.keras        # PAPNet Land Classifier
│   ├── model1_efficientnet.keras  # EfficientNetB0 Land Classifier
│   ├── model2_suitability.keras   # Model 2: Site Suitability Predictor
│   ├── model2_scaler.pkl          # Model 2 Scaler
│   ├── model3_recommender.keras   # Model 3: Site Recommender (8 profiles)
│   └── model3_scaler.pkl          # Model 3 Scaler
├── training/                      # Original EfficientNet training notebook & artifacts
│   ├── project_notebook.ipynb
│   └── ...
└── training_papnet/               # PAPNet training folder
    ├── papnet_model.py            # Keras Pattern Affinity Layers & Model Builders
    ├── project_notebook_papnet.ipynb # Complete end-to-end PAPNet Jupyter notebook
    ├── train_papnet.py            # Standalone CLI training script
    ├── model1_papnet.keras        # Trained PAPNet model
    ├── model2_suitability.keras   # Trained Model 2
    ├── model3_recommender.keras   # Trained Model 3
    └── README.md                  # Detailed PAPNet documentation
```

---

## 3. Installation

```bash
python -m venv venv
```

Activate:

- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows:
  ```bash
  venv\Scripts\activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Training PAPNet

You can train PAPNet via either the CLI script or the interactive Jupyter notebook:

### CLI Script:
```bash
# Full training on EuroSAT
python training_papnet/train_papnet.py --epochs 20 --batch-size 32

# Quick verification demo
python training_papnet/train_papnet.py --quick-demo --epochs 2
```

### Jupyter Notebook:
Open and run [`training_papnet/project_notebook_papnet.ipynb`](training_papnet/project_notebook_papnet.ipynb) in Jupyter Lab, VS Code, or Google Colab.

---

## 5. Running the Web Application

To run with PAPNet (default when `models/model1_papnet.keras` exists):
```bash
python app.py
```

To explicitly force a specific architecture:
```bash
# Run with PAPNet
MODEL1_TYPE=papnet python app.py

# Run with EfficientNet
MODEL1_TYPE=efficientnet python app.py
```

Open:
[http://127.0.0.1:5000](http://127.0.0.1:5000)
