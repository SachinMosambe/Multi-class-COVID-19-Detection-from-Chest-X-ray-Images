# 🩺 Multi-Class COVID-19 Detection from Chest X-Ray Images

## 🧭 Overview

This project builds an **AI-driven medical imaging system** to automatically classify **Chest X-ray images** into three categories:

* **COVID-19**
* **Viral Pneumonia**
* **Normal**

The system leverages **deep learning (CNN + Transfer Learning)** and provides an **end-to-end workflow** — from **data preprocessing**, **model training**, and **evaluation** to **deployment via API or Streamlit**.

---

## 🧠 Skills Acquired

* Medical Image Preprocessing & Data Augmentation
* Deep Learning using CNN Architectures
* Transfer Learning (ResNet-50, VGG-16, DenseNet)
* Multi-class Classification
* Model Evaluation (ROC-AUC, F1-Score, Confusion Matrix)
* Explainable AI (Grad-CAM visualizations)
* Deployment using **Flask**, **Streamlit**, or **Docker**

---

## 🏥 Domain

**Healthcare / Medical Imaging / AI for Diagnostics**

---

## 📌 Problem Statement

Rapid and accurate diagnosis of COVID-19 using chest X-rays can support hospitals and remote healthcare systems.
The objective is to build a **multi-class classification model** that distinguishes between **COVID-19**, **Viral Pneumonia**, and **Normal** X-ray images with high accuracy and explainability.

---

## 💼 Business Use Cases

| Use Case                    | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| **Clinical Support**        | Helps radiologists with fast triage decisions        |
| **Remote Healthcare**       | Enables diagnostic assistance in low-resource areas  |
| **Public Health Screening** | Automates large-scale population screening           |
| **Training Tools**          | Educational use for medical students & professionals |

---

## ⚙️ Project Workflow

### **1. Data Exploration & Preprocessing**

* Dataset: *COVID-19 Chest X-ray Dataset* (Pranav Raikokte, Kaggle)
* Classes: `COVID-19`, `Viral Pneumonia`, `Normal`
* Techniques:

  * Image resizing & normalization
  * Class balancing with data augmentation (rotation, flip, brightness jitter)
  * Directory structure split into train/test folders

### **2. Model Development**

* Baseline CNN model for reference
* Transfer Learning models fine-tuned:

  * `ResNet-50`
  * `VGG-16`
  * `DenseNet-121`
* Loss Function: `Categorical Cross-Entropy`
* Optimizer: `Adam`
* Frameworks: TensorFlow / Keras

### **3. Model Evaluation**

| Metric                        | Purpose                              |
| ----------------------------- | ------------------------------------ |
| Accuracy                      | Overall performance                  |
| Precision / Recall / F1-Score | Class-specific performance           |
| ROC-AUC                       | Model discrimination ability         |
| Confusion Matrix              | Misclassification insights           |
| Grad-CAM                      | Visual explainability of predictions |

### **4. Model Fine-Tuning**

* Layer freezing/unfreezing for pre-trained networks
* Learning rate scheduling and dropout regularization
* Comparative analysis of pretrained models (logged in Jupyter)

### **5. Deployment**

* API endpoint using **Flask (`api.py`)** for image inference
* Interactive **Streamlit web app (`app.py`)** for end-users
* Hosted on **AWS EC2**, integrated with **S3 bucket** for model storage
* Optional containerization via **Docker**

---

## 🧩 Project Structure

```
├── .github/workflows/deploy.yml       # CI/CD pipeline to AWS EC2
├── Covid19-dataset/                   # Dataset folder (train/test images)
├── output/                            # Model outputs & results
├── .gitignore
├── Pretrained_model_comparison.ipynb  # Comparison of transfer learning models
├── finetuning_best_model.ipynb        # Fine-tuning and optimization
├── app.py                             # Streamlit web interface
├── api.py                             # Flask API for model inference
├── requirements.txt                   # Dependencies
└── README.md                          # Project documentation
```

---

## 🧱 Deployment (AWS EC2 + S3 + GitHub Actions)

The workflow automatically deploys updates from GitHub to EC2.

### **GitHub Actions Workflow (`deploy.yml`):**

```yaml
name: Deploy to EC2

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up SSH
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.EC2_KEY }}" > ~/.ssh/ec2_key.pem
          chmod 600 ~/.ssh/ec2_key.pem

      - name: Deploy to EC2
        run: |
          ssh -o StrictHostKeyChecking=no -i ~/.ssh/ec2_key.pem ${{ secrets.EC2_USER }}@${{ secrets.EC2_HOST }} << 'EOF'
            cd ~/Bank-Term-Deposit-Subscription-Prediction
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            aws s3 cp s3://${{ secrets.S3_BUCKET }}/models/model_for_inference models/model_for_inference --no-sign-request
            nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > app.log 2>&1 &
          EOF
```

---

## 📊 Expected Results

* ≥ 95% validation accuracy on test dataset
* High recall & F1-score for **COVID-19** class
* Grad-CAM visualizations showing lung opacity focus regions
* Fully functional web interface for real-time X-ray diagnosis

---

## 🧾 Evaluation Metrics

| Metric                  | Description                        |
| ----------------------- | ---------------------------------- |
| Accuracy                | Overall correctness of predictions |
| Precision / Recall / F1 | Clinical reliability               |
| ROC-AUC                 | Discrimination between classes     |
| Confusion Matrix        | Error distribution                 |
| Grad-CAM                | Model explainability visualization |

---

## 👨‍💻 Author

**Sachin Dattatraay Mosambe**
*AI | Deep Learning | Medical Imaging*
📧 [LinkedIn](https://www.linkedin.com/in/sachinmosambe)

---

## 🪪 License

Released under the **MIT License** — free to use, modify, and distribute with attribution.

---

⭐ **If you find this project useful, please give it a star on GitHub!**

