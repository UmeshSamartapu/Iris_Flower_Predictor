# Iris Flower Predictor

[![Preview image](https://via.placeholder.com/800x400.png?text=Add+Screenshot+Here)](https://example.com)

A machine learning web application for classifying Iris flower species using Random Forest and FastAPI.

## Features

- 📊 Form-based input for flower measurements
- 🔮 Prediction of Iris species (Setosa, Versicolor, Virginica)
- 🚀 FastAPI backend for efficient predictions
- 💻 Responsive web interface
- ☁️ Automatic model download from Hugging Face
- 📱 Mobile-friendly design

## Technologies Used

- **Backend:** FastAPI, Python
- **ML Framework:** scikit-learn, joblib
- **Frontend:** HTML5, CSS3, Jinja2 templating
- **Deployment:** Render (via render.yaml)

## Getting Started 

### 1. Clone the repository:
```bash
git clone https://github.com/UmeshSamartapu/smart-iris-predictor.git
cd smart-iris-predictor
```

### 2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Access the web interface at:
```bash
http://localhost:8000
```
#### Enter flower measurements to get species predictions

## Project Structure

```bash
smart-iris-predictor/
├── ModelMaking/
│   └── ModelMaking.py    # Model training code
├── templates/            # HTML templates
│   └── index.html
├── static/               # Static assets
│   ├── uploads/          # (Reserved for future use)
│   └── results/          # (Reserved for future use)
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
├── render.yaml           # Deployment configuration
└── rf_model.pkl          # Pretrained Random Forest model
```

### License
```bash

```

### Acknowledgments

- scikit-learn for machine learning framework

- Hugging Face for model hosting

- FastAPI documentation and community

- **Maintainer:** Umesh Samartapu



## Demo 
### You can watch the ([youtube video](    )) for demo
<p align="center">
  <img src=" " />
</p>



## 📫 Let's Connect

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/umeshsamartapu/)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://x.com/umeshsamartapu)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:umeshsamartapu@gmail.com)
[![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/umeshsamartapu/)
[![Buy Me a Coffee](https://img.shields.io/badge/-Buy%20Me%20a%20Coffee-FBAD19?style=flat-square&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/umeshsamartapu)

---

🔥 Always exploring new technologies and solving real-world problems with code!
