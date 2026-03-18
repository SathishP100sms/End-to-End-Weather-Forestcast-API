# 🌦️ AI Probabilistic Weather Forecast

An AI-powered probabilistic weather forecasting system that predicts current weather and a 7-day forecast with uncertainty bands (10%–90%) using deep learning.

## 🚀 Live Demo
Frontend: https://your-netlify-link.netlify.app  
Backend: https://your-render-link.onrender.com  

## ✨ Features
- Real-time weather data
- 7-day probabilistic forecast
- Confidence intervals (10%–90%)
- Interactive charts (Chart.js)
- Dark/Light mode UI

## 🧠 Model
- Attention-based LSTM
- Time series forecasting
- Built using TensorFlow/Keras

## 🛠️ Tech Stack
Frontend: HTML, CSS, JavaScript  
Backend: FastAPI  
ML: TensorFlow, Keras  
Deployment: Netlify, Render  

## 📂 Project Structure
ai-weather-forecast/
│
├── backend/
├── frontend/
├── requirements.txt
└── README.md

## ⚙️ Setup

### Clone Repo
git clone https://github.com/your-username/ai-weather-forecast.git

### Backend
pip install -r requirements.txt
uvicorn main:app --reload

### Frontend
Open index.html

## 🌍 Deployment
- Backend → Render
- Frontend → Netlify

## 🔐 Env
OPENWEATHER_API_KEY=your_key

## 📊 Example API
{
  "city": "London",
  "forecast": [
    {"median_50": 23, "lower_10": 20, "upper_90": 26}
  ]
}

## 🎯 Use Cases
- ML portfolio
- Weather prediction
- Time-series learning

## 📜 License
MIT
