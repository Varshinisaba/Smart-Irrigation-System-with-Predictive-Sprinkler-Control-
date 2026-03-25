# 🌱 Smart Irrigation System using IoT, Edge Computing & Machine Learning

## 📌 Overview

This project presents an intelligent **Smart Irrigation System** that integrates **IoT, Edge Computing, Fog Computing, and Machine Learning** to optimize water usage in agriculture.

The system collects real-time environmental data using sensors and applies predictive analytics (LSTM model) to make efficient irrigation decisions, reducing water wastage and improving crop productivity.

---

## 🎯 Objectives

* Automate irrigation based on real-time environmental conditions
* Predict water requirements using machine learning
* Reduce water consumption and improve efficiency
* Implement a scalable architecture using Edge & Fog computing

---

## ⚙️ Tech Stack

* **Hardware:** ESP32 (Edge Device)
* **Programming:** Python
* **Machine Learning:** LSTM (Long Short-Term Memory)
* **Communication:** LoRa Simulation
* **Architecture:** Edge → Fog → Cloud
* **Data Processing:** Simulation & Real Data Adapters

---

## 🧠 Key Features

* 📡 Real-time data collection using IoT sensors
* 🤖 LSTM-based prediction for irrigation needs
* 🌐 Fog computing for task scheduling and processing
* 📊 Simulation of environmental datasets
* 📉 Performance evaluation using SDG-based metrics
* 🔗 Integration of edge, fog, and simulation layers

---

## 🏗️ System Architecture

The system follows a **multi-layer architecture**:

* **Edge Layer:**
  ESP32 collects sensor data (temperature, humidity, soil moisture)

* **Fog Layer:**
  Processes data and schedules irrigation tasks efficiently

* **Cloud/Simulation Layer:**
  Runs machine learning models and evaluates system performance

---

## 📂 Project Structure

```
smartIrrigation/
│
├── edge/                # ESP32 code & edge ML logic
├── fog/                 # Fog scheduler and processing
├── simulation/          # Dataset generation & adapters
├── lora_sim/            # LoRa communication simulation
├── evaluation/          # SDG metrics evaluation
│
├── run_simulation.py    # Main execution file
├── ml_evaluation.py     # ML model evaluation
├── fix_lstm.py          # LSTM model fixes
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/Varshinisaba/smartIrrigation.git
cd smartIrrigation
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

*(If requirements.txt is not present, install necessary libraries like numpy, pandas, tensorflow, etc.)*

### 3. Run the simulation

```
python run_simulation.py
```

---

## 📊 Output & Results

* Efficient irrigation scheduling
* Reduced water usage
* Improved prediction accuracy using LSTM
* Performance evaluated using sustainability metrics

---

## 🌍 Applications

* Smart agriculture systems
* Precision farming
* Water resource management
* IoT-based environmental monitoring

---

## 🚀 Future Enhancements

* Integration with real-time cloud platforms
* Mobile/web dashboard for monitoring
* Deployment with real sensors and actuators
* Advanced ML models for higher accuracy

---



