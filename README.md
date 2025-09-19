# ✈️ Flight Price & Customer Satisfaction Prediction 🌟

Welcome to the **Flight Price and Customer Satisfaction Prediction** repository!  
This project delivers **two end-to-end predictive solutions** in the **Travel** and **Customer Experience** domains:

1. **Flight Price Prediction (Regression)** 🏷️ → Estimate flight ticket prices based on travel details.  
2. **Customer Satisfaction Prediction (Classification)** 💬 → Predict passenger satisfaction using demographic and service data.  

Both projects are built with **Python**, **Machine Learning**, **Streamlit**, and integrated with **MLflow** for experiment tracking and model management.

---

## 🛫 Project 1: Flight Price Prediction (Regression) 🏷️

### 🔧 Skills Learned
- Python  
- Streamlit  
- Machine Learning (Regression)  
- Data Analysis  
- MLflow  

### 🌍 Domain
Travel & Tourism  

### 📝 Problem Statement
Flight pricing is dynamic and influenced by multiple factors. This project builds an **end-to-end ML pipeline** to predict **flight ticket prices** based on departure time, source, destination, airline, and other details. The final solution is deployed as a **Streamlit app** where users can explore flight price trends and make predictions.

### 💡 Business Use Cases
- Help **travelers** plan trips with accurate ticket price forecasts.  
- Support **travel agencies** in marketing and pricing strategies.  
- Enable **businesses** to budget for employee travel.  
- Assist **airlines** in optimizing ticket pricing policies.  

### 🛠️ Approach
1. **Data Preprocessing** 🔄  
   - Handle missing values, duplicates, and inconsistent formats.  
   - Feature engineering (e.g., price per minute, duration splits).  

2. **Model Development** 💡  
   - Perform **EDA** for insights.  
   - Train regression models (Linear Regression, Random Forest, XGBoost).  
   - Track experiments with **MLflow** (RMSE, R²).  

3. **Streamlit App** 📱  
   - Route/airline filtering.  
   - Interactive price visualizations.  
   - On-demand price prediction.  

### 🔍 Results
- Clean and enriched dataset.  
- High-performance regression models.  
- Interactive **Streamlit dashboard** for predictions.  

### 📂 Dataset
- **Source**: [Flight_Price.csv](#)  
- **Features** include: Airline, Date_of_Journey, Source & Destination, Route, Departure/Arrival Time, Duration, Total Stops, Additional Info.  

### 📦 Deliverables
- Python scripts for preprocessing & modeling.  
- MLflow-logged models and metrics.  
- Streamlit app for flight price prediction.  

📊 Results:  


---

## 👨‍💻 Project 2: Customer Satisfaction Prediction (Classification) 💬

### 🔧 Skills Learned
- Python  
- Machine Learning (Classification)  
- Data Analysis  
- Streamlit  
- MLflow  

### 🌍 Domain
Customer Experience  

### 📝 Problem Statement
Passenger satisfaction depends on both service quality and personal factors. This project develops a **classification model** that predicts whether a passenger is **Satisfied / Neutral / Dissatisfied** using demographics, flight distance, and service ratings. The model is deployed as an **interactive Streamlit app**.

### 💡 Business Use Cases
- Improve **customer experience** by identifying dissatisfaction factors.  
- Provide **management insights** to enhance services.  
- Assist **marketing teams** in targeted campaigns.  
- Support **retention strategies** by predicting churn risk.  

### 🛠️ Approach
1. **Data Preprocessing** 🔄  
   - Handle missing values and categorical encodings.  
   - Normalize and group numerical values (e.g., Age Groups).  

2. **Model Development** 💡  
   - Perform **EDA** to uncover patterns.  
   - Train classification models (Logistic Regression, Random Forest, Gradient Boosting).  
   - Track metrics (Accuracy, F1-score) with **MLflow**.  

3. **Streamlit App** 📱  
   - Satisfaction visualizations.  
   - Interactive input form for predictions.  
   - Real-time classification output.  

### 🔍 Results
- Cleaned dataset prepared for modeling.  
- High-accuracy classification models.  
- Intuitive **Streamlit interface** for predictions.  

### 📂 Dataset
- **Source**: [Passenger_Satisfaction.csv](#)  
- **Features** include: Gender, Customer Type, Age, Travel Purpose, Class, Flight Distance, Service Ratings, Satisfaction.  

### 📦 Deliverables
- Preprocessing & ML training scripts.  
- MLflow-logged classification models.  
- Streamlit app for passenger satisfaction prediction.  

📊 Results:  


---
