import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Flight & Passenger Prediction App", page_icon="✈️", layout="wide")

# -----------------------------
# Load Data & Models
# -----------------------------
@st.cache_data
def load_flight_data():
    return pd.read_csv("cleaned_flight_data.csv")

@st.cache_resource
def load_models():
    return {
        "Decision Tree": joblib.load("Decision Tree_model.pkl"),
        "Gradient Boosting": joblib.load("Gradient Boosting_model.pkl"),
        "Lasso Regression": joblib.load("Lasso Regression_model.pkl"),
        "Linear Regression": joblib.load("Linear Regression_model.pkl"),
        "Random Forest": joblib.load("Random Forest_model.pkl"),
        "Ridge Regression": joblib.load("Ridge Regression_model.pkl"),
        "XGBoost": joblib.load("XGBoost_model.pkl"),
    }

@st.cache_resource
def load_satisfaction_models():
    return {
        "Logistic Regression": joblib.load("Logistic_Regression.pkl"),
        "Random Forest": joblib.load("Random_Forest.pkl"),
        "Gradient Boosting": joblib.load("Gradient_Boosting.pkl"),
        "K-Nearest Neighbors": joblib.load("K-Nearest_Neighbors.pkl"),
        "XGBoost": joblib.load("XGBoost.pkl"),
    }

flight_data = load_flight_data()
flight_models = load_models()
satisfaction_models = load_satisfaction_models()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Flight Price", "Customer Satisfaction", "Creator Info"])

# -----------------------------
# Page 1: Overview
# -----------------------------
if page == "Overview":
    st.title("✈️ Flight & Passenger Prediction App")
    st.image("flight.jpeg")

    st.markdown("""
    ## 📌 Project Overview

    This application combines two powerful **machine learning projects**:  

    ### 🛫 Flight Price Prediction  
    - Predicts airline ticket prices using historical data.  
    - Explore flight data with filters by **source, destination, and airline**.  
    - Visualize price trends using **boxplots and histograms**.  
    - Predict flight prices using models like Decision Tree, Random Forest, XGBoost, etc.  

    ### 🙂 Customer Satisfaction Prediction  
    - Predicts whether passengers are **Satisfied** or **Dissatisfied** based on flight experience.  
    - Uses engineered features like **Flight Distance, Total Delay, Service Score, Age Group**, etc.  
    - Try out multiple ML models like Logistic Regression, Random Forest, Gradient Boosting, KNN, and XGBoost.  

    ---
    This tool helps both **travelers and airline analysts** to:  
    - Compare routes and prices.  
    - Analyze passenger experience.  
    - Estimate flight prices and satisfaction levels.  
    """)

# -----------------------------
# Page 2: Flight Price
# -----------------------------
elif page == "Flight Price":
    st.title("🛫 Flight Price Prediction")

    # Sidebar filters
    st.sidebar.header("🔎 Filter Flights")
    source_filter = st.sidebar.selectbox("Source", sorted([c.replace("Source_", "") for c in flight_data.columns if c.startswith("Source_")]))
    destination_filter = st.sidebar.selectbox("Destination", sorted([c.replace("Destination_", "") for c in flight_data.columns if c.startswith("Destination_")]))
    airline_filter = st.sidebar.selectbox("Airline", sorted([c.replace("Airline_", "") for c in flight_data.columns if c.startswith("Airline_")]))

    # Filtered Data
    source_col, dest_col, airline_col = f"Source_{source_filter}", f"Destination_{destination_filter}", f"Airline_{airline_filter}"
    mask = (flight_data[source_col] == 1) & (flight_data[dest_col] == 1) & (flight_data[airline_col] == 1)
    filtered_data = flight_data[mask]

    # -----------------------------
    # Tabs for Data, Visualizations, Prediction
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["📋 Filtered Data", "📊 Visualizations", "🧮 Prediction"])

    # Tab 1: Filtered Data
    with tab1:
        st.subheader(f"📋 Filtered Data — {source_filter} ➝ {destination_filter} | Airline: {airline_filter}")
        if filtered_data.empty:
            st.warning("No flights available for the selected filters.")
        else:
            # KPIs
            col1, col2, col3 = st.columns(3)
            col1.metric("Min Price", f"₹{filtered_data['Price'].min():,.0f}")
            col2.metric("Average Price", f"₹{filtered_data['Price'].mean():,.0f}")
            col3.metric("Max Price", f"₹{filtered_data['Price'].max():,.0f}")

            st.dataframe(filtered_data, use_container_width=True)

            # Download option
            st.download_button("⬇️ Download Filtered Data", filtered_data.to_csv(index=False), "filtered_flights.csv", "text/csv")

    # Tab 2: Visualizations
    with tab2:
        st.subheader("📊 Visualizations")
        if filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            viz_tab1, viz_tab2 = st.tabs(["Boxplot", "Histogram"])
            with viz_tab1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=filtered_data, x=airline_col, y="Price", palette="viridis", ax=ax)
                ax.set_title("Price Distribution per Airline", fontsize=14)
                st.pyplot(fig)
            with viz_tab2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(filtered_data["Price"], kde=True, color="skyblue", ax=ax)
                ax.set_title("Price Frequency Distribution", fontsize=14)
                st.pyplot(fig)

    # Tab 3: Prediction
    with tab3:
        st.subheader("🧮 Predict Flight Price")
        with st.form("prediction_form"):
            departure_date = st.date_input("Departure Date", min_value=datetime.today())
            departure_time = st.time_input("Departure Time", value=datetime(2023, 1, 1, 12, 0).time())
            journey_day, journey_month = departure_date.day, departure_date.month
            dep_hour, dep_min = departure_time.hour, departure_time.minute

            duration_hours = st.number_input("Flight Duration (Hours)", 0, 24, 2)
            duration_mins = st.number_input("Flight Duration (Minutes)", 0, 59, 30)
            total_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
            price_per_minute = st.number_input("Price per Minute (₹)", 1, 100, 1)

            model_choice = st.selectbox("Select Model", list(flight_models.keys()))
            submitted = st.form_submit_button("Predict Price")

        if submitted:
            # Build input vector
            input_features = {
                "Total_Stops": total_stops,
                "Journey_Day": journey_day,
                "Journey_Month": journey_month,
                "Dep_hour": dep_hour,
                "Dep_min": dep_min,
                "Arr_hour": (dep_hour + duration_hours) % 24,
                "Arr_min": (dep_min + duration_mins) % 60,
                "Duration_hours": duration_hours,
                "Duration_mins": duration_mins,
                "Price_per_minute": price_per_minute,
            }
            for col in flight_data.columns:
                if col.startswith("Airline_") or col.startswith("Source_") or col.startswith("Destination_"):
                    input_features[col] = 0
            input_features[f"Airline_{airline_filter}"] = 1
            input_features[f"Source_{source_filter}"] = 1
            input_features[f"Destination_{destination_filter}"] = 1

            input_vector = pd.DataFrame([input_features])

            try:
                model = flight_models[model_choice]
                predicted_price = model.predict(input_vector)[0]
                st.success(f"✈️ Estimated Flight Price: ₹{predicted_price:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# Page 3: Customer Satisfaction
# -----------------------------
elif page == "Customer Satisfaction":
    st.title("🙂 Passenger Satisfaction Prediction")

    features = [
        'Flight Distance', 'Total Delay', 'Is Delayed',
        'Age Group', 'Service Score', 'Gender_Female', 'Gender_Male',
        'Customer Type_Loyal Customer', 'Customer Type_disloyal Customer',
        'Travel Purpose Class_Business travel - Business',
        'Travel Purpose Class_Business travel - Eco',
        'Travel Purpose Class_Business travel - Eco Plus',
        'Travel Purpose Class_Personal Travel - Business',
        'Travel Purpose Class_Personal Travel - Eco',
        'Travel Purpose Class_Personal Travel - Eco Plus'
    ]

    # Collect inputs
    flight_distance = st.number_input("Flight Distance", min_value=0, step=1)
    total_delay = st.number_input("Total Delay (Minutes)", min_value=0, step=1)
    is_delayed = st.selectbox("Is Delayed?", [0, 1])
    age_group = st.selectbox("Age Group", ["Young", "Adult", "Senior"])
    service_score = st.slider("Service Score", 1, 5)
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    travel_class = st.selectbox(
        "Travel Purpose & Class",
        [
            "Business travel - Business", "Business travel - Eco", "Business travel - Eco Plus",
            "Personal Travel - Business", "Personal Travel - Eco", "Personal Travel - Eco Plus"
        ]
    )

    # Placeholder
    input_data = {col: 0 for col in features}
    input_data['Flight Distance'] = flight_distance
    input_data['Total Delay'] = total_delay
    input_data['Is Delayed'] = is_delayed

    age_group_map = {"Young": 0, "Adult": 1, "Senior": 2}
    input_data['Age Group'] = age_group_map[age_group]
    input_data['Service Score'] = service_score

    input_data[f"Gender_{gender}"] = 1
    input_data[f"Customer Type_{customer_type}"] = 1
    input_data[f"Travel Purpose Class_{travel_class}"] = 1

    input_df = pd.DataFrame([input_data])

    # Model Selection
    model_choice = st.selectbox("Choose Model", list(satisfaction_models.keys()))
    model = satisfaction_models[model_choice]

    # Prediction
    if st.button("Predict Satisfaction"):
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ Passenger is **Satisfied**")
        else:
            st.error("⚠️ Passenger is **Dissatisfied**")

# PAGE 4: Creator Info
elif page == "Creator Info":
    st.title(":female-technologist: Creator Info")
    st.write("""
    **Developed by:** [Karuna]  
             
    **Skills:** Python, Machine Learning, Data Cleaning, Data Preparation, Data Visualization
             
    **GitHub:** [https://github.com/karuna-2828/Flight-Price-Customer-Satisfaction-Prediction]
    """)