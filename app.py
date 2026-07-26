import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import time
from math import radians, sin, cos, sqrt, asin

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Zomato Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .stApp {
            background: linear-gradient(180deg, #1a1c23 0%, #0e1117 100%);
        }
        .hero {
            padding: 1.6rem 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #E23744 0%, #b3202a 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(226,55,68,0.25);
        }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p { margin: .35rem 0 0 0; opacity: .9; font-size: 1rem; }
        .result-card {
            padding: 1.8rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #1f6feb22, #1f6feb0d);
            border: 1px solid #1f6feb55;
            text-align: center;
        }
        .result-card h2 { color: #4fa3ff; font-size: 2.6rem; margin: 0; }
        .result-card p { color: #aaa; margin-top: .3rem; }
        .metric-pill {
            background: #1c1f26;
            border-radius: 12px;
            padding: .8rem 1rem;
            border: 1px solid #2a2e37;
        }
        section[data-testid="stSidebar"] {
            background-color: #12141a;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Delivery_Time_Prediction.pkl")

model = load_model()

# ----------------------------------------------------------------------------
# CONSTANTS — must mirror the encoding used during training
# ----------------------------------------------------------------------------
WEATHER_OPTIONS = ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
ORDER_TYPE_OPTIONS = ["Buffet", "Drinks", "Meal", "Snack"]
VEHICLE_TYPE_OPTIONS = ["bicycle", "electric_scooter", "motorcycle", "scooter"]
CITY_OPTIONS = ["Metropolitian", "Semi-Urban", "Urban"]
TRAFFIC_OPTIONS = ["Low", "Medium", "High", "Jam"]
FESTIVAL_OPTIONS = ["No", "Yes"]

# Categories dropped by OneHotEncoder(drop='first') -- alphabetically first each time
WEATHER_ENC = WEATHER_OPTIONS[1:]        # Fog, Sandstorms, Stormy, Sunny, Windy
ORDER_TYPE_ENC = ORDER_TYPE_OPTIONS[1:]  # Drinks, Meal, Snack
VEHICLE_TYPE_ENC = VEHICLE_TYPE_OPTIONS[1:]  # electric_scooter, motorcycle, scooter
CITY_ENC = CITY_OPTIONS[1:]              # Semi-Urban, Urban


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def one_hot(value, categories_after_drop, all_categories):
    """Returns a list matching OneHotEncoder(drop='first') output order."""
    row = []
    for cat in categories_after_drop:
        row.append(1.0 if value == cat else 0.0)
    return row


def build_feature_vector(inputs: dict) -> np.ndarray:
    """
    Reconstructs the exact 28-column feature order produced by the
    ColumnTransformer used during training:
      [Road_traffic_density(ordinal)]
      + [Weather one-hot (5)]
      + [Type_of_order one-hot (3)]
      + [Type_of_vehicle one-hot (3)]
      + [City one-hot (2)]
      + [Festival(ordinal)]
      + [remainder passthrough: Delivery_person_Age, Delivery_person_Ratings,
         Restaurant_latitude, Restaurant_longitude, Delivery_location_latitude,
         Delivery_location_longitude, Vehicle_condition, multiple_deliveries,
         Order_Hour, Order_Minute, Pickup_Hour, Pickup_Minute, distance_km]
    """
    row = []

    # t1: Road_traffic_density ordinal
    row.append(float(TRAFFIC_OPTIONS.index(inputs["traffic"])))

    # t2: one-hot blocks
    row += one_hot(inputs["weather"], WEATHER_ENC, WEATHER_OPTIONS)
    row += one_hot(inputs["order_type"], ORDER_TYPE_ENC, ORDER_TYPE_OPTIONS)
    row += one_hot(inputs["vehicle_type"], VEHICLE_TYPE_ENC, VEHICLE_TYPE_OPTIONS)
    row += one_hot(inputs["city"], CITY_ENC, CITY_OPTIONS)

    # t3: Festival ordinal
    row.append(float(FESTIVAL_OPTIONS.index(inputs["festival"])))

    # remainder passthrough (order matters!)
    distance = haversine(
        inputs["rest_lat"], inputs["rest_lon"],
        inputs["del_lat"], inputs["del_lon"],
    )
    row += [
        inputs["age"],
        inputs["ratings"],
        inputs["rest_lat"],
        inputs["rest_lon"],
        inputs["del_lat"],
        inputs["del_lon"],
        inputs["vehicle_condition"],
        inputs["multiple_deliveries"],
        inputs["order_hour"],
        inputs["order_minute"],
        inputs["pickup_hour"],
        inputs["pickup_minute"],
        distance,
    ]

    return np.array(row).reshape(1, -1), distance


# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown("""
    <div class="hero">
        <h1>🛵 Zomato Delivery Time Predictor</h1>
        <p>A machine learning app that estimates food delivery time using a Random Forest model
        trained on real-world delivery operations data.</p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR — INPUTS
# ----------------------------------------------------------------------------
st.sidebar.header("📦 Order & Delivery Details")

with st.sidebar.expander("🧑 Delivery Partner", expanded=True):
    age = st.slider("Delivery person age", 18, 60, 29)
    ratings = st.slider("Delivery person rating", 1.0, 5.0, 4.6, 0.1)
    vehicle_type = st.selectbox("Vehicle type", VEHICLE_TYPE_OPTIONS, index=2)
    vehicle_condition = st.selectbox("Vehicle condition (0=worst, 3=best)", [0, 1, 2, 3], index=2)

with st.sidebar.expander("📍 Locations", expanded=True):
    st.caption("Restaurant location")
    rest_lat = st.number_input("Restaurant latitude", value=12.9716, format="%.6f")
    rest_lon = st.number_input("Restaurant longitude", value=77.5946, format="%.6f")
    st.caption("Delivery location")
    del_lat = st.number_input("Delivery latitude", value=12.9352, format="%.6f")
    del_lon = st.number_input("Delivery longitude", value=77.6245, format="%.6f")
    city = st.selectbox("City type", CITY_OPTIONS, index=2)

with st.sidebar.expander("🌦️ Conditions", expanded=True):
    weather = st.selectbox("Weather conditions", WEATHER_OPTIONS, index=4)
    traffic = st.selectbox("Road traffic density", TRAFFIC_OPTIONS, index=1)
    festival = st.selectbox("Festival day?", FESTIVAL_OPTIONS, index=0)

with st.sidebar.expander("🍽️ Order Details", expanded=True):
    order_type = st.selectbox("Type of order", ORDER_TYPE_OPTIONS, index=2)
    multiple_deliveries = st.selectbox("Multiple deliveries", [0, 1, 2, 3], index=0)
    order_time = st.time_input("Order placed at", value=time(19, 30))
    pickup_time = st.time_input("Order picked up at", value=time(19, 42))

predict_btn = st.sidebar.button("🔮 Predict Delivery Time", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# MAIN AREA
# ----------------------------------------------------------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("📋 Order Summary")
    m1, m2, m3 = st.columns(3)
    dist_preview = haversine(rest_lat, rest_lon, del_lat, del_lon)
    with m1:
        st.markdown(f"<div class='metric-pill'><b>Distance</b><br>{dist_preview:.2f} km</div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-pill'><b>Traffic</b><br>{traffic}</div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-pill'><b>Weather</b><br>{weather}</div>", unsafe_allow_html=True)

    st.map(pd.DataFrame({
        "lat": [rest_lat, del_lat],
        "lon": [rest_lon, del_lon],
    }), size=40, color="#E23744")

with col2:
    st.subheader("🎯 Prediction")
    if predict_btn:
        inputs = dict(
            age=age, ratings=ratings, vehicle_type=vehicle_type,
            vehicle_condition=vehicle_condition, rest_lat=rest_lat, rest_lon=rest_lon,
            del_lat=del_lat, del_lon=del_lon, city=city, weather=weather,
            traffic=traffic, festival=festival, order_type=order_type,
            multiple_deliveries=multiple_deliveries,
            order_hour=order_time.hour, order_minute=order_time.minute,
            pickup_hour=pickup_time.hour, pickup_minute=pickup_time.minute,
        )
        try:
            X, dist = build_feature_vector(inputs)
            pred = model.predict(X)[0]
            st.markdown(f"""
                <div class="result-card">
                    <p>Estimated Delivery Time</p>
                    <h2>{pred:.0f} min</h2>
                    <p>≈ {pred/60:.1f} hours · {dist:.2f} km trip</p>
                </div>
            """, unsafe_allow_html=True)
            st.success("Prediction generated successfully using the trained Random Forest model.")
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")
    else:
        st.info("Fill in the details on the left and click **Predict Delivery Time**.")

st.divider()

with st.expander("ℹ️ About this project"):
    st.markdown("""
    This app predicts **food delivery time (in minutes)** using a **Random Forest Regressor**
    trained on the Zomato delivery operations dataset. Features include delivery partner
    details, weather, traffic density, order type, vehicle type, city type, and the
    great-circle **distance** between restaurant and delivery location (Haversine formula).

    **Model:** RandomForestRegressor (`n_estimators=200`, `max_depth=15`, `min_samples_leaf=5`)
    **Tech stack:** Python · scikit-learn · pandas · Streamlit
    """)

st.caption("Built with Streamlit · Model: RandomForestRegressor · Data: Zomato Delivery Operations Dataset")
