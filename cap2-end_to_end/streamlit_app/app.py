"""
Streamlit Frontend for Housing Price Prediction API.

This application provides an interactive web interface for predicting
California housing prices using the FastAPI backend.

Author: Carlos Daniel Jiménez
Date: January 2024
"""

import os
import time
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
API_PREDICT_ENDPOINT = f"{API_URL}/api/v1/predict"
API_HEALTH_ENDPOINT = f"{API_URL}/health"
API_MODEL_INFO_ENDPOINT = f"{API_URL}/api/v1/model/info"

# Page config
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_health() -> Dict[str, Any]:
    """
    Check if API is healthy and return status.

    Returns:
        dict: API health status
    """
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        dict: Model information
    """
    try:
        response = requests.get(API_MODEL_INFO_ENDPOINT, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "unknown", "model_loaded": False}


def make_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make prediction request to API.

    Args:
        features: Dictionary with housing features

    Returns:
        dict: Prediction response
    """
    payload = {"instances": [features]}

    response = requests.post(API_PREDICT_ENDPOINT, json=payload, timeout=10)
    response.raise_for_status()

    return response.json()


def create_feature_comparison_chart(features: Dict[str, Any]) -> go.Figure:
    """
    Create a radar chart comparing input features to typical values.

    Args:
        features: Input features dictionary

    Returns:
        Plotly figure
    """
    # Typical values for California housing (normalized)
    typical_values = {
        "Median Age": 28.0,
        "Total Rooms": 2635.0,
        "Total Bedrooms": 537.0,
        "Population": 1425.0,
        "Households": 499.0,
        "Median Income": 3.87,
    }

    # Current values
    current_values = {
        "Median Age": features["housing_median_age"],
        "Total Rooms": features["total_rooms"],
        "Total Bedrooms": features["total_bedrooms"],
        "Population": features["population"],
        "Households": features["households"],
        "Median Income": features["median_income"],
    }

    # Normalize values (percentage of typical)
    categories = list(typical_values.keys())
    typical_normalized = [100] * len(categories)
    current_normalized = [(current_values[cat] / typical_values[cat]) * 100 for cat in categories]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=typical_normalized,
            theta=categories,
            fill="toself",
            name="Typical CA Home",
            line=dict(color="lightblue", width=2),
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=current_normalized,
            theta=categories,
            fill="toself",
            name="Your Input",
            line=dict(color="#FF4B4B", width=2),
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 200])),
        showlegend=True,
        title="Feature Comparison (% of Typical Values)",
    )

    return fig


def create_location_map(latitude: float, longitude: float) -> go.Figure:
    """
    Create a map showing the house location.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Plotly figure
    """
    fig = go.Figure(
        go.Scattermapbox(
            lat=[latitude],
            lon=[longitude],
            mode="markers",
            marker=go.scattermapbox.Marker(size=15, color="red"),
            text=["House Location"],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=latitude, lon=longitude), zoom=9),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
    )

    return fig


# Header
st.title("🏠 California Housing Price Predictor")
st.markdown("""
Predict median house values in California using machine learning.
This application uses a **Random Forest** model trained on the California Housing dataset.
""")

# API Health Check in sidebar
with st.sidebar:
    st.header("🔌 API Status")

    if st.button("🔄 Refresh Status"):
        st.rerun()

    health_status = check_api_health()

    if health_status.get("status") == "healthy":
        st.success("✅ API is healthy")
        model_info = get_model_info()

        if model_info.get("model_loaded"):
            st.info(f"**Model:** {model_info.get('model_version', 'unknown')}")
        else:
            st.warning("⚠️ Model not loaded")
    else:
        st.error("❌ API is unavailable")
        st.caption(f"Error: {health_status.get('error', 'Unknown error')}")
        st.stop()

    st.markdown("---")

    # Input features
    st.header("📊 House Features")

    # Location
    st.subheader("📍 Location")
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.number_input(
            "Longitude", value=-122.23, format="%.2f", help="Geographic longitude coordinate"
        )
    with col2:
        latitude = st.number_input(
            "Latitude", value=37.88, format="%.2f", help="Geographic latitude coordinate"
        )

    # Area characteristics
    st.subheader("🏘️ Area Characteristics")

    housing_median_age = st.slider(
        "Median Age (years)",
        min_value=1,
        max_value=52,
        value=41,
        help="Median age of houses in the block",
    )

    total_rooms = st.number_input(
        "Total Rooms", value=880, step=100, min_value=1, help="Total number of rooms in the block"
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        value=129,
        step=10,
        min_value=1,
        help="Total number of bedrooms in the block",
    )

    population = st.number_input(
        "Population", value=322, step=50, min_value=1, help="Total population in the block"
    )

    households = st.number_input(
        "Households", value=126, step=10, min_value=1, help="Number of households in the block"
    )

    median_income = st.number_input(
        "Median Income (×$10k)",
        value=8.3252,
        format="%.4f",
        step=0.1,
        min_value=0.0,
        help="Median income in units of $10,000",
    )

    # Ocean proximity
    st.subheader("🌊 Ocean Proximity")
    ocean_proximity = st.selectbox(
        "Distance to Ocean",
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        help="Proximity to the ocean",
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Location Preview")
    location_map = create_location_map(latitude, longitude)
    st.plotly_chart(location_map, use_container_width=True)

with col2:
    st.subheader("📈 Key Metrics")

    # Calculate derived features
    rooms_per_household = total_rooms / households if households > 0 else 0
    bedrooms_per_household = total_bedrooms / households if households > 0 else 0
    population_per_household = population / households if households > 0 else 0

    st.metric("Rooms per Household", f"{rooms_per_household:.2f}")
    st.metric("Bedrooms per Household", f"{bedrooms_per_household:.2f}")
    st.metric("Population per Household", f"{population_per_household:.2f}")

# Prediction button
st.markdown("---")
if st.button("🔮 Predict House Price", type="primary"):
    # Build features dictionary
    features = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    # Make prediction
    with st.spinner("🔄 Making prediction..."):
        start_time = time.time()

        try:
            result = make_prediction(features)
            prediction = result["predictions"][0]["predicted_price"]
            model_version = result["model_version"]
            response_time = (time.time() - start_time) * 1000

            # Display results
            st.success("✅ Prediction Complete!")

            # Main prediction result
            st.markdown("---")
            st.subheader("💰 Predicted Median House Value")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h1 style='color: #FF4B4B; margin: 0;'>${prediction:,.2f}</h1>
                    <p style='margin: 0; color: #666;'>Median House Value</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.metric("Model Version", model_version)

            with col3:
                st.metric("Response Time", f"{response_time:.0f}ms")

            # Feature comparison
            st.markdown("---")
            st.subheader("📊 Feature Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Radar chart
                radar_chart = create_feature_comparison_chart(features)
                st.plotly_chart(radar_chart, use_container_width=True)

            with col2:
                # Input features table
                st.markdown("**Input Features**")
                features_df = pd.DataFrame([features]).T
                features_df.columns = ["Value"]
                st.dataframe(features_df, use_container_width=True)

            # Price context
            st.markdown("---")
            st.subheader("💡 Price Context")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Price per Room",
                    f"${prediction / total_rooms:,.2f}",
                    help="Predicted price divided by total rooms",
                )

            with col2:
                st.metric(
                    "Price per Household",
                    f"${prediction / households:,.2f}",
                    help="Predicted price divided by number of households",
                )

            with col3:
                # Income to price ratio
                income_to_price_ratio = (median_income * 10000) / prediction
                st.metric(
                    "Income-to-Price Ratio",
                    f"{income_to_price_ratio:.2%}",
                    help="Median income as percentage of house price",
                )

            # Additional insights
            with st.expander("📋 Detailed Analysis"):
                st.markdown("""
                ### How to interpret the results:

                **Predicted Price**: Median house value for the specified area and features.

                **Feature Comparison**: Shows how your input compares to typical California homes.
                - Values >100% indicate above-average features
                - Values <100% indicate below-average features

                **Price Context**:
                - **Price per Room**: Higher values suggest premium locations or features
                - **Income-to-Price Ratio**: A healthy ratio is typically 20-30%

                **Model Information**:
                - Model: Random Forest Regressor
                - Training Data: California Housing Dataset
                - Features: 9 input features (location, demographics, house characteristics)
                """)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
            st.info("Please make sure the API is running at " + API_URL)

        except KeyError as e:
            st.error(f"❌ Invalid response format: {str(e)}")

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🚀 **Powered by:**")
    st.caption("FastAPI + scikit-learn + MLflow")

with col2:
    st.caption("📚 **Dataset:**")
    st.caption("California Housing (1990)")

with col3:
    st.caption("👨‍💻 **Author:**")
    st.caption("Carlos Daniel Jiménez")

# About section in sidebar
with st.sidebar:
    st.markdown("---")
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        ### Housing Price Predictor

        This application predicts median house values in California
        based on various features:

        - **Location**: Longitude and latitude
        - **Demographics**: Population, households, median income
        - **Housing**: Rooms, bedrooms, house age
        - **Geography**: Ocean proximity

        **Model**: Random Forest Regressor trained on the California
        Housing dataset (1990 census data).

        **Tech Stack**:
        - Frontend: Streamlit
        - Backend: FastAPI
        - ML: scikit-learn
        - Tracking: MLflow + W&B
        - Deployment: Docker + Cloud Run

        **Source Code**: [GitHub](https://github.com/your-repo)
        """)

    with st.expander("🎯 Example Scenarios"):
        st.markdown("""
        ### Try these locations:

        **San Francisco Bay Area**:
        - Longitude: -122.23
        - Latitude: 37.88
        - Income: 8.3252

        **Los Angeles Inland**:
        - Longitude: -118.30
        - Latitude: 34.26
        - Income: 1.4936

        **San Diego Coastal**:
        - Longitude: -117.15
        - Latitude: 32.75
        - Income: 5.5
        """)
