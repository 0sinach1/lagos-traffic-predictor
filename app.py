import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Lagos Traffic Predictor",
    page_icon="🚦",
    layout="centered"
)

# ── Load and train ────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    url = "https://raw.githubusercontent.com/0sinach1/lagos-traffic-predictor/main/data/lagos_traffic_data_2months.csv"
    df = pd.read_csv(url)

    # Features
    features = ['hour', 'is_weekend', 'is_rush_hour', 'distance_km',
                 'normal_duration_min', 'delay_min', 'route_name', 'day_of_week']
    target = 'traffic_label'

    df = df.dropna(subset=[target])

    # Encode categoricals
    le_route = LabelEncoder()
    le_day = LabelEncoder()
    le_target = LabelEncoder()

    df['route_encoded'] = le_route.fit_transform(df['route_name'])
    df['day_encoded'] = le_day.fit_transform(df['day_of_week'])
    df['target_encoded'] = le_target.fit_transform(df[target])

    X = df[['hour', 'is_weekend', 'is_rush_hour', 'distance_km',
            'normal_duration_min', 'delay_min', 'route_encoded', 'day_encoded']]
    y = df['target_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.27, random_state=42
    )

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, le_route, le_day, le_target, df, acc

with st.spinner("Loading model..."):
    model, le_route, le_day, le_target, df, acc = load_and_train()

# ── Header ────────────────────────────────────────────────────
st.title("🚦 Lagos Traffic Predictor")
st.markdown(
    f"XGBoost-style model trained on **{len(df):,} real Lagos traffic records** across 10 routes. "
    f"Model accuracy: **{acc:.1%}** — built by "
    f"[Elvis Osinachi](https://ifeanyiosinachi.vercel.app)."
)
st.divider()

# ── Inputs ────────────────────────────────────────────────────
st.subheader("Journey Details")

col1, col2 = st.columns(2)

routes = sorted(df['route_name'].unique().tolist())
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

with col1:
    route = st.selectbox("Route", routes)
    hour = st.slider("Hour of Day", 0, 23, 8)
    day = st.selectbox("Day of Week", days)

with col2:
    is_weekend = st.checkbox("Weekend?", value=day in ['Saturday', 'Sunday'])
    is_rush_hour = st.checkbox("Rush Hour?", value=hour in [7, 8, 9, 17, 18, 19])

# Get route stats from data
route_stats = df[df['route_name'] == route][['distance_km', 'normal_duration_min', 'delay_min']].mean()
distance_km = route_stats['distance_km']
normal_duration = route_stats['normal_duration_min']

st.caption(f"Route stats: {distance_km:.1f} km · Normal duration: {normal_duration:.0f} mins")

delay = st.slider("Expected Delay (mins)", -20, 60,
                  int(route_stats['delay_min']) if is_rush_hour else 0)

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍 Predict Traffic", use_container_width=True):

    route_enc = le_route.transform([route])[0]
    day_enc = le_day.transform([day])[0] if day in le_day.classes_ else 0

    input_data = pd.DataFrame([{
        'hour': hour,
        'is_weekend': int(is_weekend),
        'is_rush_hour': int(is_rush_hour),
        'distance_km': distance_km,
        'normal_duration_min': normal_duration,
        'delay_min': delay,
        'route_encoded': route_enc,
        'day_encoded': day_enc
    }])

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    label = le_target.inverse_transform([pred])[0]

    # Display result
    st.subheader("Prediction")

    color_map = {
        'None': ('🟢', 'success', 'No significant traffic expected.'),
        'Light': ('🟡', 'warning', 'Light traffic — minor delays possible.'),
        'Moderate': ('🟠', 'warning', 'Moderate traffic — plan extra time.'),
        'Heavy': ('🔴', 'error', 'Heavy traffic — expect significant delays.'),
        'Severe': ('🔴', 'error', 'Severe congestion — consider alternative routes.'),
    }

    icon, style, msg = color_map.get(label, ('⚪', 'info', ''))

    if style == 'success':
        st.success(f"{icon} **{label.upper()} TRAFFIC** on {route}")
    elif style == 'error':
        st.error(f"{icon} **{label.upper()} TRAFFIC** on {route}")
    else:
        st.warning(f"{icon} **{label.upper()} TRAFFIC** on {route}")

    st.markdown(msg)

    # Confidence breakdown
    with st.expander("Confidence breakdown"):
        classes = le_target.inverse_transform(range(len(proba)))
        conf_df = pd.DataFrame({'Traffic Level': classes, 'Confidence': proba})
        conf_df = conf_df.sort_values('Confidence', ascending=False)
        st.dataframe(conf_df, use_container_width=True, hide_index=True)

st.divider()

# ── Route Explorer ────────────────────────────────────────────
with st.expander("📊 Explore Route Data"):
    selected = st.selectbox("Select route to explore", routes, key="explore")
    route_df = df[df['route_name'] == selected][['hour', 'traffic_label', 'delay_min']].copy()
    avg = route_df.groupby('hour')['delay_min'].mean().reset_index()
    avg.columns = ['Hour', 'Avg Delay (mins)']
    st.line_chart(avg.set_index('Hour'))
application = app

st.caption(
    "Model: GradientBoosting | Dataset: 10 Lagos routes, 2 months | "
    "Accuracy: {:.1%} | GitHub: [0sinach1](https://github.com/0sinach1)".format(acc)
)
