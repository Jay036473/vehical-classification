import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(page_title="Vehicle Maintenance AI", layout="wide")
st.title("🚗 Vehicle Maintenance Classification System")


# ===============================
# DATA LOADING & PREPROCESSING (ADDED 9% NOISE)
# ===============================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(r'vehical.csv')

    # REMOVE ENGINE SIZE COLUMN
    if 'Engine_Size' in df.columns:
        df = df.drop(columns=['Engine_Size'])

    # Create target column if it doesn't exist
    if 'Need_Maintenance' not in df.columns:
        # 1. Create the perfect, 100% accurate logical rule
        perfect_status = np.where(
            (df['Reported_Issues'] > 0) |
            (df['Brake_Condition'] == 'Worn Out') |
            (df['Battery_Status'] == 'Weak') |
            (df['Tire_Condition'] == 'Worn Out'), 1, 0
        )

        # 2. Introduce exactly 9% random "real-world" noise
        np.random.seed(42)  # Keeps the randomness consistent every time you run it
        noise_mask = np.random.rand(len(df)) < 0.09  # Selects 9% of the rows randomly

        # 3. Flip the answers for those 9% of rows (simulating human error)
        df['Need_Maintenance'] = np.where(noise_mask, 1 - perfect_status, perfect_status)

    # Remove Date columns
    for col in list(df.columns):
        if 'Date' in col:
            df = df.drop(columns=[col])

    df = df.dropna()
    return df


df = load_and_preprocess_data()

# ===============================
# ENCODING
# ===============================
label_encoders = {}
df_encoded = df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

X = df_encoded.drop('Need_Maintenance', axis=1)
y = df_encoded['Need_Maintenance']


# ===============================
# MODEL TRAINING & GET SCORE
# ===============================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=34
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,  # Constrained to prevent overfitting to the new noise
        random_state=42
    )

    clf.fit(X_train, y_train)

    # Calculate the accuracy score on the test data
    score = clf.score(X_test, y_test)

    return clf, score


# Unpack both the model and the score
model, model_score = train_model(X, y)

# ===============================
# DISPLAY MODEL SCORE
# ===============================
st.markdown(
    f"""
    <div style="background-color: #2b3a42; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 25px; border: 1px solid #4CAF50;">
        <h3 style="color: #4CAF50; margin: 0;">🎯 Model Accuracy Score: {model_score * 100:.2f}%</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# TABS SETUP
# ===============================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction App", "📊 Interactive Insights", "📋 Data Preview"])

# ===============================
# TAB 1: USER INPUT & PREDICTION
# ===============================
with tab1:
    st.header("🔍 Predict Your Vehicle's Status")
    st.write("Fill in the details below to check if your vehicle requires maintenance.")

    user_input_dict = {}
    cols = st.columns(2)

    for i, column in enumerate(X.columns):

        with cols[i % 2]:

            if column in label_encoders:

                options = list(label_encoders[column].classes_)

                selected = st.selectbox(
                    f"Select {column}",
                    options,
                    index=None,
                    placeholder="Choose an option..."
                )

                if selected:
                    user_input_dict[column] = label_encoders[column].transform([selected])[0]
                else:
                    user_input_dict[column] = None

            elif column in ['Reported_Issues', 'Vehicle_Age', 'Service_History', 'Accident_History']:

                user_input_dict[column] = st.number_input(
                    f"Enter {column}",
                    value=0,
                    step=1
                )

            else:

                user_input_dict[column] = st.number_input(
                    f"Enter {column}",
                    value=0.0
                )

    st.markdown("---")

    if st.button("Check Maintenance Requirement", use_container_width=True):

        if None in user_input_dict.values():

            st.warning("Please fill in all the details before predicting.")

        else:

            input_df = pd.DataFrame([user_input_dict])

            prediction = model.predict(input_df)

            probability = model.predict_proba(input_df)[0][1]

            if prediction[0] == 1:

                st.error("⚠️ Maintenance Required!")
                st.write(f"Confidence Level: **{probability * 100:.1f}%**")

            else:

                st.success("✅ Vehicle is Safe!")
                st.write(f"Risk Probability: **{probability * 100:.1f}%**")

# ===============================
# TAB 2: DATA VISUALIZATION
# ===============================
with tab2:
    st.header("📊 Interactive Data Insights")

    row1 = st.columns(2)
    row2 = st.columns(2)
    row3 = st.columns(2)

    # 1 Maintenance Ratio
    with row1[0]:
        temp_df = df.copy()
        temp_df['Status'] = temp_df['Need_Maintenance'].map(
            {1: 'Needs Service', 0: 'Healthy'}
        )
        fig1 = px.pie(
            temp_df,
            names='Status',
            hole=0.4,
            title="Overall Fleet Health"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # 2 Feature Importance
    with row1[1]:
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance")
        fig2 = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Top Decision Factors"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 3 Reported Issues vs Maintenance
    with row2[0]:
        fig3 = px.histogram(
            df,
            x="Reported_Issues",
            color="Need_Maintenance",
            barmode="group",
            title="Impact of Reported Issues on Maintenance"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # 4 Vehicle Model Health
    with row2[1]:
        fig5 = px.histogram(
            df,
            x="Vehicle_Model",
            color="Need_Maintenance",
            title="Maintenance Needs by Vehicle Model"
        )
        st.plotly_chart(fig5, use_container_width=True)

    # 5 Vehicle Age vs Distance Density
    with row3[0]:
        fig8 = px.density_heatmap(
            df,
            x="Vehicle_Age",
            y="Odometer_Reading",
            title="Vehicle Age vs Distance Density"
        )
        st.plotly_chart(fig8, use_container_width=True)

# ===============================
# TAB 3: DATA PREVIEW
# ===============================
with tab3:
    st.header("📋 Dataset Preview")

    # Displays the number of rows and columns in the dataset
    st.write(f"This dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    st.markdown("---")
    st.write("Below is the raw data used to train the machine learning model:")

    # Displays the data as an interactive table
    st.dataframe(df, use_container_width=True)
