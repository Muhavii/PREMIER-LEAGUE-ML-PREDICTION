import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('premier_league_predictor.pkl')

st.title("Premier League Match Outcome Predictor")

# User inputs for match details
home_team = st.selectbox("Select Home Team", options=["Chelsea", "Arsenal", "Liverpool", "Manchester United", "Tottenham"])
away_team = st.selectbox("Select Away Team", options=["Chelsea", "Arsenal", "Liverpool", "Manchester United", "Tottenham"])

hs = st.number_input("Home Shots (HS)", min_value=0)
as_ = st.number_input("Away Shots (AS)", min_value=0)
hst = st.number_input("Home Shots on Target (HST)", min_value=0)
ast = st.number_input("Away Shots on Target (AST)", min_value=0)
hf = st.number_input("Home Fouls (HF)", min_value=0)
af = st.number_input("Away Fouls (AF)", min_value=0)
hc = st.number_input("Home Corners (HC)", min_value=0)
ac = st.number_input("Away Corners (AC)", min_value=0)
hy = st.number_input("Home Yellow Cards (HY)", min_value=0)
ay = st.number_input("Away Yellow Cards (AY)", min_value=0)

if st.button("Predict Match Outcome"):
    # Prepare input data for prediction
    new_match_data = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'HS': hs,
        'AS': as_,
        'HST': hst,
        'AST': ast,
        'HF': hf,
        'AF': af,
        'HC': hc,
        'AC': ac,
        'HY': hy,
        'AY': ay,
    }
    
    # Convert new match data to DataFrame and preprocess as needed
    new_match_df = pd.DataFrame([new_match_data])
    
    # One-hot encode the input data
    new_match_encoded = pd.get_dummies(new_match_df, drop_first=True)

    # Align columns with training set by adding missing columns with default values of 0
    for column in model.feature_names_in_:  # Use feature_names_in_ from your model if available
        if column not in new_match_encoded.columns:
            new_match_encoded[column] = 0
            
    new_match_encoded = new_match_encoded[model.feature_names_in_]  # Ensure order matches X

    # Make prediction
    prediction = model.predict(new_match_encoded)

    # Display predicted outcome
    st.success(f"The predicted outcome is: {prediction[0]}")  # Outputs H, D, or A

    #TO USE THIS APP, RUN THE COMMAND: streamlit run app.py