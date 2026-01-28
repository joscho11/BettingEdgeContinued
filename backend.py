from fastapi import FastAPI
import joblib
import pandas as pd
import xgboost as xgb

#sklearn version from training 1.6.1
from sklearn import compose

class FinalCfg:
    pass

app = FastAPI()
model = joblib.load("C:\\Users\\josep\\Desktop\\random_stuff\\random_files\\fantasy_model.pkl")

# TEMP feature builder (replace with real data later)
def build_features(home_team: str, away_team: str, week: int):
    """
    This must match the feature order used in training.
    Replace dummy values with real stats later.
    """
    return pd.DataFrame([{
        "home_advantage": 1,
        "home_off_rating": 0.6,
        "away_def_rating": 0.4,
        "pace_diff": 0.1
    }])

@app.get("/spread")
def get_spread(home: str, away: str, week: int):
    X = build_features(home, away, week)
    spread = model.predict(X)[0]

    return {
        "week": week,
        "home_team": home,
        "away_team": away,
        "model_spread": round(float(spread), 2)
    }