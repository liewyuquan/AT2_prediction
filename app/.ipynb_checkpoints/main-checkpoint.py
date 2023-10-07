from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

xgbr_2 = load('../model/xgbr_2_best.joblib')

@app.get("/")
def read_root():
    return {"Hello": "World" "You" "Are" "Beautiful"}

@app.get('/health/', status_code=200)
def healthcheck():
    return 'AT2 API is all ready to go!'

def format_features(
    item_id: str,
    store_id: str,
    date: str,
    revenue: float,
    ):
    return {
        'Item_id': [general_health],
        'Store_id': [checkup],
        'date': [exercise],
        'Revenue': [heart_disease],
    }

@app.get('/sales/stores/items/')
def predict(
    item_id: str,
    store_id: str,
    date: str,
    revenue: float,
):
    features = format_features(
        item_id,
        store_id,
        date,
        revenue,
        )
    obs = pd.DataFrame(features)
    pred = xgbr_2.predict(obs)
    return JSONResponse(pred.tolist())