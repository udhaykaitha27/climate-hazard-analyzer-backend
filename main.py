import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
from scipy.stats import linregress
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Climate Hazard Trend Analyzer API")

# Enable CORS for all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Safe MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
cache_collection = None
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db = client["climate_db"]
    cache_collection = db["weather_cache"]
    # Test connection
    client.server_info()
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Warning: MongoDB connection failed ({e}). Running without cache.")

@app.get("/")
def read_root():
    return {"message": "Climate Hazard API is running successfully!"}

@app.get("/api/analyze")
def analyze_climate_hazard(
    lat: float, 
    lon: float, 
    start_year: int = 1990, 
    end_year: int = 2020, 
    hazard_type: str = "heatwave"
):
    cache_key = f"{lat}_{lon}_{start_year}_{end_year}_{hazard_type}"
    
    # 1. Check cache if available
    if cache_collection is not None:
        try:
            cached_data = cache_collection.find_one({"cache_key": cache_key})
            if cached_data:
                cached_data["_id"] = str(cached_data["_id"])
                return {"source": "cache", "data": cached_data["result"]}
        except Exception:
            pass

    # 2. Fetch from Open-Meteo Historical Weather API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "temperature_2m_max"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to weather API: {str(e)}")
    
    daily_data = data.get("daily", {})
    times = daily_data.get("time", [])
    tmax = daily_data.get("temperature_2m_max", [])

    if not times or not tmax:
        raise HTTPException(status_code=404, detail="No weather data available for these coordinates.")

    # 3. Process into Pandas DataFrame
    df = pd.DataFrame({"date": times, "tmax": tmax})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df.dropna(subset=['tmax'])

    if df.empty:
        raise HTTPException(status_code=400, detail="Insufficient valid temperature data for this region.")

    # 4. Climate Logic: Heatwave Definition (3+ consecutive days > 95th percentile)
    p95 = np.percentile(df['tmax'], 95)
    df['is_hot'] = df['tmax'] > p95
    df['consecutive_group'] = (~df['is_hot']).cumsum()
    
    heatwave_spans = df[df['is_hot']].groupby('consecutive_group').filter(lambda x: len(x) >= 3)
    yearly_counts = heatwave_spans.groupby('year').size().reindex(range(start_year, end_year + 1), fill_value=0)
    
    years = [int(yr) for yr in yearly_counts.index]
    frequencies = [int(freq) for freq in yearly_counts.values]

    # 5. Trend Analysis via Linear Regression
    slope, intercept, r_value, p_value, std_err = linregress(years, frequencies)
    total_change_pct = ((frequencies[-1] - frequencies[0]) / max(frequencies[0], 1)) * 100

    trend_direction = "increased" if slope > 0 else "decreased"
    insight_text = (
        f"In this region, heatwave events have {trend_direction} "
        f"by approximately {abs(total_change_pct):.1f}% between {start_year} and {end_year}."
    )

    result_payload = {
        "region": {"lat": float(lat), "lon": float(lon)},
        "threshold_95th_temp_c": round(float(p95), 2),
        "years": years,
        "frequencies": frequencies,
        "trend": {
            "slope": round(float(slope), 4),
            "r_squared": round(float(r_value**2), 4),
            "p_value": round(float(p_value), 4)
        },
        "summary_insight": insight_text
    }

    # Save to MongoDB cache if available
    if cache_collection is not None:
        try:
            cache_collection.insert_one({"cache_key": cache_key, "result": result_payload})
        except Exception:
            pass

    return {"source": "api", "data": result_payload}