from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

DATA_FILE = os.path.join(script_dir, 'data', 'Sample_arvyax_reflective_training.csv')

app = FastAPI(title="Emotion-Aware Decision System API")

class JournalInput(BaseModel):
    id: int
    journal_text: str
    ambience_type: Optional[str] = None
    duration_min: Optional[float] = None
    sleep_hours: Optional[float] = None
    energy_level: Optional[float] = None
    stress_level: Optional[float] = None
    time_of_day: Optional[str] = "morning"
    previous_day_mood: Optional[str] = None
    face_emotion_hint: Optional[str] = None
    reflection_quality: Optional[str] = "vague"

class BatchInput(BaseModel):
    entries: List[JournalInput]

print("Loading Emotion-Aware System...")
print(f"Looking for data at: {DATA_FILE}")
system = None
try:
    from src.emotion_system import EmotionAwareSystem
    train_df = pd.read_csv(DATA_FILE)
    system = EmotionAwareSystem()
    system.fit(train_df)
    print("System loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    import traceback
    traceback.print_exc()
    system = None

@app.get("/")
def root():
    return {
        "name": "Emotion-Aware Decision System API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": system is not None
    }

@app.post("/predict")
def predict(input_data: JournalInput):
    if system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([input_data.dict()])
        predictions = system.predict(df)
        
        result = predictions.iloc[0].to_dict()
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(batch_input: BatchInput):
    if system is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([entry.dict() for entry in batch_input.entries])
        predictions = system.predict(df)
        
        results = predictions.to_dict(orient='records')
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/states")
def get_states():
    return {"states": EmotionAwareSystem.STATES}

@app.get("/actions")
def get_actions():
    return {"actions": EmotionAwareSystem.ACTIONS}

@app.get("/timing")
def get_timing():
    return {"timing_options": EmotionAwareSystem.TIMING}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)