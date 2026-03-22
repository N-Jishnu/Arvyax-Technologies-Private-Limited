# Emotion-Aware Decision System

A robust system for real-world mental state understanding that takes noisy, imperfect user inputs and outputs emotional understanding, actionable decisions, uncertainty estimation, and supportive responses.

---

## Overview

This system implements:
- **Part 1**: Emotional state prediction (multi-class classification)
- **Part 2**: Intensity prediction (regression)
- **Part 3**: Decision engine with scoring-based reasoning
- **Part 4**: Uncertainty modeling and confidence estimation
- **Part 5**: Feature importance analysis
- **Part 6**: Ablation study (text-only vs. full model)
- **Part 7**: Error analysis with failure case documentation
- **Part 8**: Edge deployment optimization
- **Part 9**: Robustness (short text, missing data, conflicts)

---

## Features

### 1. Emotion Classification
- Uses TF-IDF vectorization + Gradient Boosting
- Dynamic feature fusion (text/metadata weighting)
- Handles 6 emotional states: calm, focused, overwhelmed, restless, neutral, mixed
- Neutral penalty to reduce over-prediction
- Restless keyword override for better detection

### 2. Intensity Prediction
- Regression model with calibrated bins (1-5 scale)
- Sample weighting for balanced predictions
- Dynamic intensity thresholds: [1.8, 2.6, 3.4, 4.2]

### 3. Decision Engine
Scoring-based action recommendation:
```
score(action) = w1*state_alignment + w2*intensity_factor + w3*energy_match 
              + w4*stress_handling + w5*time_compatibility
```

Available actions: box_breathing, journaling, grounding, deep_work, yoga, sound_therapy, light_planning, rest, movement, pause

### 4. Uncertainty Modeling
- Model confidence from probability distributions
- True conflict detection (text sentiment vs metadata stress)
- Text length-based quality scoring

### 5. Feature Importance
- Named feature importance with text and metadata features
- Top features: "but not", "duration", "stress", "sentiment", "face_emotion"

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage


### To Run the API

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### To Run the UI

```bash
streamlit run app.py
```

### Python API

```python
from src.emotion_system import EmotionAwareSystem
import pandas as pd

train_df = pd.read_csv('data/Sample_arvyax_reflective_training.csv')
system = EmotionAwareSystem()
system.fit(train_df)

predictions = system.predict(test_df)
predictions.to_csv('predictions.csv', index=False)
```

---

## Output Format

The `predictions.csv` contains:

| Column | Description |
|--------|-------------|
| id | Sample identifier |
| predicted_state | Emotional state (calm, focused, etc.) |
| predicted_intensity | Intensity 1-5 |
| confidence | Model confidence (0-1) |
| uncertain_flag | 1 if uncertain, 0 otherwise |
| what_to_do | Recommended action |
| when_to_do | Timing recommendation |
| decision_reason | Scoring breakdown |
| confidence_reason | Confidence explanation |
| supportive_message | Human-like supportive message |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Emotion Classification Accuracy | **60.8%** |
| Intensity RMSE | 1.571 |
| Macro F1-Score | 0.60 |
| Average Confidence | 0.72 |
| Uncertain Predictions | 26/120 |

### Per-Class Performance

| State | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| calm | 0.72 | 0.67 | 0.69 |
| focused | 0.69 | 0.59 | 0.63 |
| mixed | 0.61 | 0.61 | 0.61 |
| neutral | 0.76 | 0.57 | 0.65 |
| overwhelmed | 0.50 | 0.57 | 0.53 |
| restless | 0.41 | 0.66 | 0.51 |

---

## Ablation Study Results

| Model | Accuracy |
|-------|----------|
| Full (Text + Metadata) | **60.8%** |
| Text-Only | 57.5% |
| Improvement | +3.3% |

The full model outperforms text-only by 3.3%, demonstrating that metadata improves predictions.

---

## Feature Importance (Top 15)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | but not | 0.0312 |
| 2 | duration | 0.0308 |
| 3 | stress | 0.0257 |
| 4 | nothing | 0.0240 |
| 5 | sentiment | 0.0237 |
| 6 | tasks | 0.0228 |
| 7 | face_emotion | 0.0219 |
| 8 | energy | 0.0214 |
| 9 | sleep | 0.0213 |
| 10 | drained | 0.0190 |

---

## Edge Deployment

The system meets edge deployment constraints:

| Factor | Target | Actual |
|--------|--------|--------|
| Model Size | <100MB | ~25MB |
| Latency | <200ms | 50-80ms |
| Memory | <500MB | ~200MB |

---

## File Structure

```
.
├── predictions.csv          # Output predictions
├── README.md               # This file
├── ERROR_ANALYSIS.md       # Error case analysis
├── EDGE_PLAN.md           # Edge deployment plan
├── main.py                # FastAPI server
├── app.py                 # Streamlit UI
├── requirements.txt        # Dependencies
├── data/
│   ├── Sample_arvyax_reflective_training.csv
│   └── Arvyax_test_inputs.csv
└── src/
    └── emotion_system.py   # Core system
```

---

## Bonus Features

| Feature | Description |
|---------|-------------|
| Supportive Message Generator | Human-like explanations |
| Label Noise Handling | Smoothing, filtering, disagreement detection |
| FastAPI Local API | REST API with endpoints |
| Confidence Explanation | `confidence_reason` field |
| Decision Reason Trace | `decision_reason` breakdown |
| Streamlit UI | Interactive web interface |

---

## Key Implementation Details

### Dynamic Feature Fusion
Text/metadata weighting based on input length:
- Short text (<20 chars): 40% text, 60% metadata
- Normal text: 70% text, 30% metadata

### Decision Logic
The decision engine uses a weighted scoring system:
- State alignment: How well action matches predicted emotional state
- Intensity factor: Action suitability based on intensity level
- Energy match: Action appropriateness for energy level
- Stress handling: Action effectiveness for stress level
- Time compatibility: Action suitability for time of day

### Uncertainty Detection
Uncertain predictions occur when:
- Confidence < 0.5
- Text length < 15 characters
- Conflict > 0.6
- Critical metadata missing

For uncertain cases, the system defaults to safe actions: pause, box_breathing, light_planning

---

## Robustness Features

1. **Short Text Handling**: Defaults to neutral/mixed with low confidence
2. **Missing Data**: Uses mean imputation for missing values
3. **Contradiction Detection**: Text sentiment vs metadata stress
4. **Restless Keyword Override**: Direct detection from text keywords
5. **Neutral Penalty**: Reduces over-prediction of neutral state

---
