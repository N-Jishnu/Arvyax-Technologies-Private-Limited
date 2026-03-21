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

---

## Features

### 1. Emotion Classification
- Uses TF-IDF vectorization + Gradient Boosting
- Handles 6 emotional states: calm, focused, overwhelmed, restless, neutral, mixed
- Processes noisy, sarcastic, and short text inputs

### 2. Intensity Prediction
- Regression model predicting 1-5 scale intensity
- Uses same feature set as emotion classification

### 3. Decision Engine
Scoring-based action recommendation:
```
score(action) = w1*state_alignment + w2*intensity_factor + w3*energy_match 
              + w4*stress_handling + w5*time_compatibility
```

Available actions: box_breathing, journaling, grounding, deep_work, yoga, sound_therapy, light_planning, rest, movement, pause

### 4. Uncertainty Modeling
- Model confidence from probability distributions
- Input quality score based on text length
- Conflict detection between text and metadata

---

## Installation

```bash
pip install pandas numpy scikit-learn
```

---

## Usage

### Run the System

```python
from src.emotion_system import EmotionAwareSystem
import pandas as pd

# Load data
train_df = pd.read_csv('Sample_arvyax_reflective_training.csv')
test_df = pd.read_csv('arvyax_test_inputs.csv')

# Create and train system
system = EmotionAwareSystem()
system.fit(train_df)

# Make predictions
predictions = system.predict(test_df)
predictions.to_csv('predictions.csv', index=False)
```

### Run from Command Line

```bash
python src/emotion_system.py
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

---

## Model Performance

| Metric | Value |
|--------|-------|
| Emotion Classification Accuracy | 57% |
| Intensity RMSE | 1.525 |
| Average Confidence | 0.71 |
| Uncertain Predictions | 16/120 |

---

## Feature Importance

| Feature Type | Contribution |
|--------------|--------------|
| journal_text | 60-70% |
| stress_level | 10-15% |
| energy_level | 10-15% |
| sleep_hours | 5-10% |
| time_of_day | 5-10% |

---

## Ablation Study Results

| Model | Accuracy |
|-------|----------|
| Full (Text + Metadata) | 57.1% |
| Text-Only | 57.5% |

The ablation study shows minimal difference between text-only and full models, suggesting text features carry most predictive power.

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
├── ERROR_ANALYSIS.md        # Error case analysis
├── EDGE_PLAN.md            # Edge deployment plan
├── Sample_arvyax_reflective_training.csv  # Training data
├── arvyax_test_inputs.csv   # Test data
└── src/
    └── emotion_system.py    # Main system code
```

---

## Key Implementation Details

### Decision Logic

The decision engine uses a weighted scoring system:
- State alignment: How well action matches predicted emotional state
- Intensity factor: Action suitability based on intensity level
- Energy match: Action appropriateness for energy level
- Stress handling: Action effectiveness for stress level
- Time compatibility: Action suitability for time of day

### Uncertainty Detection

Uncertain predictions occur when:
- Confidence < 0.45
- Text length < 15 characters
- Critical metadata (energy, stress) missing

For uncertain cases, the system defaults to safe actions: pause, box_breathing, light_planning

---

## Robustness Features

1. **Short Text Handling**: Defaults to neutral with low confidence
2. **Missing Data**: Uses mean imputation for missing values
3. **Contradiction Detection**: Weighted fusion for text-metadata conflicts

---
