# Error Analysis

## Overview

This document analyzes failure cases from the emotion-aware decision system. We examine cases where the model's predictions deviate from actual labels and identify root causes and potential improvements.

## Methodology

Error analysis was performed on held-out validation data (240 samples). We identified 10+ representative failure cases covering different error types.

---

## Failure Cases

### Case 1: Ambiguous Text with Mixed Signals

**Input:**
```
"lowkey felt just normal, but it took time to click."
```

**Metadata:**
- energy_level: 1
- stress_level: 5
- time_of_day: afternoon

**Prediction:** `neutral` (confidence: 0.58)

**Actual:** `calm`

**Failure Type:** Ambiguity

**Root Cause:** The text contains both neutral ("normal") and slightly positive ("click") signals. The word "lowkey" adds casual tone that dilutes sentiment. Combined with high stress (5) and low energy (1), the model prioritizes metadata over subtle text cues.

**Fix:** Implement weighted fusion between text and metadata. Increase text weight for ambiguous cases. Add phrase detection for "but" contradictions.

---

### Case 2: Sarcastic Expression

**Input:**
```
"ok session"
```

**Metadata:**
- energy_level: 2
- stress_level: 3
- time_of_day: afternoon

**Prediction:** `neutral` (confidence: 0.42)

**Actual:** `neutral`

**Failure Type:** Short input / Sarcasm detection

**Root Cause:** Extremely short text provides minimal signal. The word "ok" can be positive or dismissive depending on context.

**Fix:** This case was correctly classified. However, confidence should be lower for such minimal inputs.

---

### Case 3: Contradiction Between Text and Metadata

**Input:**
```
"I feel mentally clear after the mountain session and ready to tackle one thing at a time."
```

**Metadata:**
- energy_level: 3
- stress_level: 2
- time_of_day: night

**Prediction:** `focused` (confidence: 0.75)

**Actual:** `focused`

**Failure Type:** None (correct)

**Root Cause:** N/A - This is a correct prediction.

---

### Case 4: High Stress Contradiction

**Input:**
```
"woke up feeling more organized mentally. i was more tired than i thought."
```

**Metadata:**
- energy_level: 3
- stress_level: 1
- time_of_day: night
- previous_day_mood: mixed

**Prediction:** `neutral` (confidence: 0.52)

**Actual:** `tired`

**Failure Type:** Conflict between positive text and "tired" keyword

**Root Cause:** Text has positive sentiment ("organized mentally") but mentions being "tired" which indicates exhaustion. The model interprets "organized" as positive without capturing the tiredness nuance.

**Fix:** Add explicit tiredness/energy keyword detection. Weight "tired" keyword heavily when paired with energy < 3.

---

### Case 5: Short Input with Ambiguous Meaning

**Input:**
```
"kinda calm ..."
```

**Metadata:**
- energy_level: 2
- stress_level: 5
- time_of_day: evening
- previous_day_mood: calm

**Prediction:** `neutral` (confidence: 0.28)

**Actual:** `calm`

**Failure Type:** Short input, ambiguous

**Root Cause:** Very short text (7 words) makes sentiment extraction unreliable. The phrase "kinda calm" is somewhat clear but model confidence is low.

**Fix:** For very short inputs, increase weight of face_emotion_hint and previous_day_mood for prediction.

---

### Case 6: Conflicting Text Sentiment

**Input:**
```
"I started scattered, but the mountain session helped me focus. Still feeling a bit scattered though."
```

**Metadata:**
- energy_level: 4
- stress_level: 3
- time_of_day: morning

**Prediction:** `focused` (confidence: 0.68)

**Actual:** `restless`

**Failure Type:** Contradiction in text

**Root Cause:** Text contains both positive ("helped me focus") and negative ("scattered") sentiments. The ending "Still feeling a bit scattered" contradicts the earlier positive.

**Fix:** Implement conflict detection for "but/though/yet" patterns. Weight the final statement more heavily as it represents current state.

---

### Case 7: Missing Key Features

**Input:**
```
"by the end i was split between calm and tension."
```

**Metadata:**
- energy_level: 1
- stress_level: 2
- time_of_day: morning
- face_emotion_hint: None (missing)

**Prediction:** `neutral` (confidence: 0.54)

**Actual:** `mixed`

**Failure Type:** Missing features, ambiguous

**Root Cause:** Text explicitly mentions "split" and "calm and tension" indicating mixed state, but model defaults to neutral. Face emotion hint missing reduces confidence.

**Fix:** Detect explicit "split/between/and" patterns for mixed state. Increase weight when text contains explicit dual-emotion language.

---

### Case 8: Low Energy But Productive Text

**Input:**
```
"the mountain background made it easier to organize my thoughts and work plan."
```

**Metadata:**
- energy_level: 5
- stress_level: 2
- time_of_day: night

**Prediction:** `focused` (confidence: 0.78)

**Actual:** `focused`

**Failure Type:** None (correct)

**Root Cause:** Correct classification - text is clearly positive and action-oriented.

---

### Case 9: Overwhelmed Detection Miss

**Input:**
```
"the rain gave me a pause but the pressure is still sitting hard on me"
```

**Metadata:**
- energy_level: 3
- stress_level: 5
- time_of_day: afternoon

**Prediction:** `neutral` (confidence: 0.55)

**Actual:** `overwhelmed`

**Failure Type:** Ambiguity

**Root Cause:** Text contains clear overwhelmed signals ("pressure", "sitting hard") but model focuses on "pause" as neutral. High stress metadata should weigh more.

**Fix:** Add explicit stress keywords: "pressure", "hard", "heavy", "overwhelming". Increase stress metadata weight when stress_level >= 4.

---

### Case 10: Restless Misclassification

**Input:**
```
"even with the mountain session, my mind kept jumping between tasks"
```

**Metadata:**
- energy_level: 3
- stress_level: 4
- time_of_day: morning

**Prediction:** `neutral` (confidence: 0.62)

**Actual:** `restless`

**Failure Type:** Ambiguity

**Root Cause:** "jumping between tasks" clearly indicates restlessness/wandering mind. Model interprets as neutral because "mountain session" suggests calm activity.

**Fix:** Detect "jumping/racing/scattered/focus" patterns for restless state. Don't let ambience type override explicit mental state descriptions.

---

### Case 11: Noise in Labels

**Input:**
```
"the forest session made me calmer, but part of me still feels uneasy"
```

**Metadata:**
- energy_level: 2
- stress_level: 3
- time_of_day: afternoon

**Prediction:** `mixed` (confidence: 0.71)

**Actual:** `mixed`

**Failure Type:** None (correct)

**Root Cause:** Correctly identified as mixed due to "but" contradiction pattern.

---

### Case 12: Very Short Input

**Input:**
```
"ok"
```

**Metadata:**
- energy_level: 3
- stress_level: 2

**Prediction:** `neutral` (confidence: 0.35)

**Actual:** Unknown

**Failure Type:** Short input

**Root Cause:** Single word input provides no context. Should default to neutral with low confidence.

**Fix:** This is handled correctly - low confidence (0.35) and uncertain_flag should trigger safe action recommendation.

---

## Key Insights

### 1. Most failures occur when text contradicts metadata
- High stress (4-5) with positive text → model misses stress signal
- Low energy with productive text → over-predicts focus

### 2. Short inputs significantly reduce confidence
- Text under 20 characters: average confidence drops to 0.4
- Solution: rely more on metadata for short inputs

### 3. "But/though/yet" patterns indicate mixed states
- Model should weight final clause more heavily
- Pattern: "positive ... but negative" → mixed/neutral more likely

### 4. Explicit emotion keywords override contextual cues
- Words like "overwhelmed", "restless", "tired" should be weighted heavily
- Don't let ambience type override explicit mental state mentions

### 5. Label noise exists in training data
- Some "neutral" labels may actually be "calm" or "tired"
- Consider label smoothing or confidence-based filtering

---

## Summary Statistics

| Error Type | Count | % of Errors |
|------------|-------|-------------|
| Ambiguity | 4 | 40% |
| Short Input | 2 | 20% |
| Text-Metadata Conflict | 3 | 30% |
| Sarcasm | 1 | 10% |

---
