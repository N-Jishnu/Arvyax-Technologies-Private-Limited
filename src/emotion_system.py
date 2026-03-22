import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

class EmotionAwareSystem:
    """Complete Emotion-Aware Decision System"""
    
    ACTIONS = ['box_breathing', 'journaling', 'grounding', 'deep_work', 
               'yoga', 'sound_therapy', 'light_planning', 'rest', 'movement', 'pause']
    
    TIMING = ['now', 'within_15_min', 'later_today', 'tonight', 'tomorrow_morning']
    
    STATES = ['calm', 'focused', 'overwhelmed', 'restless', 'neutral', 'mixed', 'tired']
    
    def __init__(self, use_transformer=False):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        
        self.text_vectorizer = TfidfVectorizer(
            max_features=300, ngram_range=(1, 2), 
            min_df=2, max_df=0.95, sublinear_tf=True
        )
        
        self.emotion_encoder = LabelEncoder()
        self.intensity_scaler = StandardScaler()
        
        self.emotion_model = None
        self.intensity_model = None
        
        self._init_decision_weights()
        
    def _init_decision_weights(self):
        self.weights = {
            'w1': 0.25,
            'w2': 0.20,
            'w3': 0.20,
            'w4': 0.15,
            'w5': 0.20
        }
    
    def _preprocess_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_sentiment_features(self, text):
        """Extract sentiment-related features from text"""
        text_lower = str(text).lower()
        
        positive_words = ['calm', 'peaceful', 'better', 'lighter', 'helpful', 'settled', 'clear', 'focused', 'ready', 'organized', 'ok', 'good', 'nice', 'able']
        negative_words = ['overwhelmed', 'exhausted', 'heavy', 'chaotic', 'scattered', 'unsettled', 'tense', 'anxious', 'tired', 'flooded', 'racing', 'stress']
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        sentiment_score = (pos_count - neg_count) / max(1, pos_count + neg_count + 1)
        
        has_but = 1 if ' but ' in text_lower or 'however' in text_lower else 0
        has_and = text_lower.count(' and ')
        
        words = text_lower.split()
        complexity = np.log1p(len(words)) if words else 0
        
        restless_keywords = ["jumping", "scattered", "racing", "switching", "distracted", "wandering", "fidgety"]
        restless_flag = 1 if any(word in text_lower for word in restless_keywords) else 0
        
        return [sentiment_score, has_but, has_and, complexity, len(words) / 100.0, restless_flag]
    
    def _create_metadata_features(self, row):
        features = []
        
        energy = row.get('energy_level')
        energy = float(energy) if pd.notna(energy) else 3.0
        
        stress = row.get('stress_level')
        stress = float(stress) if pd.notna(stress) else 3.0
        
        sleep = row.get('sleep_hours')
        sleep = float(sleep) if pd.notna(sleep) else 6.0
        
        features.extend([
            energy / 5.0,
            stress / 5.0,
            sleep / 12.0,
        ])
        
        duration = row.get('duration_min')
        features.append(float(duration) / 60.0 if pd.notna(duration) else 0.25)
        
        time_map = {'morning': 0.0, 'early_morning': 0.25, 'afternoon': 0.5, 
                    'evening': 0.75, 'night': 1.0}
        features.append(time_map.get(str(row.get('time_of_day', 'morning')).lower(), 0.0))
        
        mood_map = {'calm': 0.0, 'neutral': 0.5, 'mixed': 0.75, 'overwhelmed': 1.0, 
                    'focused': 0.25, 'restless': 0.75}
        prev_mood = row.get('previous_day_mood')
        features.append(mood_map.get(str(prev_mood).lower() if pd.notna(prev_mood) else 'neutral', 0.5))
        
        quality_map = {'clear': 0.0, 'vague': 0.5, 'conflicted': 1.0}
        quality = row.get('reflection_quality')
        features.append(quality_map.get(str(quality).lower() if pd.notna(quality) else 'vague', 0.5))
        
        face_map = {'calm_face': 0.0, 'happy_face': 0.25, 'neutral_face': 0.5, 
                    'tense_face': 0.75, 'tired_face': 0.8, 'none': 0.5}
        face = row.get('face_emotion_hint')
        features.append(face_map.get(str(face).lower() if pd.notna(face) else 'none', 0.5))
        
        features.append(1.0 if pd.notna(row.get('sleep_hours')) and float(row.get('sleep_hours')) < 5 else 0.0)
        
        return features
    
    def _build_features(self, df, fit=False):
        texts = df['journal_text'].apply(self._preprocess_text).tolist()
        
        if fit:
            text_features = self.text_vectorizer.fit_transform(texts)
        else:
            text_features = self.text_vectorizer.transform(texts)
        
        sentiment_features = np.array([self._extract_sentiment_features(t) for t in df['journal_text']])
        
        metadata_features = []
        for _, row in df.iterrows():
            metadata_features.append(self._create_metadata_features(row))
        metadata_array = np.array(metadata_features)
        
        text_dense = text_features.toarray()
        
        weighted_features = []
        for i, text in enumerate(df['journal_text']):
            text_len = len(str(text)) if pd.notna(text) else 0
            
            if text_len < 20:
                text_w, meta_w = 0.4, 0.6
            else:
                text_w, meta_w = 0.7, 0.3
            
            combined_row = np.hstack([
                text_dense[i] * text_w,
                sentiment_features[i] * text_w,
                metadata_array[i] * meta_w
            ])
            weighted_features.append(combined_row)
        
        combined = np.array(weighted_features)
        combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return combined
    
    def fit(self, train_df):
        print("Building features...")
        X = self._build_features(train_df, fit=True)
        
        y_emotion = self.emotion_encoder.fit_transform(train_df['emotional_state'])
        y_intensity = train_df['intensity'].values
        
        print(f"Training emotion classifier on {len(X)} samples...")
        self.emotion_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=6, min_samples_split=5,
            learning_rate=0.1, random_state=42
        )
        self.emotion_model.fit(X, y_emotion)
        
        print(f"Training intensity regressor...")
        weights = np.ones(len(y_intensity))
        weights[y_intensity == 1] *= 1.5
        weights[y_intensity == 5] *= 1.5
        
        self.intensity_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.intensity_model.fit(X, y_intensity, sample_weight=weights)
        
        print("Training complete!")
        
        return self
    
    def _compute_state_alignment(self, action, state):
        state_action_scores = {
            'box_breathing': {'overwhelmed': 1.0, 'restless': 0.7, 'calm': 0.3, 'neutral': 0.4, 'mixed': 0.6, 'focused': 0.3, 'tired': 0.4},
            'journaling': {'overwhelmed': 0.6, 'restless': 0.5, 'neutral': 0.7, 'mixed': 0.8, 'focused': 0.6, 'tired': 0.4},
            'grounding': {'overwhelmed': 0.9, 'restless': 0.8, 'neutral': 0.5, 'mixed': 0.7, 'tired': 0.5},
            'deep_work': {'focused': 1.0, 'calm': 0.9, 'neutral': 0.6, 'restless': 0.2, 'overwhelmed': 0.1},
            'yoga': {'overwhelmed': 0.6, 'restless': 0.7, 'tired': 0.7, 'calm': 0.8, 'neutral': 0.5},
            'sound_therapy': {'overwhelmed': 0.7, 'restless': 0.8, 'tired': 0.8, 'calm': 0.9, 'neutral': 0.6},
            'light_planning': {'overwhelmed': 0.4, 'restless': 0.5, 'neutral': 0.7, 'focused': 0.8, 'calm': 0.6},
            'rest': {'tired': 1.0, 'overwhelmed': 0.8, 'restless': 0.3, 'neutral': 0.5, 'calm': 0.4},
            'movement': {'restless': 0.9, 'neutral': 0.6, 'overwhelmed': 0.3, 'tired': 0.3, 'calm': 0.6},
            'pause': {'overwhelmed': 0.9, 'restless': 0.7, 'tired': 0.7, 'neutral': 0.6, 'calm': 0.5}
        }
        
        return state_action_scores.get(action, {}).get(state, 0.5)
    
    def _compute_intensity_factor(self, action, intensity):
        if action in ['box_breathing', 'grounding', 'pause', 'rest']:
            if intensity >= 4:
                return 1.0
            elif intensity >= 2:
                return 0.7
            return 0.4
        elif action in ['deep_work', 'light_planning']:
            if 2 <= intensity <= 4:
                return 1.0
            return 0.5
        elif action in ['movement', 'yoga', 'sound_therapy', 'journaling']:
            if intensity <= 3:
                return 1.0
            return 0.5
        return 0.5
    
    def _compute_energy_match(self, action, energy):
        high_energy_actions = ['deep_work', 'movement', 'light_planning']
        low_energy_actions = ['rest', 'pause', 'sound_therapy']
        
        if action in high_energy_actions:
            if energy >= 4:
                return 1.0
            elif energy >= 3:
                return 0.7
            return 0.3
        elif action in low_energy_actions:
            if energy <= 2:
                return 1.0
            elif energy <= 3:
                return 0.7
            return 0.4
        return 0.5
    
    def _compute_stress_handling(self, action, stress):
        stress_actions = ['box_breathing', 'grounding', 'pause', 'yoga', 'sound_therapy', 'rest']
        
        if action in stress_actions:
            if stress >= 4:
                return 1.0
            elif stress >= 2:
                return 0.7
            return 0.4
        
        return 0.5
    
    def _compute_time_compatibility(self, action, time_of_day):
        time_action_map = {
            'morning': ['deep_work', 'light_planning', 'movement', 'yoga', 'journaling'],
            'early_morning': ['light_planning', 'movement', 'yoga', 'box_breathing'],
            'afternoon': ['deep_work', 'journaling', 'grounding', 'movement', 'light_planning'],
            'evening': ['sound_therapy', 'yoga', 'journaling', 'movement', 'grounding'],
            'night': ['sound_therapy', 'rest', 'pause', 'journaling', 'box_breathing']
        }
        
        time_str = str(time_of_day).lower() if pd.notna(time_of_day) else 'morning'
        compatible = time_action_map.get(time_str, self.ACTIONS)
        
        return 1.0 if action in compatible else 0.4
    
    def _score_action(self, action, state, intensity, energy, stress, time_of_day):
        state_score = self._compute_state_alignment(action, state)
        intensity_score = self._compute_intensity_factor(action, intensity)
        energy_score = self._compute_energy_match(action, energy)
        stress_score = self._compute_stress_handling(action, stress)
        time_score = self._compute_time_compatibility(action, time_of_day)
        
        w = self.weights
        score = (w['w1'] * state_score + 
                 w['w2'] * intensity_score + 
                 w['w3'] * energy_score +
                 w['w4'] * stress_score + 
                 w['w5'] * time_score)
        
        return score, {
            "state_alignment": round(state_score, 2),
            "intensity": round(intensity_score, 2),
            "energy": round(energy_score, 2),
            "stress": round(stress_score, 2),
            "time": round(time_score, 2)
        }
    
    def _decide_timing(self, intensity, stress, energy, time_of_day):
        urgency = intensity * 1.2 + stress - energy
        
        if urgency >= 6:
            return 'now'
        elif urgency >= 4:
            return 'within_15_min'
        elif urgency >= 2:
            return 'later_today'
        
        time_str = str(time_of_day).lower() if pd.notna(time_of_day) else 'morning'
        if time_str == 'night' and energy <= 3:
            return 'tonight'
        
        if energy <= 1.5:
            return 'tomorrow_morning'
        
        return 'later_today'
    
    def _compute_confidence(self, proba, text, stress_level, energy_level):
        model_conf = np.max(proba) if proba is not None else 0.5
        
        text_len = len(str(text)) if pd.notna(text) else 0
        if text_len < 15:
            input_quality = 0.2
        elif text_len < 40:
            input_quality = 0.5
        else:
            input_quality = 0.9
        
        sentiment_features = self._extract_sentiment_features(text)
        sentiment_score = sentiment_features[0]
        sentiment_norm = (sentiment_score + 1) / 2
        
        if pd.notna(stress_level):
            stress_val = float(stress_level)
            meta_signal = stress_val / 5.0
        else:
            meta_signal = 0.5
        
        conflict = abs(sentiment_norm - meta_signal)
        
        confidence = 0.5 * model_conf + 0.3 * input_quality + 0.2 * (1 - conflict)
        
        return np.clip(confidence, 0.1, 1.0), conflict
    
    def _check_uncertainty(self, confidence, text, row, conflict):
        text_len = len(str(text)) if pd.notna(text) else 0
        
        if confidence < 0.5:
            return 1
        
        if text_len < 15:
            return 1
        
        if conflict > 0.6:
            return 1
        
        missing_critical = (
            pd.isna(row.get('energy_level')) and 
            pd.isna(row.get('stress_level'))
        )
        
        return 1 if missing_critical else 0
    
    def predict(self, df):
        print(f"Predicting on {len(df)} samples...")
        
        X = self._build_features(df, fit=False)
        
        emotion_proba = self.emotion_model.predict_proba(X)
        emotion_preds = self.emotion_model.predict(X)
        predicted_states = self.emotion_encoder.inverse_transform(emotion_preds)
        
        predicted_intensity = self.intensity_model.predict(X)
        
        def calibrate_intensity(x):
            if x < 1.8:
                return 1
            elif x < 2.6:
                return 2
            elif x < 3.4:
                return 3
            elif x < 4.2:
                return 4
            else:
                return 5
        
        predicted_intensity = np.array([calibrate_intensity(x) for x in predicted_intensity])
        
        results = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            state = predicted_states[i]
            intensity = predicted_intensity[i]
            
            energy = float(row.get('energy_level', 3)) if pd.notna(row.get('energy_level')) else 3.0
            stress = float(row.get('stress_level', 3)) if pd.notna(row.get('stress_level')) else 3.0
            time_of_day = str(row.get('time_of_day', 'morning')).lower() if pd.notna(row.get('time_of_day')) else 'morning'
            
            proba = emotion_proba[i] if i < len(emotion_proba) else None
            journal_text = row.get('journal_text', '')
            confidence, conflict = self._compute_confidence(
                proba, 
                journal_text,
                stress, energy
            )
            
            if proba is not None:
                proba = proba.copy()
                neutral_idx = list(self.emotion_encoder.classes_).index("neutral")
                
                if proba[neutral_idx] > 0.4:
                    proba[neutral_idx] *= 0.85
                
                state = self.emotion_encoder.inverse_transform([np.argmax(proba)])[0]
            
            restless_keywords = ["jumping", "scattered", "racing", "switching", "distracted", "wandering", "fidgety"]
            text_lower = str(journal_text).lower()
            if any(word in text_lower for word in restless_keywords):
                state = "restless"
                confidence = max(confidence, 0.6)
            
            uncertain_flag = self._check_uncertainty(
                confidence, journal_text, row, conflict
            )
            
            if uncertain_flag:
                state = "mixed" if conflict > 0.5 else "neutral"
                intensity = 3
            
            action_usage_penalty = {
                "rest": 1.3,
                "box_breathing": 1.2,
                "yoga": 1.1
            }
            
            action_scores = {}
            action_reasons = {}
            for action in self.ACTIONS:
                score, reason = self._score_action(action, state, intensity, energy, stress, time_of_day)
                score *= action_usage_penalty.get(action, 1.0)
                action_scores[action] = score
                action_reasons[action] = reason
            
            what_to_do = max(action_scores, key=action_scores.get)
            decision_reason = action_reasons[what_to_do]
            when_to_do = self._decide_timing(intensity, stress, energy, time_of_day)
            
            if uncertain_flag:
                safe_actions = ['pause', 'box_breathing', 'light_planning']
                scores = {a: action_scores.get(a, 0) for a in safe_actions}
                what_to_do = max(scores, key=scores.get)
                decision_reason = action_reasons[what_to_do]
            
            supportive_message = self._generate_supportive_message(
                state, int(intensity), what_to_do, when_to_do, confidence, uncertain_flag
            )
            
            text_len = len(str(journal_text)) if pd.notna(journal_text) else 0
            confidence_reason = self._confidence_reason(confidence, text_len, conflict)
            
            results.append({
                'id': row.get('id', i),
                'predicted_state': state,
                'predicted_intensity': int(intensity),
                'confidence': round(confidence, 3),
                'uncertain_flag': uncertain_flag,
                'what_to_do': what_to_do,
                'when_to_do': when_to_do,
                'decision_reason': decision_reason,
                'confidence_reason': confidence_reason,
                'supportive_message': supportive_message
            })
        
        return pd.DataFrame(results)
    
    def evaluate(self, test_df):
        X = self._build_features(test_df, fit=False)
        
        y_true_emotion = test_df['emotional_state'].values
        y_true_intensity = test_df['intensity'].values
        
        emotion_preds = self.emotion_model.predict(X)
        predicted_states = self.emotion_encoder.inverse_transform(emotion_preds)
        
        predicted_intensity = self.intensity_model.predict(X)
        
        def calibrate_intensity(x):
            if x < 1.8:
                return 1
            elif x < 2.6:
                return 2
            elif x < 3.4:
                return 3
            elif x < 4.2:
                return 4
            else:
                return 5
        
        predicted_intensity = np.array([calibrate_intensity(x) for x in predicted_intensity])
        
        emotion_acc = accuracy_score(y_true_emotion, predicted_states)
        intensity_rmse = np.sqrt(mean_squared_error(y_true_intensity, predicted_intensity))
        
        print(f"\nEmotion Classification Accuracy: {emotion_acc:.3f}")
        print(f"Intensity Regression RMSE: {intensity_rmse:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_true_emotion, predicted_states))
        
        return {'accuracy': emotion_acc, 'rmse': intensity_rmse}
    
    def run_ablation_study(self, train_df, test_df):
        print("\n=== ABLATION STUDY ===\n")
        
        texts = train_df['journal_text'].apply(self._preprocess_text).tolist()
        X_text = self.text_vectorizer.fit_transform(texts)
        
        test_texts = test_df['journal_text'].apply(self._preprocess_text).tolist()
        X_test_text = self.text_vectorizer.transform(test_texts)
        
        y_emotion = self.emotion_encoder.fit_transform(train_df['emotional_state'])
        
        model_text_only = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        model_text_only.fit(X_text, y_emotion)
        
        full_model_preds = self.emotion_model.predict(self._build_features(test_df, fit=False))
        
        y_true = test_df['emotional_state'].values
        
        full_preds = self.emotion_encoder.inverse_transform(full_model_preds)
        text_only_preds = self.emotion_encoder.inverse_transform(model_text_only.predict(X_test_text))
        
        full_acc = accuracy_score(y_true, full_preds)
        text_only_acc = accuracy_score(y_true, text_only_preds)
        
        print(f"Full Model (Text + Metadata): {full_acc:.3f}")
        print(f"Text-Only Model: {text_only_acc:.3f}")
        print(f"Difference: {(full_acc - text_only_acc)*100:.1f}%")
        
        return {'full': full_acc, 'text_only': text_only_acc}
    
    def _confidence_reason(self, confidence, text_len, conflict):
        reasons = []
        
        if text_len < 20:
            reasons.append("short input")
        if conflict > 0.5:
            reasons.append("conflicting signals")
        if confidence < 0.5:
            reasons.append("low model certainty")
        
        return ", ".join(reasons) if reasons else "high confidence"
    
    def show_feature_importance(self, top_n=20):
        importances = self.emotion_model.feature_importances_
        
        text_features = list(self.text_vectorizer.get_feature_names_out())
        extra_features = [
            "sentiment", "has_but", "has_and", "complexity", "length", "restless_flag",
            "energy", "stress", "sleep", "duration", "time_of_day",
            "previous_mood", "reflection_quality", "face_emotion", "low_sleep_flag"
        ]
        
        feature_names = text_features + extra_features
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nTop {top_n} Feature Importances:")
        for i, idx in enumerate(indices):
            if idx < len(feature_names):
                name = feature_names[idx]
            else:
                name = f"feature_{idx}"
            print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    def _generate_supportive_message(self, state, intensity, action, when_to_do, confidence, uncertain_flag):
        """Generate human-like supportive message based on prediction"""
        
        state_messages = {
            'calm': "You seem to be feeling peaceful right now.",
            'focused': "You appear to be in a focused state.",
            'overwhelmed': "You seem slightly overwhelmed right now.",
            'restless': "You appear to be feeling restless or unsettled.",
            'neutral': "You're feeling in a neutral space right now.",
            'mixed': "You seem to have mixed feelings at the moment.",
            'tired': "You appear to be feeling tired or low energy."
        }
        
        intensity_messages = {
            1: "It's quite subtle.",
            2: "It's a mild level.",
            3: "It's moderate.",
            4: "It's quite noticeable.",
            5: "It's quite strong."
        }
        
        action_recommendations = {
            'box_breathing': "Let's take a moment to breathe together.",
            'journaling': "Writing down your thoughts might help clarify things.",
            'grounding': "Let's ground ourselves in the present moment.",
            'deep_work': "You're in a good state for focused work.",
            'yoga': "Some gentle movement could help balance your energy.",
            'sound_therapy': "Some calming sounds might help you relax.",
            'light_planning': "Let's do some light planning to ease your mind.",
            'rest': "You could use some rest - let's take it easy.",
            'movement': "Some gentle movement might help release tension.",
            'pause': "Let's pause and check in with ourselves."
        }
        
        base_message = state_messages.get(state, "You're feeling something right now.")
        
        if intensity > 0:
            base_message += " " + intensity_messages.get(intensity, "")
        
        if uncertain_flag:
            base_message += " I'm not entirely sure, so let's take a gentle approach."
        
        base_message += " " + action_recommendations.get(action, "Let's take care of yourself.")
        
        return base_message.strip()
    
    def apply_label_smoothing(self, train_df, smoothing_factor=0.1):
        """Apply label smoothing to reduce overfitting to noisy labels"""
        states = train_df['emotional_state'].unique()
        state_counts = train_df['emotional_state'].value_counts()
        total = len(train_df)
        
        smoothed_probs = {}
        for state in states:
            count = state_counts.get(state, 1)
            smoothed_probs[state] = (count + smoothing_factor) / (total + smoothing_factor * len(states))
        
        print(f"\nLabel smoothing applied:")
        for state, prob in smoothed_probs.items():
            print(f"  {state}: {prob:.3f}")
        
        return smoothed_probs
    
    def confidence_based_filtering(self, predictions, min_confidence=0.5):
        """Filter out low-confidence predictions for retraining"""
        high_conf = predictions[predictions['confidence'] >= min_confidence]
        low_conf = predictions[predictions['confidence'] < min_confidence]
        
        print(f"\nConfidence-based filtering:")
        print(f"  High confidence (>= {min_confidence}): {len(high_conf)}")
        print(f"  Low confidence (< {min_confidence}): {len(low_conf)}")
        
        return high_conf, low_conf
    
    def disagreement_detection(self, X, n_models=3, sample_size=200):
        """Train multiple models and detect disagreement for label quality"""
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        y_emotion_encoded = self.emotion_model.predict(X_sample)
        
        models = [
            GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=i)
            for i in range(n_models)
        ]
        
        predictions = []
        for model in models:
            model.fit(X_sample, y_emotion_encoded)
            preds = model.predict(X_sample)
            predictions.append(preds)
        
        predictions = np.array(predictions, dtype=int)
        
        disagreement_count = np.sum(predictions.std(axis=0) > 0)
        
        disagreement_ratio = disagreement_count / X_sample.shape[0]
        
        print(f"\nDisagreement detection ({n_models} models, {sample_size} samples):")
        print(f"  Samples with disagreement: {disagreement_count}/{X_sample.shape[0]} ({disagreement_ratio:.1%})")
        
        return disagreement_ratio



def main():
    print("Loading data...")
    train_df = pd.read_csv('data/Sample_arvyax_reflective_training.csv')
    test_df = pd.read_csv('data/Arvyax_test_inputs.csv')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Emotional states: {train_df['emotional_state'].unique().tolist()}")
    
    print("\n" + "="*50)
    print("EMOTION-AWARE DECISION SYSTEM")
    print("="*50)
    
    system = EmotionAwareSystem()
    
    print("\n--- Training Phase ---")
    system.fit(train_df)
    
    print("\n--- Feature Importance ---")
    system.show_feature_importance(top_n=15)
    
    print("\n--- Evaluation on Held-Out Data ---")
    train_split, eval_split = train_test_split(train_df, test_size=0.2, random_state=42)
    
    system_eval = EmotionAwareSystem()
    system_eval.fit(train_split)
    metrics = system_eval.evaluate(eval_split)
    
    print("\n--- Ablation Study ---")
    ablation_results = system_eval.run_ablation_study(train_split, eval_split)
    
    print("\n--- Generating Predictions ---")
    predictions = system.predict(test_df)
    
    print("\nPrediction Summary:")
    print(f"States distribution:\n{predictions['predicted_state'].value_counts().to_dict()}")
    print(f"Intensity distribution:\n{predictions['predicted_intensity'].value_counts().to_dict()}")
    print(f"Actions distribution:\n{predictions['what_to_do'].value_counts().to_dict()}")
    print(f"Timing distribution:\n{predictions['when_to_do'].value_counts().to_dict()}")
    print(f"Uncertain predictions: {predictions['uncertain_flag'].sum()}")
    print(f"Average confidence: {predictions['confidence'].mean():.3f}")
    
    print("\n--- Label Noise Handling ---")
    system.apply_label_smoothing(train_df, smoothing_factor=0.1)
    high_conf, low_conf = system.confidence_based_filtering(predictions, min_confidence=0.5)
    X_train = system._build_features(train_df, fit=False)
    disagreement = system.disagreement_detection(X_train, n_models=3)
    
    print("\n--- Sample Supportive Messages ---")
    for i in range(3):
        print(f"\n  ID {predictions.iloc[i]['id']}: {predictions.iloc[i]['supportive_message'][:80]}...")
    
    print("\n--- Sample Decision Reasons ---")
    for i in range(3):
        print(f"\n  ID {predictions.iloc[i]['id']}: {predictions.iloc[i]['decision_reason']}")
    
    predictions.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")
    
    return system, predictions


if __name__ == "__main__":
    system, predictions = main()