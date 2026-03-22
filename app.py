import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.emotion_system import EmotionAwareSystem

st.set_page_config(page_title="Emotion-Aware Decision System", page_icon="🧠")

st.title("Emotion-Aware Decision System")
st.markdown("### Understand your mental state and get personalized recommendations")

@st.cache_resource
def load_model():
    train_df = pd.read_csv('data/Sample_arvyax_reflective_training.csv')
    system = EmotionAwareSystem()
    system.fit(train_df)
    return system

system = load_model()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Journal Entry")
    journal_text = st.text_area(
        "How are you feeling?",
        placeholder="Write about your mood, thoughts, or experience...",
        height=150
    )

with col2:
    st.subheader("Your State")
    
    energy_level = st.slider(
        "Energy Level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very low, 5 = Very high"
    )
    
    stress_level = st.slider(
        "Stress Level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very calm, 5 = Very stressed"
    )
    
    time_of_day = st.selectbox(
        "Time of Day",
        options=["morning", "early_morning", "afternoon", "evening", "night"],
        index=0
    )

if st.button("Analyze", type="primary", use_container_width=True):
    if journal_text.strip():
        with st.spinner("Analyzing..."):
            input_data = pd.DataFrame([{
                'id': 1,
                'journal_text': journal_text,
                'energy_level': energy_level,
                'stress_level': stress_level,
                'time_of_day': time_of_day,
                'ambience_type': 'cafe',
                'duration_min': 15,
                'sleep_hours': 7,
                'previous_day_mood': 'neutral',
                'face_emotion_hint': 'none',
                'reflection_quality': 'vague'
            }])
            
            predictions = system.predict(input_data)
            pred = predictions.iloc[0]
            
            st.markdown("---")
            st.subheader("Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Emotional State", pred['predicted_state'].replace("np.str_", ""))
            
            with col2:
                st.metric("Intensity", f"{pred['predicted_intensity']}/5")
            
            with col3:
                st.metric("Confidence", f"{pred['confidence']:.0%}")
            
            st.markdown("---")
            st.subheader("Recommendation")
            
            recommendation_col1, recommendation_col2 = st.columns(2)
            
            with recommendation_col1:
                st.info(f"**Action:** {pred['what_to_do'].replace('_', ' ').title()}")
                st.info(f"**Timing:** {pred['when_to_do'].replace('_', ' ').title()}")
            
            with recommendation_col2:
                if pred['uncertain_flag'] == 1:
                    st.warning("⚠️ Low confidence - Recommendation may need review")
                st.caption(f"Reason: {pred['confidence_reason']}")
            
            st.markdown("---")
            st.subheader("Supportive Message")
            st.success(pred['supportive_message'])
            
            with st.expander("Decision Breakdown"):
                st.json(pred['decision_reason'])
    else:
        st.warning("Please enter some text to analyze")

st.markdown("---")
st.caption("Emotion-Aware Decision System | Built with Streamlit")
