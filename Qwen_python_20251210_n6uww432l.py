import streamlit as st
from src.predict import SentimentAnalyzer

st.title("SentimentScope 🎯")
st.subheader("Real-time Sentiment Analysis for Tweets & Reviews")

analyzer = SentimentAnalyzer(model_type='distilbert')

user_input = st.text_area("Enter text:", "This movie was absolutely amazing!")
if st.button("Analyze"):
    result = analyzer.predict(user_input)
    sentiment = result['label'].capitalize()
    confidence = f"{result['confidence']:.2%}"
    
    # Emoji + color coding
    emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}[result['label']]
    color = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}[result['label']]
    
    st.markdown(f"<h2 style='color:{color}'>{emoji} {sentiment} ({confidence} confidence)</h2>", 
                unsafe_allow_html=True)
    
    # Confidence bar chart
    st.bar_chart({k: v for k, v in result['all_scores'].items()}, height=200)