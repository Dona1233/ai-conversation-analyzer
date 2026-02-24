import streamlit as st
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary data for NLTK
@st.cache_resource
def download_nltk():
    nltk.download('vader_lexicon')

download_nltk()

st.set_page_config(page_title="AgreeDetector", page_icon="🤝")
st.title("🤝 AI Conversation Analyzer")

# Load models
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sia = SentimentIntensityAnalyzer()
    return model, sia

model, sia = load_models()

# Logic functions
toxic_words = ["stupid", "idiot", "dumb", "hate", "shut up", "nonsense", "useless", "worst"]

def calculate_toxicity(text):
    score = sum(1 for word in toxic_words if word in text.lower())
    return min(score * 15, 100)

# User Interface
statement = st.text_area("Original Statement", "I think we should focus on the new project.")
reply = st.text_area("The Reply", "I hate that idea, it's totally useless.")

if st.button("Analyze Conversation"):
    with st.spinner('Analyzing patterns...'):
        # Agreement Score
        emb1 = model.encode(statement, convert_to_tensor=True)
        emb2 = model.encode(reply, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        agreement_score = round((similarity + 1) / 2 * 100, 2)
        
        # Sentiment & Emotion
        statement_sent = sia.polarity_scores(statement)['compound']
        reply_sent = sia.polarity_scores(reply)['compound']
        emotion_score = round(abs(reply_sent) * 100, 2)
        toxicity_score = calculate_toxicity(reply)
        escalation_risk = round(abs(statement_sent - reply_sent) * 100, 2)

        # Display Results in Columns
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Agreement", f"{agreement_score}%")
        col2.metric("Toxicity", f"{toxicity_score}%")
        col3.metric("Escalation Risk", f"{escalation_risk}%")

        if toxicity_score > 30:
            st.error("⚠️ High toxicity detected in the reply.")
        elif agreement_score > 70:
            st.success("✅ The conversation appears constructive.")