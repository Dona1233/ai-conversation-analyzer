from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import torch

app = Flask(__name__)

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
sia = SentimentIntensityAnalyzer()

# Simple toxicity keywords
toxic_words = [
    "stupid", "idiot", "dumb", "hate", "shut up",
    "nonsense", "useless", "worst"
]

def calculate_toxicity(text):
    score = 0
    for word in toxic_words:
        if word in text.lower():
            score += 1
    return min(score * 15, 100)

def emotional_intensity(text):
    sentiment = sia.polarity_scores(text)
    return abs(sentiment['compound']) * 100

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        statement = request.form["statement"]
        reply = request.form["reply"]

        # Agreement Score
        emb1 = model.encode(statement, convert_to_tensor=True)
        emb2 = model.encode(reply, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        agreement_score = round((similarity + 1) / 2 * 100, 2)

        disagreement_score = round(100 - agreement_score, 2)

        # Sentiment
        statement_sent = sia.polarity_scores(statement)['compound']
        reply_sent = sia.polarity_scores(reply)['compound']

        # Emotional Intensity
        emotion_score = round(emotional_intensity(reply), 2)

        # Toxicity
        toxicity_score = calculate_toxicity(reply)

        # Emotional Escalation Risk
        escalation_risk = round(abs(statement_sent - reply_sent) * 100, 2)

        # Polarization Index
        polarization_index = round((agreement_score * escalation_risk) / 100, 2)

        results = {
            "agreement": agreement_score,
            "disagreement": disagreement_score,
            "emotion": emotion_score,
            "toxicity": toxicity_score,
            "escalation": escalation_risk,
            "polarization": polarization_index
        }

    return render_template("index.html", results=results)
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AgreeDetector", page_icon="🔍")
st.title("🤝 AgreeDetector")
st.write("Compare two statements to see if they align.")

@st.cache_resource # This prevents the model from reloading every time
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

col1, col2 = st.columns(2)
with col1:
    sent1 = st.text_area("Statement 1", "The weather is great.")
with col2:
    sent2 = st.text_area("Statement 2", "It is a beautiful day.")

if st.button("Check Agreement"):
    emb1 = model.encode(sent1)
    emb2 = model.encode(sent2)
    score = util.cos_sim(emb1, emb2).item()
    
    st.divider()
    st.subheader(f"Agreement Score: {score:.2f}")
    if score > 0.7:
        st.success("High Agreement! ✅")
    else:
        st.warning("Low Agreement. ⚠️")