from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

nltk.download('vader_lexicon')

app = Flask(__name__)

# Load models
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
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([statement, reply])
        similarity = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        agreement_score = round(similarity * 100, 2)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)