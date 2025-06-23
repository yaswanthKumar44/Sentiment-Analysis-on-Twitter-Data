import streamlit as st
import joblib
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import time

# Load model & vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|[^A-Za-z\s]","", text)
    return text.lower().strip()

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stButton button {
        background-color: #4A90E2;
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 12px;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #357ABD;
    }
    .title {
        text-align: center;
        color: #333333;
        font-size: 45px;
        font-weight: 900;
        margin-top: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #666666;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        color: gray;
        margin-top: 2rem;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📝 Analyze Tweet", "☁️ WordClouds"])

# App Title and Subtitle
st.markdown("<h1 class='title'>Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict tweet sentiment using ML trained on Sentiment140 dataset</p>", unsafe_allow_html=True)
st.markdown("---")

# Home Page
if page == "🏠 Home":
    st.markdown("## 📊 Project Overview")
    st.write("""
    This app uses **Logistic Regression** trained on over **1.6M tweets** from the **Sentiment140 dataset**.
    
    🔹 **Real-time sentiment prediction**  
    🔹 **WordCloud visualizations**  
    🔹 Professional, clean dashboard interface  
    🔹 Developed using **Streamlit**, **Scikit-learn**, and **WordCloud**
    """)

    st.markdown("## 📈 Dataset Summary")
    col1, col2 = st.columns(2)
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=['target','id','date','query','user','text'])
    df = df[df['target'] != 2]
    df['sentiment'] = df['target'].map({0: 'Negative', 4: 'Positive'})

    col1.metric("Total Tweets", f"{len(df):,}")
    col2.metric("Positive %", f"{round(len(df[df['sentiment']=='Positive'])/len(df)*100,2)} %")

    st.markdown("## 📚 About Sentiment140 Dataset")
    st.write("The dataset contains tweets labeled automatically as positive or negative based on emoticons.")

# Analyze Tweet Page
elif page == "📝 Analyze Tweet":
    st.markdown("<h3 class='section-title'>🔍 Enter a tweet to analyze sentiment:</h3>", unsafe_allow_html=True)
    user_input = st.text_area("Type your tweet here", "")

    if st.button("🔍 Analyze"):
        if user_input:
            with st.spinner("Analyzing sentiment... please wait ⏳"):
                time.sleep(1.5)  # simulate processing time
                cleaned = clean_text(user_input)
                vect_text = vectorizer.transform([cleaned])
                prediction = model.predict(vect_text)[0]

            st.markdown("## 🎯 Prediction Result")
            if prediction == 'Positive':
                st.success("✅ Positive Sentiment Detected!")
                st.markdown("😊 This tweet expresses positive emotions.")
            else:
                st.error("🚫 Negative Sentiment Detected!")
                st.markdown("😠 This tweet expresses negative emotions.")
        else:
            st.warning("⚠️ Please enter a tweet.")

# WordClouds Page
elif page == "☁️ WordClouds":
    st.markdown("<h3 class='section-title'>☁️ Sentiment WordCloud Visualizations</h3>", unsafe_allow_html=True)
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=['target','id','date','query','user','text'])
    df = df[df['target'] != 2]
    df['sentiment'] = df['target'].map({0: 'Negative', 4: 'Positive'})

    df['clean_text'] = df['text'].apply(clean_text)

    col1, col2 = st.columns(2)
    for sentiment, col in zip(['Positive', 'Negative'], [col1, col2]):
        with col:
            st.markdown(f"#### {sentiment} Tweets")
            with st.spinner("Generating word cloud... ⏳"):
                time.sleep(1)
                text = " ".join(df[df['sentiment']==sentiment]['clean_text'].sample(8000, random_state=42))
                wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

# Footer
st.markdown("<div class='footer'>© 2025 P. Yaswanth Kumar — Powered by Streamlit | Sentiment140 | Logistic Regression</div>", unsafe_allow_html=True)
