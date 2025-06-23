# Sentiment-Analysis-on-Twitter-Data

# 📝 Twitter Sentiment Analyzer (Streamlit App)

A professional-grade, interactive Streamlit web app for performing real-time sentiment analysis on tweets using a Logistic Regression model trained on the **Sentiment140 dataset**.

## 📊 Overview

This project uses Natural Language Processing (NLP) techniques and machine learning to classify tweets as **Positive** or **Negative**.  
It also provides engaging visualizations like **WordClouds** for sentiment-specific tweets and a clean, modern dashboard interface.

---

## ✨ Features

- 📌 Real-time tweet sentiment prediction  
- 📌 Logistic Regression model trained on 1.6 million tweets  
- 📌 Beautiful **WordCloud visualizations** for both Positive and Negative tweets  
- 📌 **Animated loading spinners** during analysis and generation  
- 📌 Clean **Streamlit sidebar navigation**  
- 📌 Custom CSS for a modern, professional dashboard look  
- 📌 Emoji-based sentiment result feedback  

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit**
- **Scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **WordCloud**
- **Joblib**

---

## 📚 Dataset

- **Name:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size:** 1.6M tweets with positive and negative labels

---

## 📂 Project Structure

```

sentiment140\_streamlit\_app/
├── app.py
├── sentiment\_model.pkl
├── tfidf\_vectorizer.pkl
├── training.1600000.processed.noemoticon.csv
├── requirements.txt
└── README.md

````

---

## 🚀 How to Run

1️⃣ **Clone this repository**
```bash
git clone https://github.com/yourusername/sentiment140_streamlit_app.git
cd sentiment140_streamlit_app
````

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

3️⃣ **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 📈 Future Enhancements

* 🔄 Integrate live Twitter API for real-time tweet fetching
* 🤖 Replace Logistic Regression with fine-tuned BERT or RoBERTa model
* ☁️ Deploy on Streamlit Cloud for public online access
* 📱 Make the app mobile-optimized

---

## 📃 License

This project is open-source and available for educational, research, and personal use.

---

## 🙌 Acknowledgements

* [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
* [Streamlit](https://streamlit.io/)
* [Scikit-learn](https://scikit-learn.org/stable/)

---

## 👨‍💻 Author

**P. Yaswanth Kumar**
