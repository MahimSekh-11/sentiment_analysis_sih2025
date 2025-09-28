import streamlit as st

def show():
    st.title("ℹ️ About SentimentAI")

    st.markdown("""
    SentimentAI is an advanced sentiment analysis platform designed to classify emotions
    from text data (positive, negative, neutral) with high accuracy.  
    It combines state-of-the-art transformer models with interactive visualization to
    make sentiment insights more accessible.
    """)

    # ---------------- Project Goal ----------------
    st.markdown("### 🎯 Goal")
    st.markdown("""
    - Provide **instant sentiment insights** on single comments.  
    - Enable **batch analysis** for uploaded CSV datasets.  
    - Generate **interactive analytics & trends** to track sentiment over time.  
    - Deliver a **lightweight, user-friendly interface** for both individuals and organizations.
    """)

    # ---------------- Workflow ----------------
    st.markdown("### 🔄 Workflow")
    st.markdown("""
    1. **Input**: User enters text or uploads a dataset.  
    2. **Preprocessing**: Text is cleaned, tokenized, and prepared for the model.  
    3. **Model Prediction**:  
       - Uses a **fine-tuned BERT transformer** from Hugging Face.  
       - Each comment is classified into **positive / negative / neutral** with a confidence score.  
    4. **Post-processing**: Results are aggregated, visualized, and stored in the history.  
    5. **Output**: Sentiment distribution, trends, and word-level insights.  
    """)

    # ---------------- Why Better ----------------
    st.markdown("### 🚀 Why SentimentAI is Better")
    st.markdown("""
    - **High Accuracy**: Outperforms traditional methods like VADER or Naive Bayes by
      leveraging transformer-based deep learning.  
    - **Consistency**: Same input always produces the same result (deterministic API).  
    - **Scalability**: Works on single text, batch datasets, and integrates with APIs.  
    - **Explainability**: Beyond labels, it shows **confidence levels** and why a
      text was classified that way.  
    - **Visualization**: Interactive dashboards for deeper insights, unlike many static models.  
    """)

    # ---------------- Impact ----------------
    st.markdown("### 🌍 Impact")
    st.markdown("""
    - **Businesses** can monitor customer feedback in real-time.  
    - **Researchers** can analyze large-scale text datasets quickly.  
    - **Individuals** can understand emotional tone in conversations or social media.  
    - Helps reduce **bias in decision-making** by providing data-driven sentiment insights.  
    """)

    # ---------------- Technology Stack ----------------
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    - **Frontend**: Streamlit, Plotly  
    - **Backend**: FastAPI  
    - **ML Models**: Hugging Face BERT (fine-tuned), optional VADER baseline  
    - **Data Handling**: Pandas, JSON storage  
    """)

    # ---------------- About the Project ----------------
    st.markdown("### 📜 About This Project")
    st.markdown("""
    SentimentAI was built as a complete end-to-end system:  
    - **Model training** (fine-tuned on real-world datasets).  
    - **Backend service** for predictions and integrations.  
    - **Frontend interface** for interactive visualization.  

    The project demonstrates how **cutting-edge NLP** can be deployed in a simple,
    scalable, and user-friendly way.
    """)
