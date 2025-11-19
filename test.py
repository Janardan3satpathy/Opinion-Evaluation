import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import time
import plotly.express as px
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
# Ensure NLTK data is downloaded in the deployment environment
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, cohen_kappa_score,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from io import StringIO

# --- Contributor Data for Footer ---
CONTRIBUTORS = [
    { 
        'name': 'Janardan Satapathy', 
        'linkedIn': 'https://www.linkedin.com/in/janardan-satapathy-48189b328', 
        'email': 'janardan3satpathy@gmail.com', 
        'facebook': 'https://www.facebook.com/share/1C6uApUB6S/', 
        'instagram': 'https://www.instagram.com/_king_of_all_acids_?igsh=MTJrd2dsMHN2b25pYw==', 
        'phone': '8249393625' 
    },
    { 
        'name': 'Aman Raj', 
        'linkedIn': 'https://www.linkedin.com/in/aman-raj-68048b398', 
        'email': 'amanrajdhull@gmail.com', 
        'facebook': 'https://www.facebook.com/share/1AupTyTsrv/', 
        'instagram': 'https://www.instagram.com/imamanraj_', 
        'phone': '8298304000' 
    },
    { 
        'name': 'Aman Mishra', 
        'linkedIn': 'https://www.linkedin.com/in/aman-mishra-9aa085285', 
        'email': 'amanhanu473@gmail.com', 
        'facebook': 'https://www.facebook.com/share/1EabMPAs1T/', 
        'instagram': 'https://www.instagram.com/aman_hanu?igsh=a2tnOG1zeHdwNDN3', 
        'phone': '9555161472' 
    }
]

# --- NLTK Downloads and Initialization (Cached with st.cache_resource) ---
@st.cache_resource
def initialize_ml_components():
    """Download required NLTK data and initialize text tools once."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}. Check network connection.")
        return None, None
        
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(stopwords.words('english'))
    return lemmatizer, STOPWORDS

lemmatizer, STOPWORDS = initialize_ml_components()

# --- Custom Preprocessing Transformer (Backend Logic) ---
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """A custom transformer to handle all text cleaning steps within the pipeline."""
    def __init__(self):
        self.lemmatizer = lemmatizer 
        self.stopwords = STOPWORDS
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        """Cleans text: HTML removal, lowercasing, stop word filtering, and lemmatization."""
        if not isinstance(text, str): 
            return ""
        # 1. Remove HTML tags (<br />)
        text = re.sub(r'<.*?>', '', text)
        # 2. Clean non-alphabetic characters and lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        words = word_tokenize(text)
        processed_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stopwords and len(word) > 1
        ]
        return " ".join(processed_words)

# --- PLOTTING FUNCTIONS ---
def plot_sentiment_distribution(df):
    """Generates a Plotly pie chart for sentiment distribution."""
    sentiment_counts = df['sentiment'].map({1: 'Positive', 0: 'Negative'}).value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    color_map = {'Positive': 'green', 'Negative': 'red'}
    fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment', 
        title='<b>Dataset Sentiment Distribution (Pre-Training)</b>',
        color='Sentiment',
        color_discrete_map=color_map
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True, 
        margin=dict(t=50, b=0, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    return fig

def plot_review_length_distribution(df):
    """Generates a histogram for review length distribution (as a proxy for categories)."""
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
    fig = px.histogram(
        df, 
        x='review_length', 
        color='sentiment',
        color_discrete_map={1: 'green', 0: 'red'},
        nbins=50, 
        opacity=0.7,
        title='<b>Distribution of Review Lengths by Sentiment (Feature Visualization)</b>',
        labels={'review_length': 'Review Length (Words)', 'sentiment': 'Sentiment (1=Pos, 0=Neg)'}
    )
    fig.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        bargap=0.05
    )
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=10))
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=10))
    return fig

def plot_confusion_matrix_counts(TN, FP, FN, TP):
    """Generates a Plotly bar chart for confusion matrix counts (TP, FP, FN, TN)."""
    data = pd.DataFrame({
        'Metric': ['True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)'],
        'Count': [TP, FP, FN, TN],
        'Type': ['Correct', 'Error', 'Error', 'Correct']
    })
    color_map = {'Correct': '#48c78e', 'Error': '#ff3860'}
    fig = px.bar(
        data, 
        x='Metric', 
        y='Count', 
        color='Type',
        color_discrete_map=color_map,
        title='<b>Confusion Matrix Counts (Test Set Performance)</b>',
        labels={'Metric': 'Classification Result Type', 'Count': 'Count of Instances'},
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title=None,
        yaxis_title="Count"
    )
    return fig

# --- Training and Evaluation Pipeline Function (Cached with st.cache_data) ---
@st.cache_data(max_entries=10, show_spinner=False)
def process_and_train_model(df_hash, df, classifier_name):
    """Runs the full ML process: splitting, preprocessing, training, and evaluation."""
    start_time = time.time()
    if df.shape[0] < 10:
        return "Error", "Dataset too small for training (min 10 rows required).", None, None, None, None, None
    
    # --- FLEXIBLE COLUMN CHECK AND RENAME ---
    review_col, sentiment_col = None, None
    if 'review' in df.columns and 'sentiment' in df.columns:
        review_col, sentiment_col = 'review', 'sentiment'
    elif 'text' in df.columns and 'label' in df.columns:
        review_col, sentiment_col = 'text', 'label'
    if review_col is None:
         return "Error", "CSV must contain columns named 'review' and 'sentiment' OR 'text' and 'label'. Please check your header names.", None, None, None, None, None
    df_copy = df.copy()
    if review_col != 'review':
        df_copy.rename(columns={review_col: 'review'}, inplace=True)
    if sentiment_col != 'sentiment':
        df_copy.rename(columns={sentiment_col: 'sentiment'}, inplace=True)
    # Map sentiment to numerical labels: 1 for positive, 0 for negative/other
    df_copy['sentiment'] = df_copy['sentiment'].astype(str).str.lower().apply(lambda x: 1 if 'pos' in x or x == '1' else 0)
    
    X = df_copy['review']
    y = df_copy['sentiment']
    sentiment_dist_df = df_copy[['review', 'sentiment']].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Select Classifier
    if classifier_name == 'logistic_regression':
        classifier = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    elif classifier_name == 'naive_bayes_classifier':
        classifier = MultinomialNB()
    elif classifier_name == 'support_vector_machine_svm':
        classifier = LinearSVC(random_state=42, max_iter=1000, dual=True)
    elif classifier_name == 'decision_trees':
        classifier = DecisionTreeClassifier(random_state=42)
    elif classifier_name == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) 
    else:
        classifier = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

    # 2. Define the Complete Pipeline
    pipeline = Pipeline([
        ('cleaner', TextPreprocessor()), 
        ('tfidf', TfidfVectorizer(max_features=5000)), 
        ('clf', classifier)
    ])
    
    # 3. Train the Model
    try:
        pipeline.fit(X_train, y_train)
    except ValueError as e:
        return "Error", f"Training failed. Check data format: {e}", None, None, None, None, None

    # 4. Evaluate
    y_pred = pipeline.predict(X_test)
    end_time = time.time()
    pipeline_time = end_time - start_time
    
    # --- CALCULATE ALL REQUESTED METRICS ---
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    total_instances = len(y_test)
    correctly_classified = TP + TN
    incorrectly_classified = FP + FN
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mean_y_test = np.mean(y_test)
    sum_abs_error = np.sum(np.abs(y_pred - y_test))
    sum_abs_baseline_error = np.sum(np.abs(y_test - mean_y_test))
    rae = sum_abs_error / sum_abs_baseline_error if sum_abs_baseline_error > 0 else 0
    roc_auc_str = "N/A (Model lacks probability output)"
    prc_auc_str = "N/A (Model lacks probability output)"
    precision_pos = 0
    recall_pos = 0
    f1_pos = 0

    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        try:
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            roc_auc_str = f"{roc_auc:.4f}"
        except Exception:
            pass
    try:
        precision_pos = precision_score(y_test, y_pred, pos_label=1)
        recall_pos = recall_score(y_test, y_pred, pos_label=1)
        f1_pos = f1_score(y_test, y_pred, pos_label=1)
    except Exception:
        pass
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    metrics = {
        "Pipeline_Time": f"{pipeline_time:.2f} seconds",
        "Overall_Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
        "Correctly_Classified_Instances": f"{correctly_classified} / {total_instances}",
        "Incorrectly_Classified_Instances": f"{incorrectly_classified} / {total_instances}",
        "Mean_Absolute_Error": f"{mae:.4f}",
        "Root_Mean_Square_Error": f"{rmse:.4f}",
        "Relative_Absolute_Error": f"{rae:.4f}",
        "Kappa_Statistics": f"{kappa:.4f}",
        "MCC": f"{mcc:.4f}",
        "ROC_Area": roc_auc_str,
        "PRC_Area": prc_auc_str,
        "Detailed_Classification_Metrics": {
            "TP_Rate (Recall)": f"{tpr:.4f}",
            "FP_Rate": f"{fpr:.4f}",
            "Precision (Positive Class)": f"{precision_pos:.4f}",
            "Recall (Positive Class)": f"{recall_pos:.4f}",
            "F1_Score (F-Measure)": f"{f1_pos:.4f}",
        },
        "Confusion_Matrix (TN, FP, FN, TP)": cm.tolist(),
        "Test_Set_Size": total_instances
    }

    # Generate plots
    sentiment_fig = plot_sentiment_distribution(sentiment_dist_df)
    review_length_fig = plot_review_length_distribution(sentiment_dist_df)
    cm_counts_fig = plot_confusion_matrix_counts(TN, FP, FN, TP)

    # Return the trained pipeline along with metrics and figures
    return "Success", "Analysis Complete", metrics, pipeline, sentiment_fig, review_length_fig, cm_counts_fig

# --- Live Prediction Function (Uses the trained model) ---
def predict_sentiment_live(model, text):
    """Predicts sentiment for a single review, including Neutral classification."""
    if not text or len(text.strip()) < 5:
        return None, 0.0, "Please enter a non-empty review."

    try:
        # 1. Check if the model has predict_proba
        has_proba = hasattr(model.named_steps['clf'], 'predict_proba')
        
        if has_proba:
            positive_proba = model.predict_proba([text])[:, 1][0]
            confidence = max(positive_proba, 1 - positive_proba) * 100
            
            # Nuanced Thresholding
            if 0.45 <= positive_proba <= 0.55:
                sentiment_label = "Neutral 😐"
            elif positive_proba > 0.55:
                sentiment_label = "Positive 🎉"
                confidence = positive_proba * 100
            else:
                sentiment_label = "Negative 😞"
                confidence = (1 - positive_proba) * 100
        else:
            # Fallback for models without probability output (like LinearSVC)
            prediction = model.predict([text])[0]
            sentiment_label = "Positive (High Confidence)" if prediction == 1 else "Negative (High Confidence)"
            confidence = 90.0
            
        return sentiment_label, confidence, None
        
    except Exception as e:
        return None, 0.0, f"An error occurred during prediction: {e}"


# --- 2. LAYOUT COMPONENT FUNCTIONS (Header and Footer) ---
def render_header():
    """Renders the custom header with the logo and college information."""
    st.markdown(
        '<header class="header-section shadow-2xl shadow-pink-900/50 py-6 sm:py-8 rounded-b-xl border-b border-pink-900 mb-8">',
        unsafe_allow_html=True
    )
    
    # Use a single container with margin auto for center alignment
    st.markdown('<div style="margin-left: auto; margin-right: auto; max-width: 1200px;">', unsafe_allow_html=True)
    
    # Use columns just for the logo/title alignment, but within a centered container
    logo_title_cols = st.columns([1.5, 13]) 
    
    with logo_title_cols[0]:
        try:
            st.image('sct logo.jpg', width=150) 
        except FileNotFoundError:
            st.error("Logo file 'sct logo.jpg' not found. Ensure it is in the same directory.")
            st.markdown('<div style="height: 150px;"></div>', unsafe_allow_html=True)
    with logo_title_cols[1]:
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; text-align: left; padding-left: 15px;">
                <h1 class="text-xl sm:text-3xl font-extrabold tracking-tight text-white" style="margin: 0; padding: 0; white-space: nowrap;">
                    SHA-SHIB COLLEGE OF TECHNOLOGY, BHOPAL
                </h1>
                <div style="white-space: nowrap; font-size: 0.75rem; margin-top: 0.25rem; color: #fbbf24;">
                    AN ISO 9001:2008 CERTIFIED ENGINEERING COLLEGE. APPROVED BY A.I.C.T.E GOVT. OF INDIA, NEW DELHI & AFFILIATED TO R.G.P.V. & RECOGNISED BY DTE, BHOPAL (M.P.)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True) # Close the centering container
    st.markdown('</header>', unsafe_allow_html=True)

def render_footer():
    """Renders the custom footer with contributor links."""
    social_links_html_list = []
    for c in CONTRIBUTORS:
        link_block = (
            '<div class="flex flex-col items-center p-2 rounded-lg bg-white/10 hover:bg-white/20 transition duration-150">'
            f'<span class="font-bold text-orange-200 text-sm mb-1">{c["name"]}</span>' 
            '<div class="flex space-x-3 text-sm">'
            f'<a href="{c["linkedIn"]}" target="_blank" title="LinkedIn: {c["name"]}" class="text-white hover:text-blue-300">🔗</a>'
            f'<a href="mailto:{c["email"]}" title="Email: {c["name"]}" class="text-white hover:text-red-300">📧</a>'
            f'<a href="{c["facebook"]}" target="_blank" title="Facebook: {c["name"]}" class="text-white hover:text-blue-500">📘</a>'
            f'<a href="{c["instagram"]}" target="_blank" title="Instagram: {c["name"]}" class="text-white hover:text-pink-400">📸</a>'
            f'<a href="tel:{c["phone"]}" title="Phone: {c["name"]} ({c["phone"]})" class="text-white hover:text-green-400">📞</a>'
            '</div>'
            '</div>'
        )
        social_links_html_list.append(link_block)
    social_links_block = "\n".join(social_links_html_list)
    footer_html = (
        '<footer class="footer-section text-white py-6 mt-10 shadow-2xl shadow-pink-900/50 flex flex-col justify-center items-center rounded-t-xl border-t border-pink-900">'
            '<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs sm:text-sm w-full">'
                '<h3 class="text-lg font-bold text-white mb-2">Developed By</h3>'
                '<div class="flex flex-wrap justify-center items-center space-x-6 space-y-2 mb-4">'
                    + social_links_block +
                '</div>'
                '<p class="mb-1">'
                    'Copyright © <span class="font-bold text-orange-300">2025</span> Janardan & Aman. All rights are reserved.'
                '</p>'
                '<p class="font-light text-pink-200">'
                    'Made and maintained for SHA-SHIB College of Technology'
                '</p>'
            '</div>'
        '</footer>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)

# --- 3. Main Application Function ---
def main_app():
    if lemmatizer is None:
        st.error("Application setup failed due to NLTK initialization error. Please try restarting.")
        return 
    if 'trained_pipeline' not in st.session_state:
        st.session_state['trained_pipeline'] = None
    st.set_page_config(
        page_title="IMDb Review Sentiment Analyzer (Full Stack)", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    # Inject Custom CSS (Streamlit components need styling)
    custom_css = """
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(180deg, #330000 0%, #1a002a 50%, #0d0d19 100%) !important; 
            color: white;
            min-height: 100vh;
        }
        .header-section {
            background: linear-gradient(to right, #ec4899 0%, #f97316 50%, #ec4899 100%) !important; 
        }
        .footer-section {
            background: none !important; 
        }
        .result-box-bg {
            background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
            border: 1px solid #6366f1;
            padding: 2rem;
            border-radius: 0.75rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
        }
        .stButton>button {
            color: white !important;
            background-color: #1d4ed8 !important;
            border-color: #1e40af !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.15s ease-in-out !important;
            border-radius: 0.5rem;
            font-weight: 600;
        }
        .stButton>button:hover {
             background-color: #2563eb !important;
             transform: scale(1.02);
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #2563eb !important;
            color: white !important;
            border: 1px solid #60a5fa !important;
            border-radius: 0.5rem;
        }
        .stFileUploader label {
            color: #f472b6 !important;
            font-weight: 600;
        }
        .header-separator {
            border: 0;
            height: 5px;
            background-image: linear-gradient(to right, rgba(255, 255, 255, 0), #f97316, rgba(255, 255, 255, 0));
            margin: 0;
            padding: 0;
            opacity: 0.8;
        }
        .modebar {
            background-color: transparent !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    render_header()
    st.markdown('<div class="max-w-7xl mx-auto px-4 sm:px-8"><hr class="header-separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="max-w-7xl mx-auto px-4 sm:px-8">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")

    # 1. ABOUT SECTION BOX
    with col1:
        st.markdown("""
            <div class="bg-gradient-to-br from-cyan-600 to-blue-800 p-8 rounded-xl shadow-2xl hover:shadow-xl hover:shadow-cyan-500/50 transition-shadow duration-300 text-center border border-cyan-400 h-full">
                <h2 class="text-2xl font-bold text-white mb-6 border-b-2 pb-2 border-cyan-300 inline-block">
                    About
                </h2>
                <div class="text-left">
                    <h3 class="text-xl font-semibold text-cyan-200 mb-3">Opinion Evaluation</h3>
                    <p class="text-gray-200 leading-relaxed text-sm">
                        Opinion Evaluation is the structured process of examining unstructured customer feedback (reviews/Opinions) to identify patterns, gauge sentiment, and uncover actionable insights. It serves as a vital tool for businesses to understand user satisfaction and improve products or services.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 2. UPLOAD DATASET SECTION BOX
    with col2:
        st.markdown("""
            <div class="bg-gradient-to-br from-teal-600 to-green-800 p-8 rounded-xl shadow-2xl hover:shadow-xl hover:shadow-green-500/50 transition-shadow duration-300 text-center border border-teal-400 h-full">
                <h2 class="2xl font-bold text-white mb-6 border-b-2 pb-2 border-teal-300 inline-flex items-center space-x-3">
                    <span>Upload Dataset</span>
                </h2>
                <p class="text-teal-200 mb-6 text-sm">
                    Only <span class="font-bold text-white">.CSV</span> files are supported for dataset input.
                </p>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select CSV File", 
            type=['csv'], 
            key="file_uploader",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. ALGORITHM SELECTION BOX
    with col3:
        model_options = {
            'supervised_learning': {
                'logistic_regression': 'Logistic Regression',
                'naive_bayes_classifier': 'Naive Bayes Classifier',
                'support_vector_machine_svm': 'Support Vector Machine (SVM)',
                'decision_trees': 'Decision Trees',
                'random_forest': 'Random Forest',
            },
            'unsupervised_learning': {'coming_soon': 'Unsupervised Learning (Coming Soon)'},
            'semi_supervised_learning': {'coming_soon': 'Semi-Supervised Learning (Coming Soon)'},
            'reinforcement_learning': {'coming_soon': 'Reinforcement Learning (Coming Soon)'}
        }
        st.markdown("""
            <div class="bg-gradient-to-br from-red-600 to-orange-800 p-8 rounded-xl shadow-2xl hover:shadow-xl hover:shadow-red-500/50 transition-shadow duration-300 text-center border border-red-400 h-full">
                <h2 class="2xl font-bold text-white mb-6 border-b-2 pb-2 border-red-300 inline-flex items-center space-x-3">
                    <span>Select Algorithm</span>
                </h2>
                <p class="text-red-200 mb-6 text-sm">
                    Choose the machine learning approach and specific model for the Evaluation.
                </p>
        """, unsafe_allow_html=True)
        analysis_type = st.selectbox(
            "Evaluation Type:",
            options=model_options.keys(),
            format_func=lambda x: x.replace('_', ' ').title(),
            index=0,
            key="analysis_type_select"
        )
        current_models = model_options.get(analysis_type, {})
        is_model_select_disabled = (analysis_type != 'supervised_learning')
        selected_model_key = st.selectbox(
            "Select the model to be used:",
            options=list(current_models.keys()),
            format_func=lambda x: current_models[x],
            index=0,
            disabled=is_model_select_disabled,
            key="model_select"
        )
        is_button_disabled = uploaded_file is None or selected_model_key == 'coming_soon'
        st.button(
            "Run Evaluation", 
            key="run_analysis_button",
            disabled=is_button_disabled
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ANALYSIS LOGIC BLOCK (Run on Button Click) ---
    if st.session_state.run_analysis_button and uploaded_file is not None and not is_button_disabled:
        file_content = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(file_content))
        df_hash = hash(file_content) 
        with st.spinner(f"Running {current_models[selected_model_key]} on {uploaded_file.name}... (This may take a moment on the first run)"):
            status, message, metrics, pipeline, sentiment_fig, review_length_fig, cm_counts_fig = process_and_train_model(
                df_hash, 
                df, 
                selected_model_key
            )
            st.session_state['trained_pipeline'] = pipeline
            st.session_state['last_metrics'] = metrics
            st.session_state['last_status'] = status
            st.session_state['last_message'] = message
            st.session_state['last_model'] = current_models[selected_model_key]
            st.session_state['sentiment_fig'] = sentiment_fig
            st.session_state['review_length_fig'] = review_length_fig
            st.session_state['cm_counts_fig'] = cm_counts_fig

    # --- 4. RESULTS SECTION BOX (Full Width) ---
    st.markdown("""
        <div class="result-box-bg w-full mt-8">
            <h2 class="text-2xl font-bold text-white mb-6 border-b-2 pb-2 border-indigo-300 inline-block">
                Evaluation Results
            </h2>
    """, unsafe_allow_html=True)

    if 'last_status' in st.session_state and st.session_state.last_status == "Success":
        metrics = st.session_state.last_metrics

        # 1. ANALYSIS COMPLETE MESSAGE
        st.markdown(f"""
            <h3 class="text-xl font-bold text-green-300 mb-4">{st.session_state.last_message}</h3>
            <p class="text-gray-200 mb-2">Model Used: <code class="text-pink-300 font-bold">{st.session_state.last_model}</code></p>
            <p class="text-gray-200 mb-4">Pipeline Time: <code class="text-pink-300 font-bold">{metrics['Pipeline_Time']}</code></p>
        """, unsafe_allow_html=True)

        # 2. PRE-PROCESSING VISUALIZATIONS (Sentiment & Review Length)
        st.markdown('<div class="mt-4">', unsafe_allow_html=True)
        st.markdown(
            '<h4 class="font-bold text-indigo-200 mb-2">Pre-processing Visualizations</h4>',
            unsafe_allow_html=True
        )
        # Display the Sentiment Distribution and Review Length side-by-side
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            st.plotly_chart(st.session_state.sentiment_fig, use_container_width=True)
            
        with vis_col2:
            st.plotly_chart(st.session_state.review_length_fig, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)


        # 3-6. DETAILED METRIC BREAKDOWN (Text Metrics)
        st.markdown('<div class="mt-8">', unsafe_allow_html=True)
        st.markdown(
            '<h3 class="text-xl font-bold text-pink-300 mb-4 border-b pb-2 border-pink-400">Detailed Metric Breakdown</h3>',
            unsafe_allow_html=True
        )

        # --- Two Column Structure for Detailed Metrics ---
        metric_col1, metric_col2 = st.columns(2, gap="large")

        with metric_col1:
            # 3. Classification Summary
            st.markdown("""
                <div>
                    <h4 class="font-bold text-indigo-200 mb-2">CLASSIFICATION SUMMARY</h4>
                    <p>Accuracy: <span class="text-green-300">{Overall_Accuracy}</span></p>
                    <p>Kappa Statistics: <span class="text-yellow-300">{Kappa_Statistics}</span></p>
                    <p>MCC: <span class="text-yellow-300">{MCC}</span></p>
                    <p>ROC Area: <span class="text-yellow-300">{ROC_Area}</span></p>
                    <p>PRC Area: <span class="text-yellow-300">{PRC_Area}</span></p>
                </div>
            """.format(**metrics), unsafe_allow_html=True)

            # 5. CLASSIFICATION COUNT
            st.markdown(f"""
                <div class="mt-6">
                    <h4 class="font-bold text-indigo-200 mb-2">CLASSIFICATION COUNT</h4>
                    <p>Correctly Classified: <span class="text-green-300">{metrics['Correctly_Classified_Instances']}</span></p>
                    <p>Incorrectly Classified: <span class="text-red-300">{metrics['Incorrectly_Classified_Instances']}</span></p>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            # 4. Error & Size
            st.markdown("""
                <div>
                    <h4 class="font-bold text-indigo-200 mb-2">ERROR & SIZE</h4>
                    <p>MAE: <span class="text-red-300">{Mean_Absolute_Error}</span></p>
                    <p>RMSE: <span class="text-red-300">{Root_Mean_Square_Error}</span></p>
                    <p>RAE: <span class="text-red-300">{Relative_Absolute_Error}</span></p>
                    <p>Test Set Size: <span class="text-cyan-300">{Test_Set_Size}</span></p>
                </div>
            """.format(**metrics), unsafe_allow_html=True)
            
            # 6. POSITIVE CLASS METRICS 
            st.markdown("""
                <div class="mt-6">
                    <h4 class="font-bold text-indigo-200 mb-2">POSITIVE CLASS METRICS</h4>
                    <p>Precision: <span class="text-green-300">{Precision (Positive Class)}</span></p>
                    <p>Recall (TP Rate): <span class="text-green-300">{Recall (Positive Class)}</span></p>
                    <p>F1 Score: <span class="text-green-300">{F1_Score (F-Measure)}</span></p>
                    <p>FP Rate: <span class="text-red-300">{FP_Rate}</span></p>
                </div>
            """.format(**metrics['Detailed_Classification_Metrics']), unsafe_allow_html=True)

        # Confusion Matrix Summary Text (Full Width below 2 columns)
        st.markdown(f"""
            <p class="mt-4 text-xs italic text-gray-400">Confusion Matrix: [[TN, FP], [FN, TP]] -> {metrics['Confusion_Matrix (TN, FP, FN, TP)']}</p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 7. MODEL PERFORMANCE VISUALIZATION (Confusion Matrix) - FINAL POSITION
        st.markdown('<div class="mt-8">', unsafe_allow_html=True)
        st.markdown(
            '<h3 class="text-xl font-bold text-pink-300 mb-4 border-b pb-2 border-pink-400">Model Performance Visualization (Confusion Matrix)</h3>',
            unsafe_allow_html=True
        )
        st.plotly_chart(st.session_state.cm_counts_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    
    elif 'last_status' in st.session_state and st.session_state.last_status == "Error":
        st.error(st.session_state.last_message)
    
    else:
        st.info("Upload a file, select a model, and click 'Run Evaluation' to see results.")
        
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 5. LIVE PREDICTION SECTION ---
    if st.session_state.get('trained_pipeline'):
        st.markdown("""
            <div class="result-box-bg w-full mt-8">
                <h2 class="2xl font-bold text-white mb-6 border-b-2 pb-2 border-indigo-300 inline-block">
                    Live Opinion Prediction
                </h2>
                <p class="text-gray-200 mb-4">Test the currently trained model by entering any Opinion below.</p>
            </div>
        """, unsafe_allow_html=True)

        live_predictor_container = st.container()
        
        with live_predictor_container:
            col_in, col_btn = st.columns([4, 1])

            with col_in:
                user_review = st.text_area("Enter your review here:", 
                                           placeholder="Type your review here.", 
                                           key="user_review_input", 
                                           height=100, 
                                           label_visibility="collapsed")
            
            with col_btn:
                # Add a vertical gap using custom styling to align the button
                st.markdown('<div style="height: 2.2rem;"></div>', unsafe_allow_html=True) 
                predict_button = st.button("Predict Sentiment", key="predict_sentiment_button")
            
            if predict_button:
                
                pipeline = st.session_state['trained_pipeline']
                
                # Predict sentiment
                sentiment, confidence, error_msg = predict_sentiment_live(pipeline, user_review)

                if error_msg:
                    st.error(f"Prediction Error: {error_msg}")
                else:
                    if "Neutral" in sentiment:
                         color = "text-yellow-400"
                    elif "Positive" in sentiment:
                        color = "text-green-400"
                    else:
                        color = "text-red-400"
                        
                    st.markdown(f"""
                        <div class="bg-indigo-700/50 p-4 mt-4 rounded-lg">
                            <h4 class="text-xl font-semibold mb-2 text-white">Prediction Result:</h4>
                            <p class="text-2xl font-extrabold {color}">
                                Sentiment: {sentiment}
                            </p>
                            <p class="text-gray-300 mt-1 text-sm">
                                Confidence: {confidence:.2f}% (Note: SVM/Decision Trees provide binary confidence)
                            </p>
                        </div>
                    """, unsafe_allow_html=True)


    # --- RENDER FOOTER ---
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- FOOTER SEPARATOR LINE (To separate main content from transparent footer) ---
    st.markdown('<div class="max-w-7xl mx-auto px-4 sm:px-8"><hr class="header-separator"></div>', unsafe_allow_html=True)
    
    render_footer()


if __name__ == "__main__":
    main_app()