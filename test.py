import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import time
from nltk.corpus import stopwords
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

# Initialize components globally before the main function runs
lemmatizer, STOPWORDS = initialize_ml_components()


# --- Custom Preprocessing Transformer (Backend Logic) ---
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """A custom transformer to handle all text cleaning steps within the pipeline."""
    def __init__(self):
        # Access global NLTK components here
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

# --- Training and Evaluation Pipeline Function (Cached with st.cache_data) ---
@st.cache_data(max_entries=10, show_spinner=False)
def process_and_train_model(df_hash, df, classifier_name):
    """
    Runs the full ML process: splitting, preprocessing, vectorization, training, and evaluation.
    Caches results based on input DataFrame hash and classifier name.
    """
    start_time = time.time() # Start time tracking
    
    # We use df_hash as the cache key, but work with the actual df

    if df.shape[0] < 10:
        return "Error", "Dataset too small for training (min 10 rows required).", None, None
    
    # --- FLEXIBLE COLUMN CHECK AND RENAME ---
    review_col, sentiment_col = None, None
    
    # Check for IMDB Dataset format (review, sentiment)
    if 'review' in df.columns and 'sentiment' in df.columns:
        review_col, sentiment_col = 'review', 'sentiment'
    # Check for IMDB.csv format (text, label)
    elif 'text' in df.columns and 'label' in df.columns:
        review_col, sentiment_col = 'text', 'label'
    
    if review_col is None:
         return "Error", "CSV must contain columns named 'review' and 'sentiment' OR 'text' and 'label'. Please check your header names.", None, None

    # Rename columns internally for consistent downstream processing
    df_copy = df.copy()
    if review_col != 'review':
        df_copy.rename(columns={review_col: 'review'}, inplace=True)
    if sentiment_col != 'sentiment':
        df_copy.rename(columns={sentiment_col: 'sentiment'}, inplace=True)
    # --- END FLEXIBLE COLUMN CHECK AND RENAME ---

    # Map sentiment to numerical labels: 1 for positive, 0 for negative/other
    df_copy['sentiment'] = df_copy['sentiment'].astype(str).str.lower().apply(lambda x: 1 if 'pos' in x or x == '1' else 0)
    
    X = df_copy['review']
    y = df_copy['sentiment']
    
    # Use a small test size (e.g., 20%) to ensure enough training data
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
        return "Error", f"Training failed. Check data format: {e}", None, None

    # 4. Evaluate
    y_pred = pipeline.predict(X_test)
    
    # Stop time tracking
    end_time = time.time()
    pipeline_time = end_time - start_time
    
    # --- CALCULATE ALL REQUESTED METRICS ---
    
    # Confusion Matrix Components (TN, FP, FN, TP)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    total_instances = len(y_test)
    correctly_classified = TP + TN
    incorrectly_classified = FP + FN

    # TP/FP Rates and Error Rates
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Relative Absolute Error (RAE)
    mean_y_test = np.mean(y_test)
    sum_abs_error = np.sum(np.abs(y_pred - y_test))
    sum_abs_baseline_error = np.sum(np.abs(y_test - mean_y_test))
    rae = sum_abs_error / sum_abs_baseline_error if sum_abs_baseline_error > 0 else 0

    # Probabilistic Metrics (ROC/PRC Area)
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
            pass # Keep as N/A

    # Standard metrics for the positive class
    try:
        precision_pos = precision_score(y_test, y_pred, pos_label=1)
        recall_pos = recall_score(y_test, y_pred, pos_label=1)
        f1_pos = f1_score(y_test, y_pred, pos_label=1)
    except Exception:
        pass # Keep as 0

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

    # Return the trained pipeline along with metrics
    return "Success", "Analysis Complete", metrics, pipeline

# --- Live Prediction Function (Uses the trained model) ---
def predict_sentiment_live(model, text):
    """Predicts sentiment for a single review, including Neutral classification."""
    if not text or len(text.strip()) < 5:
        return None, 0.0, "Please enter a non-empty review."

    try:
        # 1. Check if the model has predict_proba
        has_proba = hasattr(model.named_steps['clf'], 'predict_proba')
        
        # 2. Preprocess the text using the pipeline's internal methods
        # The pipeline handles cleaning and vectorization internally
        
        if has_proba:
            positive_proba = model.predict_proba([text])[:, 1][0]
            confidence = max(positive_proba, 1 - positive_proba) * 100
            
            # Nuanced Thresholding
            if 0.45 <= positive_proba <= 0.55: # Tightened threshold for 'Neutral'
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
            confidence = 90.0 # Standard high confidence for hard classification
            
        return sentiment_label, confidence, None
        
    except Exception as e:
        return None, 0.0, f"An error occurred during prediction: {e}"


# --- 2. LAYOUT COMPONENT FUNCTIONS (Header and Footer) ---

def render_header():
    """Renders the custom header with the logo placed before the title."""
    
    # Start of the header container (using Streamlit's native container for alignment)
    st.markdown(
        '<header class="header-section shadow-2xl shadow-pink-900/50 py-6 sm:py-8 rounded-b-xl border-b border-pink-900 mb-8">',
        unsafe_allow_html=True
    )
    
    # Use Streamlit columns for logo and title alignment
    header_cols = st.columns([1, 4, 1]) # 1: Logo area, 4: Title area, 1: Spacer

    with header_cols[1]: # Use the middle column for content
        # --- Logo and Title Placement ---
        # Adjusted ratio for slightly larger logo
        logo_title_cols = st.columns([1, 15]) 
        
        with logo_title_cols[0]:
            try:
                # Increased width from 65 to 80
                st.image('sct logo.jpg', width=120) 
            except FileNotFoundError:
                st.error("Logo file 'sct logo.jpg' not found. Ensure it is in the same directory.")
                st.markdown('<div style="height: 80px;"></div>', unsafe_allow_html=True) # Increased placeholder height
            
        with logo_title_cols[1]:
            st.markdown(
                """
                <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; text-align: left; padding-left: 10px;">
                    <h1 class="text-xl sm:text-3xl font-extrabold tracking-tight text-white" style="margin: 0; padding: 0;">
                        SHA-SHIB COLLEGE OF TECHNOLOGY, BHOPAL </div>
                        <div> AN ISO 9001:2008 CERTIFIED ENGINEERING COLLEGE. APPROVED BY A.I.C.T.E GOVT. OF INDIA, NEW DELHI & AFFILIATED TO R.G.P.V. & RECOGNISED BY DTE, BHOPAL (M.P.) </div>
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('</header>', unsafe_allow_html=True)


def render_footer():
    """Renders the custom footer as a distinct section with centered content and contributor links."""
    
    # Dynamically generate the contributor social link HTML
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

    # Note: The 'footer-section' CSS styling ensures it is transparent.
    footer_html = (
        '<footer class="footer-section text-white py-6 mt-10 shadow-2xl shadow-pink-900/50 flex flex-col justify-center items-center rounded-t-xl border-t border-pink-900">'
            '<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs sm:text-sm w-full">'
                
                ''
                '<h3 class="text-lg font-bold text-white mb-2">Developed By</h3>'
                '<div class="flex flex-wrap justify-center items-center space-x-6 space-y-2 mb-4">'
                    + social_links_block +
                '</div>'

                ''
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
        /* General App Styling */
        .stApp {
            font-family: 'Inter', sans-serif;
            /* Dark background gradient retained */
            background: linear-gradient(180deg, #330000 0%, #1a002a 50%, #0d0d19 100%) !important; 
            color: white;
            min-height: 100vh;
        }
        
        /* --- NEW DISTINCT SECTION STYLES --- */
        .header-section {
            /* Pink on the sides, Orange in the center (Header maintains color) */
            background: linear-gradient(to right, #ec4899 0%, #f97316 50%, #ec4899 100%) !important; 
        }
        
        .footer-section {
            /* Footer now has NO background, making it transparent and same as .stApp background */
            background: none !important; 
        }

        /* Style for the results box */
        .result-box-bg {
            background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
            border: 1px solid #6366f1;
            padding: 2rem;
            border-radius: 0.75rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
        }
        
        /* Streamlit Component Styling */
        .stButton>button {
            color: white !important;
            background-color: #1d4ed8 !important; /* Blue 700 */
            border-color: #1e40af !important; /* Blue 800 */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.15s ease-in-out !important;
            border-radius: 0.5rem;
            font-weight: 600;
        }
        .stButton>button:hover {
             background-color: #2563eb !important; /* Blue 600 */
             transform: scale(1.02);
        }
        /* Selectbox styling to match theme */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #2563eb !important; /* Blue 600 */
            color: white !important;
            border: 1px solid #60a5fa !important;
            border-radius: 0.5rem;
        }
        /* File Uploader styling */
        .stFileUploader label {
            color: #f472b6 !important; /* Pink 400 */
            font-weight: 600;
        }
        /* Custom horizontal rule for separation */
        .header-separator {
            border: 0;
            height: 5px; /* Increased width/height */
            background-image: linear-gradient(to right, rgba(255, 255, 255, 0), #f97316, rgba(255, 255, 255, 0)); /* Orange 500 */
            margin: 0;
            padding: 0;
            opacity: 0.8;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # --- RENDER HEADER ---
    render_header()
    
    # --- HEADER SEPARATOR LINE ---
    st.markdown('<div class="max-w-7xl mx-auto px-4 sm:px-8"><hr class="header-separator"></div>', unsafe_allow_html=True)
    
    # --- Main Content Area - Reverted to 3-Column Layout ---
    st.markdown('<div class="max-w-7xl mx-auto px-4 sm:px-8">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    # 1. ABOUT SECTION BOX (Col 1)
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

    # 2. UPLOAD DATASET SECTION BOX (Col 2)
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


    # 3. ALGORITHM SELECTION BOX (Col 3)
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
        
        # Get models based on selected type
        current_models = model_options.get(analysis_type, {})
        
        # Determine disabled state for model selection
        is_model_select_disabled = (analysis_type != 'supervised_learning')

        selected_model_key = st.selectbox(
            "Select the model to be used:",
            options=list(current_models.keys()),
            format_func=lambda x: current_models[x],
            index=0,
            disabled=is_model_select_disabled,
            key="model_select"
        )
        
        # Ensure the button is disabled if no file or a placeholder model is selected
        is_button_disabled = uploaded_file is None or selected_model_key == 'coming_soon'

        st.button(
            "Run Evaluation", 
            key="run_analysis_button",
            disabled=is_button_disabled
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ANALYSIS LOGIC BLOCK (Run on Button Click) ---
    if st.session_state.run_analysis_button and uploaded_file is not None and not is_button_disabled:
        
        # Read the file and calculate a hash for caching
        file_content = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(file_content))
        # Use content hash as part of the cache key
        df_hash = hash(file_content) 

        with st.spinner(f"Running {current_models[selected_model_key]} on {uploaded_file.name}... (This may take a moment on the first run)"):
            status, message, metrics, pipeline = process_and_train_model(
                df_hash, 
                df, 
                selected_model_key
            )
            
            # Store the trained pipeline in session state for live prediction
            st.session_state['trained_pipeline'] = pipeline
            st.session_state['last_metrics'] = metrics
            st.session_state['last_status'] = status
            st.session_state['last_message'] = message
            st.session_state['last_model'] = current_models[selected_model_key]

    # --- 4. RESULTS SECTION BOX (Full Width) ---
    st.markdown("""
        <div class="result-box-bg w-full mt-8">
            <h2 class="text-2xl font-bold text-white mb-6 border-b-2 pb-2 border-indigo-300 inline-block">
                Evaluation Results
            </h2>
    """, unsafe_allow_html=True)

    if 'last_status' in st.session_state and st.session_state.last_status == "Success":
        metrics = st.session_state.last_metrics
        st.markdown(f"""
            <h3 class="text-xl font-bold text-green-300 mb-4">{st.session_state.last_message}</h3>
            <p class="text-gray-200 mb-2">Model Used: <code class="text-pink-300 font-bold">{st.session_state.last_model}</code></p>
            <p class="text-gray-200 mb-4">Pipeline Time: <code class="text-pink-300 font-bold">{metrics['Pipeline_Time']}</code></p>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-white">
                <div>
                    <h4 class="font-bold text-indigo-200 mb-2">CLASSIFICATION SUMMARY</h4>
                    <p>Accuracy: <span class="text-green-300">{metrics['Overall_Accuracy']}</span></p>
                    <p>Kappa Statistics: <span class="text-yellow-300">{metrics['Kappa_Statistics']}</span></p>
                    <p>MCC: <span class="text-yellow-300">{metrics['MCC']}</span></p>
                    <p>ROC Area: <span class="text-yellow-300">{metrics['ROC_Area']}</span></p>
                    <p>PRC Area: <span class="text-yellow-300">{metrics['PRC_Area']}</span></p>
                </div>
                <div>
                    <h4 class="font-bold text-indigo-200 mb-2">ERROR & SIZE</h4>
                    <p>MAE: <span class="text-red-300">{metrics['Mean_Absolute_Error']}</span></p>
                    <p>RMSE: <span class="text-red-300">{metrics['Root_Mean_Square_Error']}</span></p>
                    <p>RAE: <span class="text-red-300">{metrics['Relative_Absolute_Error']}</span></p>
                    <p>Test Set Size: <span class="text-cyan-300">{metrics['Test_Set_Size']}</span></p>
                </div>
                <div>
                    <h4 class="font-bold text-indigo-200 mb-2">POSITIVE CLASS METRICS</h4>
                    <p>Precision: <span class="text-green-300">{metrics['Detailed_Classification_Metrics']['Precision (Positive Class)']}</span></p>
                    <p>Recall (TP Rate): <span class="text-green-300">{metrics['Detailed_Classification_Metrics']['Recall (Positive Class)']}</span></p>
                    <p>F1 Score: <span class="text-green-300">{metrics['Detailed_Classification_Metrics']['F1_Score (F-Measure)']}</span></p>
                    <p>FP Rate: <span class="text-red-300">{metrics['Detailed_Classification_Metrics']['FP_Rate']}</span></p>
                </div>
            </div>

            <div class="mt-6">
                <h4 class="font-bold text-indigo-200 mb-2">CLASSIFICATION COUNT</h4>
                <p>Correctly Classified: <span class="text-green-300">{metrics['Correctly_Classified_Instances']}</span></p>
                <p>Incorrectly Classified: <span class="text-red-300">{metrics['Incorrectly_Classified_Instances']}</span></p>
            </div>
            <p class="mt-4 text-xs italic text-gray-400">Confusion Matrix: [[TN, FP], [FN, TP]] -> {metrics['Confusion_Matrix (TN, FP, FN, TP)']}</p>

        """, unsafe_allow_html=True)
    
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
                user_review = st.text_area("Enter your review here:", key="user_review_input", height=100, label_visibility="collapsed")
            
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