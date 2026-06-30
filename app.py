import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

from model_utils import (
    load_models, predict_absa, identify_aspects,
    ASPECT_KEYWORDS, SENTIMENT_COLORS, AVAILABLE_MODELS,
    get_sentiment_score, get_overall_sentiment, get_available_models
)
from src.config import load_config

# --- Load config ---
_cfg = load_config()
APP_ICON = _cfg.get('app', {}).get('page_icon', '🧠')

# --- Page Configuration ---
st.set_page_config(
    page_title="ABSA Telecom Analyzer",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B1F3B 0%, #162447 50%, #1B1F3B 100%) !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] li,
    [data-testid="stSidebar"] label {
        color: #C8CDD8 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
        margin: 12px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }

    /* Nav buttons styled as tabs */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: none !important;
        color: #9CA3AF !important;
        text-align: left !important;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        justify-content: flex-start !important;
        transition: all 0.2s !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(108, 99, 255, 0.15) !important;
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.3) !important;
    }

    /* Sidebar selectbox styling */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #9CA3AF !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(30, 34, 60, 0.9) !important;
        border: 1px solid rgba(108, 99, 255, 0.4) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div > div {
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: #9CA3AF !important;
    }

    /* Sidebar logo */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.3rem 0 0.3rem 0;
    }
    .sidebar-logo-icon {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%);
        border-radius: 12px;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }
    .sidebar-logo-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFFFFF !important;
        letter-spacing: 0.5px;
    }
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #7B8CDE !important;
        font-weight: 500;
        margin-top: 2px;
    }
    .sidebar-desc {
        font-size: 0.75rem;
        color: #6B7280 !important;
        margin-top: 2px;
        margin-bottom: 24px;
    }

    /* Sidebar model summary */
    .model-summary-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6B7280 !important;
        margin-bottom: 10px;
        margin-top: 10px;
    }
    .select-model-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6B7280 !important;
        margin-bottom: 16px;
        margin-top: 4px;
    }
    .model-label {
        font-size: 0.8rem;
        color: #9CA3AF !important;
        font-weight: 500;
        margin-bottom: 2px;
    }
    .model-value {
        font-size: 0.9rem;
        color: #E5E7EB !important;
        font-weight: 400;
        margin-bottom: 14px;
    }
    .model-accuracy {
        font-size: 2rem;
        font-weight: 700;
        color: #4ADE80 !important;
        margin-bottom: 2px;
        line-height: 1.1;
    }
    .model-accuracy-label {
        font-size: 0.75rem;
        color: #6B7280 !important;
        margin-bottom: 16px;
    }
    .model-aspects-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E5E7EB !important;
    }
    .sentiment-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-positive { background-color: #4ADE80; }
    .dot-neutral { background-color: #FBBF24; }
    .dot-negative { background-color: #F87171; }
    .sentiment-class-item {
        font-size: 0.8rem;
        color: #C8CDD8 !important;
        margin: 4px 0;
    }

    /* ===== MAIN CONTENT ===== */
    .main-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 4px;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .metric-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
    .metric-icon-purple { background: #EDE9FE; }
    .metric-icon-green { background: #D1FAE5; }
    .metric-icon-blue { background: #DBEAFE; }
    .metric-icon-orange { background: #FEF3C7; }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 4px;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-desc {
        font-size: 0.85rem;
        color: #6B7280;
        margin-bottom: 16px;
    }

    /* Results table */
    .results-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }
    .results-table th {
        background: #F9FAFB;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
        color: #4B5563;
        border-bottom: 1px solid #E5E7EB;
    }
    .results-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #F3F4F6;
        color: #374151;
    }
    .results-table tr:last-child td {
        border-bottom: none;
    }

    /* Sentiment badges */
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-positive { background: #D1FAE5; color: #065F46; }
    .badge-negative { background: #FEE2E2; color: #991B1B; }
    .badge-neutral { background: #FEF3C7; color: #92400E; }

    /* Output card */
    .output-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* Example query label */
    .example-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        margin-bottom: 8px;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 1px solid #E5E7EB;
        margin: 1.5rem 0;
    }

    /* Overall sentiment */
    .overall-box { text-align: center; padding: 10px; }
    .overall-score { font-size: 2.2rem; font-weight: 700; color: #1F2937; }
    .overall-score-sub { font-size: 1rem; color: #9CA3AF; font-weight: 400; }
    .overall-label { font-size: 0.8rem; color: #6B7280; margin-top: 4px; }

    /* Summary */
    .summary-title { font-size: 0.85rem; font-weight: 600; color: #1F2937; margin-bottom: 6px; }
    .summary-positive { color: #059669; font-size: 0.8rem; font-weight: 600; }
    .summary-negative { color: #DC2626; font-size: 0.8rem; font-weight: 600; }
    .summary-item { font-size: 0.8rem; color: #374151; margin: 3px 0; }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stMetric"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Helpers ---
def get_badge_html(sentiment):
    cls = f"badge-{sentiment.lower()}"
    return f'<span class="badge {cls}">{sentiment}</span>'


# --- SIDEBAR ---
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">{APP_ICON}</div>
            <div>
                <div class="sidebar-logo-text">ABSA</div>
            </div>
        </div>
        <div class="sidebar-subtitle">Telecom Analyzer</div>
        <div class="sidebar-desc">Aspect Based Sentiment Analysis</div>
        """, unsafe_allow_html=True)

        # Navigation buttons
        nav_items = [
            ("🏠", "Dashboard"),
            ("💬", "Single Feedback Analysis"),
            ("📁", "Batch CSV Analysis"),
            ("📊", "Insights Dashboard"),
            ("ℹ️", "About Model")
        ]

        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = "Dashboard"

        for icon, label in nav_items:
            is_active = st.session_state['current_page'] == label
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state['current_page'] = label
                st.rerun()

        st.markdown("---")

        # Model Selector
        available = get_available_models()
        model_names = list(available.keys())

        if len(model_names) > 1:
            st.markdown('<div class="select-model-title">SELECT MODEL</div>', unsafe_allow_html=True)
            selected_model = st.selectbox(
                "Select Model",
                options=model_names,
                index=model_names.index(st.session_state.get('selected_model', model_names[0])),
                label_visibility="collapsed",
                key="model_selector"
            )
            st.session_state['selected_model'] = selected_model
            model_info = available[selected_model]
            st.caption(model_info['description'])
        else:
            st.session_state['selected_model'] = model_names[0] if model_names else 'Tuned Logistic Regression'

        st.markdown("---")

        # Model Summary
        current_model_name = st.session_state.get('selected_model', 'Tuned Logistic Regression')
        current_f1 = available.get(current_model_name, {}).get('f1', '84.9%')

        st.markdown(f"""
        <div class="model-summary-title">MODEL SUMMARY</div>
        <div class="model-label">Model</div>
        <div class="model-value">{current_model_name}</div>
        <div class="model-accuracy">{current_f1}</div>
        <div class="model-accuracy-label">Accuracy (F1)</div>
        <div class="model-label">Aspects Supported</div>
        <div class="model-aspects-value">22</div>
        <br>
        <div class="model-label">Sentiment Classes</div>
        <div class="sentiment-class-item"><span class="sentiment-dot dot-positive"></span>Positive</div>
        <div class="sentiment-class-item"><span class="sentiment-dot dot-neutral"></span>Neutral</div>
        <div class="sentiment-class-item"><span class="sentiment-dot dot-negative"></span>Negative</div>
        <br>
        <div class="model-label">Dataset Size</div>
        <div class="model-value">13,100 Feedback Records</div>
        """, unsafe_allow_html=True)

    return st.session_state['current_page']


# --- PAGE: DASHBOARD ---
def page_dashboard(model, vectorizer):
    selected_model = st.session_state.get('selected_model', 'Tuned Logistic Regression')
    available = get_available_models()
    current_f1 = available.get(selected_model, {}).get('f1', '84.9%')

    st.markdown('<div class="main-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Get actionable insights from customer feedback</div>', unsafe_allow_html=True)

    # Top metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon metric-icon-purple">📈</div>
            <div class="metric-value">{current_f1}</div>
            <div class="metric-label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-icon metric-icon-green">🎯</div>
            <div class="metric-value">22</div>
            <div class="metric-label">Aspects Supported</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-icon metric-icon-blue">😊</div>
            <div class="metric-value">3</div>
            <div class="metric-label">Sentiment Classes</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-icon metric-icon-orange">💬</div>
            <div class="metric-value">13,100</div>
            <div class="metric-label">Dataset Size</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Two columns
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">🔍 Try Single Feedback Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Enter customer feedback below and get aspect-level sentiment predictions.</div>', unsafe_allow_html=True)

        feedback_input = st.text_input(
            "feedback",
            placeholder="Type or paste customer feedback here...",
            label_visibility="collapsed",
            key="dashboard_input"
        )

        st.markdown('<div class="example-label">Example Queries:</div>', unsafe_allow_html=True)
        examples = [
            "Internet speed is good but customer support is poor",
            "Very satisfied with the network coverage and data speed",
            "High call drops and slow internet, very disappointed"
        ]
        for i, ex in enumerate(examples):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state['dashboard_feedback'] = ex

        if 'dashboard_feedback' in st.session_state:
            feedback_input = st.session_state['dashboard_feedback']

        if feedback_input:
            start_time = time.time()
            results = predict_absa(feedback_input, model, vectorizer)
            elapsed = time.time() - start_time
            st.session_state['dash_results'] = results
            st.session_state['dash_input'] = feedback_input
            st.session_state['dash_time'] = elapsed

    with col_right:
        st.markdown('<div class="section-title">📊 Sample Analysis Output</div>', unsafe_allow_html=True)

        if 'dash_results' in st.session_state and st.session_state['dash_results']:
            results = st.session_state['dash_results']
            input_text = st.session_state.get('dash_input', '')

            st.markdown(f"""<div class="output-card" style="margin-bottom:14px;">
                <div style="font-size:0.8rem;font-weight:600;color:#4B5563;margin-bottom:6px;">Feedback</div>
                <div style="font-size:0.9rem;color:#1F2937;font-style:italic;">"{input_text}"</div>
            </div>""", unsafe_allow_html=True)

            table_html = '<table class="results-table"><thead><tr><th>Aspect / Service</th><th>Sentiment</th><th>Confidence</th></tr></thead><tbody>'
            for r in results:
                badge = get_badge_html(r['sentiment'])
                conf = f"{r['confidence']:.0%}" if r['confidence'] else "—"
                table_html += f'<tr><td>{r["aspect"]}</td><td>{badge}</td><td>{conf}</td></tr>'
            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown("")
            score = get_sentiment_score(results)
            overall = get_overall_sentiment(results)
            positive_aspects = [r['aspect'] for r in results if r['sentiment'] == 'Positive']
            negative_aspects = [r['aspect'] for r in results if r['sentiment'] == 'Negative']

            ocol1, ocol2 = st.columns([1, 1])
            with ocol1:
                st.markdown(f"""<div class="overall-box">
                    <div class="overall-label">Sentiment Score</div>
                    <div class="overall-score">{score}<span class="overall-score-sub">/100</span></div>
                    <div class="overall-label">Overall: {overall}</div>
                    <div class="overall-label" style="margin-top:6px;">⏱️ {st.session_state.get('dash_time', 0)*1000:.0f}ms</div>
                </div>""", unsafe_allow_html=True)
            with ocol2:
                summary_html = '<div style="padding:8px 0;">'
                summary_html += f'<div class="summary-title">Overall Sentiment</div>'
                summary_html += f'<div style="font-weight:600;color:#1F2937;margin-bottom:8px;">🔀 {overall}</div>'
                if positive_aspects:
                    summary_html += '<div class="summary-positive">Positive Aspects</div>'
                    for a in positive_aspects:
                        summary_html += f'<div class="summary-item">✅ {a}</div>'
                if negative_aspects:
                    summary_html += '<div class="summary-negative" style="margin-top:6px;">Negative Aspects</div>'
                    for a in negative_aspects:
                        summary_html += f'<div class="summary-item">⚠️ {a}</div>'
                summary_html += '</div>'
                st.markdown(summary_html, unsafe_allow_html=True)
        else:
            st.markdown("""<div class="output-card" style="text-align:center;padding:60px 20px;color:#9CA3AF;">
                <div style="font-size:2rem;margin-bottom:10px;">📊</div>
                <div>Enter feedback on the left to see analysis results here</div>
            </div>""", unsafe_allow_html=True)


# --- PAGE: SINGLE FEEDBACK ---
def page_single_analysis(model, vectorizer):
    st.markdown('<div class="main-title">Single Feedback Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Enter customer feedback below and get aspect-level sentiment predictions</div>', unsafe_allow_html=True)

    feedback_input = st.text_area(
        "feedback",
        placeholder="Type or paste customer feedback here...\n\ne.g., Internet speed is excellent but customer support is very slow.",
        height=130,
        label_visibility="collapsed"
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("🔍 Analyze Feedback", type="primary", use_container_width=True)

    if analyze_btn and feedback_input.strip():
        start_time = time.time()
        results = predict_absa(feedback_input.strip(), model, vectorizer)
        elapsed = time.time() - start_time
        if not results:
            st.warning("No aspects detected.")
            return

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        score = get_sentiment_score(results)
        overall = get_overall_sentiment(results)
        avg_conf = np.mean([r['confidence'] for r in results if r['confidence']])

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(results)}</div><div class="metric-label">Aspects Detected</div></div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{overall}</div><div class="metric-label">Overall Sentiment</div></div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{score}/100</div><div class="metric-label">Sentiment Score</div></div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{elapsed*1000:.0f}ms</div><div class="metric-label">⏱️ Inference Time</div></div>""", unsafe_allow_html=True)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        col_t, col_c = st.columns([1, 1], gap="large")
        with col_t:
            st.markdown('<div class="section-title">Aspect-Level Results</div>', unsafe_allow_html=True)
            table_html = '<table class="results-table"><thead><tr><th>Aspect</th><th>Sentiment</th><th>Confidence</th></tr></thead><tbody>'
            for r in results:
                badge = get_badge_html(r['sentiment'])
                conf = f"{r['confidence']:.0%}" if r['confidence'] else "—"
                table_html += f'<tr><td>{r["aspect"]}</td><td>{badge}</td><td>{conf}</td></tr>'
            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
            sentiments = [r['sentiment'] for r in results]
            sent_counts = pd.Series(sentiments).value_counts()
            fig = px.pie(values=sent_counts.values, names=sent_counts.index, color=sent_counts.index,
                         color_discrete_map=SENTIMENT_COLORS, hole=0.45)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence by Aspect</div>', unsafe_allow_html=True)
        df_r = pd.DataFrame(results)
        fig_bar = px.bar(df_r, x='aspect', y='confidence', color='sentiment', color_discrete_map=SENTIMENT_COLORS,
                         labels={'aspect': '', 'confidence': 'Confidence', 'sentiment': ''})
        fig_bar.update_layout(height=300, margin=dict(t=10, b=50, l=40, r=20), xaxis_tickangle=-20,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(gridcolor='#F3F4F6'))
        st.plotly_chart(fig_bar, use_container_width=True)

    elif analyze_btn:
        st.warning("Please enter feedback text to analyze.")


# --- PAGE: BATCH CSV ---
def page_batch_analysis(model, vectorizer):
    st.markdown('<div class="main-title">Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Upload a CSV file with customer feedback to analyze in bulk</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([1, 1], gap="large")
    with col_up:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
        st.markdown("")
        st.markdown("**CSV Requirements:**")
        st.markdown("- Must contain a column named 'feedback'")
        st.markdown("- One feedback entry per row")
        st.markdown("- Supported format: .csv")

    with col_info:
        if uploaded_file is None:
            st.markdown("##### Batch Analysis Insights (Sample)")
            st.info("Upload a CSV file to see batch analysis results with charts, heatmaps, and downloadable reports.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            feedback_col = None
            possible_names = ['feedback', 'feedback_text', 'text', 'review', 'comment', 'comments', 'message']
            for col in df.columns:
                if col.lower().strip() in possible_names:
                    feedback_col = col
                    break
            if feedback_col is None:
                feedback_col = st.selectbox("Select feedback column:", options=df.columns.tolist())
            else:
                st.success(f"Auto-detected column: **{feedback_col}**")

            with st.expander(f"Preview data ({len(df)} rows)"):
                st.dataframe(df.head(10), use_container_width=True)

            if st.button("🚀 Run Analysis", type="primary"):
                valid_df = df[df[feedback_col].notna() & (df[feedback_col].astype(str).str.strip() != '')].copy()
                if len(valid_df) == 0:
                    st.error("No valid feedback entries found.")
                    return

                all_results = []
                progress = st.progress(0)
                status_text = st.empty()
                batch_start = time.time()
                for idx, (_, row) in enumerate(valid_df.iterrows()):
                    text = str(row[feedback_col]).strip()
                    res = predict_absa(text, model, vectorizer)
                    for r in res:
                        r['feedback_text'] = text
                    all_results.extend(res)
                    progress.progress((idx + 1) / len(valid_df))
                    status_text.text(f"Processing {idx + 1}/{len(valid_df)}...")
                batch_elapsed = time.time() - batch_start
                progress.empty()
                status_text.empty()

                st.success(f"✅ Done! Processed {len(valid_df)} feedbacks in **{batch_elapsed:.1f}s** ({batch_elapsed/len(valid_df)*1000:.0f}ms avg per feedback)")

                st.session_state['batch_results'] = all_results
                st.session_state['batch_df'] = valid_df

            if 'batch_results' in st.session_state:
                all_results = st.session_state['batch_results']
                valid_df = st.session_state['batch_df']
                sentiments = [r['sentiment'] for r in all_results]
                sent_counts = pd.Series(sentiments).value_counts()

                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(valid_df)}</div><div class="metric-label">Total Feedback</div></div>""", unsafe_allow_html=True)
                with mc2:
                    pos_c = sent_counts.get('Positive', 0)
                    st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#10B981;">{pos_c}</div><div class="metric-label">Positive ({pos_c/len(all_results)*100:.0f}%)</div></div>""", unsafe_allow_html=True)
                with mc3:
                    neu_c = sent_counts.get('Neutral', 0)
                    st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#F59E0B;">{neu_c}</div><div class="metric-label">Neutral ({neu_c/len(all_results)*100:.0f}%)</div></div>""", unsafe_allow_html=True)
                with mc4:
                    neg_c = sent_counts.get('Negative', 0)
                    st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#EF4444;">{neg_c}</div><div class="metric-label">Negative ({neg_c/len(all_results)*100:.0f}%)</div></div>""", unsafe_allow_html=True)

                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                results_df = pd.DataFrame(all_results)

                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
                    fig_pie = px.pie(values=sent_counts.values, names=sent_counts.index, color=sent_counts.index,
                                    color_discrete_map=SENTIMENT_COLORS, hole=0.4)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with r1c2:
                    st.markdown('<div class="section-title">Top Aspects by Frequency</div>', unsafe_allow_html=True)
                    asp_counts = results_df['aspect'].value_counts().head(8).reset_index()
                    asp_counts.columns = ['Aspect', 'Count']
                    fig_asp = px.bar(asp_counts, x='Count', y='Aspect', orientation='h', color='Count', color_continuous_scale='Blues')
                    fig_asp.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), yaxis={'categoryorder': 'total ascending'}, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig_asp, use_container_width=True)

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    st.markdown('<div class="section-title">Positive vs Negative by Aspect</div>', unsafe_allow_html=True)
                    top_asps = results_df['aspect'].value_counts().head(6).index.tolist()
                    melted = results_df[results_df['aspect'].isin(top_asps)].groupby(['aspect', 'sentiment']).size().reset_index(name='count')
                    fig_st = px.bar(melted, x='aspect', y='count', color='sentiment', color_discrete_map=SENTIMENT_COLORS, barmode='group')
                    fig_st.update_layout(height=300, margin=dict(t=10, b=50, l=40, r=10), xaxis_tickangle=-20, legend=dict(orientation='h', y=1.1))
                    st.plotly_chart(fig_st, use_container_width=True)

                with r2c2:
                    st.markdown('<div class="section-title">Aspect-wise Sentiment Heatmap</div>', unsafe_allow_html=True)
                    hm = results_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
                    for s in ['Positive', 'Neutral', 'Negative']:
                        if s not in hm.columns:
                            hm[s] = 0
                    hm = hm[['Positive', 'Neutral', 'Negative']].head(8)
                    fig_hm = go.Figure(data=go.Heatmap(z=hm.values, x=['Positive', 'Neutral', 'Negative'], y=hm.index.tolist(),
                                                       colorscale='RdYlGn', text=hm.values, texttemplate="%{text}", textfont={"size": 11}))
                    fig_hm.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_hm, use_container_width=True)

                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown('<div class="section-title">✅ Top Positive Aspects</div>', unsafe_allow_html=True)
                    pos_asp = results_df[results_df['sentiment'] == 'Positive']['aspect'].value_counts().head(5)
                    for i, (a, c) in enumerate(pos_asp.items(), 1):
                        st.markdown(f"{i}. **{a}** — {c} mentions")
                with ic2:
                    st.markdown('<div class="section-title">⚠️ Top Problem Areas</div>', unsafe_allow_html=True)
                    neg_asp = results_df[results_df['sentiment'] == 'Negative']['aspect'].value_counts().head(5)
                    for i, (a, c) in enumerate(neg_asp.items(), 1):
                        st.markdown(f"{i}. **{a}** — {c} mentions")

                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Recent Feedback (Filtered)</div>', unsafe_allow_html=True)
                fc1, fc2 = st.columns(2)
                with fc1:
                    f_aspects = st.multiselect("Aspects", sorted(results_df['aspect'].unique()), default=sorted(results_df['aspect'].unique()), key="ba_asp")
                with fc2:
                    f_sents = st.multiselect("Sentiments", ['Positive', 'Neutral', 'Negative'], default=['Positive', 'Neutral', 'Negative'], key="ba_sent")

                filtered = results_df[(results_df['aspect'].isin(f_aspects)) & (results_df['sentiment'].isin(f_sents))].copy()
                filtered['Confidence'] = filtered['confidence'].apply(lambda x: f"{x:.0%}" if x else "—")
                st.dataframe(filtered[['feedback_text', 'aspect', 'sentiment', 'Confidence']].rename(columns={
                    'feedback_text': 'Feedback', 'aspect': 'Top Aspect', 'sentiment': 'Sentiment'
                }).head(50), use_container_width=True, height=280)

                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                dc1, dc2 = st.columns(2)
                with dc1:
                    dl_df = pd.DataFrame([{'feedback_text': r.get('feedback_text', ''), 'aspect': r['aspect'], 'sentiment': r['sentiment'],
                                           'confidence': f"{r['confidence']:.4f}" if r['confidence'] else ''} for r in all_results])
                    st.download_button("📥 Download Predictions CSV", dl_df.to_csv(index=False), "absa_predictions.csv", "text/csv", use_container_width=True)
                with dc2:
                    summary = f"ABSA Summary\n{'='*40}\nTotal: {len(valid_df)}\nPositive: {pos_c}\nNeutral: {neu_c}\nNegative: {neg_c}\n"
                    st.download_button("📥 Download Summary Report", summary, "absa_summary.txt", "text/plain", use_container_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")


# --- PAGE: INSIGHTS ---
def page_insights(model, vectorizer):
    selected_model = st.session_state.get('selected_model', 'Tuned Logistic Regression')
    available = get_available_models()
    current_f1 = available.get(selected_model, {}).get('f1', '84.9%')

    # Model-specific performance data
    model_perf = {
        'Tuned Logistic Regression': {
            'metrics': [('Weighted F1', '84.9%'), ('Precision', '85.1%'), ('Recall', '84.8%'), ('Train F1', '87.3%'), ('Val F1', '84.2%')],
            'classes': [('Negative', '80.1%', '85.0%', '82.5%'), ('Neutral', '91.6%', '84.0%', '87.6%'), ('Positive', '83.7%', '85.4%', '84.5%')],
            'config': 'Logistic Regression (C=0.5, lbfgs, L2)'
        },
        'Naive Bayes': {
            'metrics': [('Weighted F1', '84.6%'), ('Precision', '85.3%'), ('Recall', '84.5%'), ('Train F1', '86.7%'), ('Val F1', '82.7%')],
            'classes': [('Negative', '79.6%', '86.5%', '82.9%'), ('Neutral', '95.5%', '79.9%', '87.0%'), ('Positive', '81.2%', '87.0%', '84.0%')],
            'config': 'MultinomialNB (alpha=0.5, SMOTE)'
        },
        'SGD-SVM': {
            'metrics': [('Weighted F1', '85.3%'), ('Precision', '85.4%'), ('Recall', '85.2%'), ('Train F1', '90.1%'), ('Val F1', '84.9%')],
            'classes': [('Negative', '82.2%', '83.9%', '83.0%'), ('Neutral', '90.5%', '85.4%', '87.9%'), ('Positive', '83.4%', '86.2%', '84.8%')],
            'config': 'SGDClassifier (modified_huber, balanced)'
        },
        'DistilBERT': {
            'metrics': [('Weighted F1', '95.6%'), ('Precision', '95.7%'), ('Recall', '95.6%'), ('Train F1', '97.0%'), ('Val F1', '96.0%')],
            'classes': [('Negative', '93.9%', '97.9%', '95.9%'), ('Neutral', '96.3%', '94.7%', '95.5%'), ('Positive', '96.7%', '94.5%', '95.6%')],
            'config': 'DistilBERT (fine-tuned, 3 epochs, lr=2e-5)'
        }
    }

    perf = model_perf.get(selected_model, model_perf['Tuned Logistic Regression'])

    st.markdown('<div class="main-title">Insights Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Performance metrics for: {selected_model}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        perf_df = pd.DataFrame(perf['metrics'], columns=['Metric', 'Score'])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        st.markdown("")
        st.markdown('<div class="section-title">Per-Class Performance</div>', unsafe_allow_html=True)
        cls_df = pd.DataFrame(perf['classes'], columns=['Class', 'Precision', 'Recall', 'F1'])
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
        config = {'Algorithm': perf['config'], 'Vectorizer': 'TF-IDF (uni+bigrams, 12,040 features)',
                  'Enhancement': 'VADER correction for low-confidence', 'Preprocessing': 'Clause splitting + lemmatization'}
        for k, v in config.items():
            st.markdown(f"**{k}:** {v}")
        st.markdown("")
        st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)
        stats = {'Records': '13,100 aspect-level rows', 'Unique Feedbacks': '6,750', 'Aspects': '22 categories',
                 'Balance': 'Pos 35.7% | Neu 33.1% | Neg 31.2%'}
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")


# --- PAGE: ABOUT ---
def page_about():
    st.markdown('<div class="main-title">About Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Technical documentation and supported aspects</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-title">Supported Aspects (22)</div>', unsafe_allow_html=True)
        aspects_sorted = sorted(ASPECT_KEYWORDS.keys())
        mid = len(aspects_sorted) // 2
        ac1, ac2 = st.columns(2)
        with ac1:
            for a in aspects_sorted[:mid]:
                st.markdown(f"• {a}")
        with ac2:
            for a in aspects_sorted[mid:]:
                st.markdown(f"• {a}")

    with col2:
        st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
1. **Input** → Raw customer feedback  
2. **Clause Splitting** → Split on but, however, though...  
3. **Aspect Detection** → Regex keyword matching  
4. **Preprocessing** → Tokenize, lemmatize, remove stopwords  
5. **TF-IDF** → Vectorize with aspect token prefix  
6. **Model** → Tuned Logistic Regression  
7. **VADER** → Correction for uncertain predictions  
8. **Output** → Aspect + Sentiment + Confidence  
        """)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Known Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
- Short text (< 5 words) may lack TF-IDF signal  
- Sarcasm/irony not detected  
- Domain-specific (telecom only)  
- Keyword-dependent aspect detection  
- Word order not captured by TF-IDF  
    """)


# --- MAIN ---
def main():
    page = render_sidebar()

    # Load the selected model
    selected_model = st.session_state.get('selected_model', 'Tuned Logistic Regression')
    try:
        model, vectorizer, label_encoder = load_models(selected_model)
    except FileNotFoundError:
        st.error("⚠️ Model files not found.")
        st.stop()

    if "Dashboard" in page and "Insights" not in page:
        page_dashboard(model, vectorizer)
    elif "Single" in page:
        page_single_analysis(model, vectorizer)
    elif "Batch" in page:
        page_batch_analysis(model, vectorizer)
    elif "Insights" in page:
        page_insights(model, vectorizer)
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
