import os
import pandas as pd
import plotly.express as px
import streamlit as st

from src.codal_loader import build_codal_dataset
from src.impact_analyzer import analyze_sentiment_impact, extract_top_drivers
from src.persian_fin_nlp import CodalRAGEngine, score_financial_sentiment

st.set_page_config(
    page_title="TSE Codal NLP | Financial Sentiment & RAG Engine",
    page_icon="📰",
    layout="wide",
)

TICKER_LABELS = {
    "فولاد": "FOOLAD (فولاد)",
    "فملی": "FEMELI (فملی)",
    "خودرو": "KHODRO (خودرو)",
    "خساپا": "KHESAPA (خساپا)",
    "شستا": "SHASTA (شستا)",
    "شپنا": "SHAPNA (شپنا)",
    "وبملت": "VBMELAT (وبملت)",
    "فارس": "FARS (فارس)",
}


@st.cache_data
def load_and_score_codal() -> pd.DataFrame:
    df = build_codal_dataset(force_refresh=False)
    # بازمحاسبه امتیاز سنتیمنت با آخرین واژگان تخصصی
    scores = [score_financial_sentiment(f"{r['title']} {r['summary']}") for _, r in df.iterrows()]
    df["sentiment_score"] = [s["sentiment_score"] for s in scores]
    df["sentiment_label"] = [s["sentiment_label"] for s in scores]
    df["matched_drivers"] = [", ".join(s["matched_drivers"]) if s["matched_drivers"] else "None" for s in scores]
    return df


st.title("📰 TSE-Codal-NLP: Persian Financial Sentiment, RAG Search & Market Impact")
st.markdown(
    "Automated NLP pipeline analyzing official **Codal** corporate disclosures (`گزارش‌های کدال`), "
    "retrieving relevant filings via **TF-IDF RAG**, and measuring post-disclosure abnormal returns."
)

df_codal = load_and_score_codal()
rag_engine = CodalRAGEngine().fit(df_codal, text_columns=["ticker", "title", "summary", "matched_drivers"])
impact_df = analyze_sentiment_impact(df_codal)
drivers_df = extract_top_drivers(df_codal, top_n=10)

# --- کارت‌های خلاصه آماری ---
bull_sub = df_codal[df_codal["sentiment_label"] == "Bullish"]
bear_sub = df_codal[df_codal["sentiment_label"] == "Bearish"]

bull_5d = float(bull_sub["ret_5d"].mean() * 100.0) if not bull_sub.empty else 0.0
bear_1d = float(bear_sub["ret_1d"].mean() * 100.0) if not bear_sub.empty else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Codal Disclosures", f"{len(df_codal):,}", f"{df_codal['ticker'].nunique()} TSE Leaders")
c2.metric("Classified Sentiment Events", f"{len(bull_sub) + len(bear_sub)}", f"{len(bull_sub)} Bullish | {len(bear_sub)} Bearish")
c3.metric("Bullish Avg 5-Day Return", f"{bull_5d:+.2f}%", f"Win Rate: {(bull_sub['ret_5d'] > 0).mean() * 100:.1f}%" if not bull_sub.empty else "-")
c4.metric("Bearish Avg 1-Day Shock", f"{bear_1d:+.2f}%", "Immediate Market Reaction")

tab1, tab2, tab3 = st.tabs([
    "🔍 Codal RAG Q&A & Semantic Search",
    "📊 Sentiment Event Study & Impact",
    "🏢 Ticker Disclosure Explorer",
])

with tab1:
    st.subheader("Ask Questions Across Official Codal Filings (Persian NLP Retrieval)")
    preset_questions = [
        "کدام شرکت‌ها پیشنهاد یا مجوز افزایش سرمایه دارند؟",
        "اطلاعیه‌های شفاف‌سازی در خصوص نوسان قیمت سهام کدام نمادها هستند؟",
        "شرکت‌های دارای تعلیق، توقف نماد یا محدودیت تولید",
        "تصمیمات مجمع عمومی عادی و تقسیم سود نقدی",
        "تغییر نرخ فروش محصولات و گزارش فعالیت ماهانه",
    ]
    selected_preset = st.selectbox("Select a Sample Analytical Query:", ["(Custom Query)"] + preset_questions)
    default_q = "" if selected_preset == "(Custom Query)" else selected_preset
    user_query = st.text_input("Enter your Persian search query:", value=default_q, placeholder="مثلاً: افزایش سرمایه از سود انباشته فولاد")
    top_k = st.slider("Number of Top Matching Disclosures:", min_value=3, max_value=15, value=7)

    if user_query.strip():
        hits = rag_engine.query(user_query, top_k=top_k)
        if hits.empty:
            st.warning("No matching disclosures found for this query.")
        else:
            show_cols = [
                "relevance_score", "ticker", "publish_jalali", "sentiment_label",
                "sentiment_score", "title", "matched_drivers", "ret_1d", "ret_5d",
            ]
            st.dataframe(
                hits[show_cols].style.format({
                    "relevance_score": "{:.1f}%",
                    "sentiment_score": "{:+.2f}",
                    "ret_1d": "{:+.2%}",
                    "ret_5d": "{:+.2%}",
                }),
                use_container_width=True,
            )

with tab2:
    st.subheader("Post-Disclosure Return Hypothesis Test by Sentiment Category")
    st.dataframe(impact_df, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        melted = impact_df.melt(
            id_vars=["Sentiment"],
            value_vars=["Avg 1D Ret (%)", "Avg 3D Ret (%)", "Avg 5D Ret (%)"],
            var_name="Horizon",
            value_name="Average Return (%)",
        )
        fig_ret = px.bar(
            melted,
            x="Horizon",
            y="Average Return (%)",
            color="Sentiment",
            barmode="group",
            color_discrete_map={"Bullish": "#00cc96", "Neutral": "#636efa", "Bearish": "#ef553b"},
            title="Average Post-Disclosure Return Across Horizons",
            template="plotly_white",
        )
        fig_ret.add_hline(y=0, line_color="black", line_width=1)
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_right:
        if not drivers_df.empty:
            fig_drv = px.bar(
                drivers_df.sort_values("Occurrences", ascending=True),
                x="Occurrences",
                y="Driver Phrase",
                orientation="h",
                title="Top Extracted Fundamental Drivers in Codal Filings",
                template="plotly_white",
                color_discrete_sequence=["#1f77b4"],
            )
            st.plotly_chart(fig_drv, use_container_width=True)

with tab3:
    st.subheader("Inspect Corporate Disclosures & Market Reactions by Ticker")
    f_col1, f_col2 = st.columns(2)
    ticker_choice = f_col1.selectbox(
        "Filter by Ticker:",
        options=["ALL"] + list(df_codal["ticker"].unique()),
        format_func=lambda x: "All Tickers" if x == "ALL" else TICKER_LABELS.get(x, x),
    )
    sent_choice = f_col2.selectbox("Filter by Sentiment:", options=["ALL", "Bullish", "Neutral", "Bearish"])

    filtered = df_codal.copy()
    if ticker_choice != "ALL":
        filtered = filtered[filtered["ticker"] == ticker_choice]
    if sent_choice != "ALL":
        filtered = filtered[filtered["sentiment_label"] == sent_choice]

    disp_cols = [
        "ticker", "publish_jalali", "sentiment_label", "sentiment_score",
        "title", "matched_drivers", "ret_1d", "ret_3d", "ret_5d",
    ]
    st.dataframe(
        filtered[disp_cols].style.format({
            "sentiment_score": "{:+.2f}",
            "ret_1d": "{:+.2%}",
            "ret_3d": "{:+.2%}",
            "ret_5d": "{:+.2%}",
        }),
        use_container_width=True,
    )