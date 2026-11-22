import pandas as pd
from src.persian_fin_nlp import (
    CodalRAGEngine,
    normalize_persian_text,
    score_financial_sentiment,
)


def test_persian_normalization():
    raw = "<p>افزايش نرخ فروش شركت فولاد مباركه به ميزان ۲۵٪</p>"
    norm = normalize_persian_text(raw)
    assert "<p>" not in norm
    assert "افزایش نرخ فروش شرکت فولاد مبارکه" in norm


def test_financial_sentiment_scoring():
    bullish_doc = "افشای اطلاعات بااهمیت گروه الف: مجوز افزایش نرخ فروش محصولات صادراتی و رشد درآمد شرکت"
    bearish_doc = "اطلاعیه توقف خط تولید به دلیل محدودیت برق و احتمال تعدیل منفی سود"

    res_bull = score_financial_sentiment(bullish_doc)
    res_bear = score_financial_sentiment(bearish_doc)

    assert res_bull["sentiment_label"] == "Bullish"
    assert res_bull["sentiment_score"] > 0.5
    assert res_bear["sentiment_label"] == "Bearish"
    assert res_bear["sentiment_score"] < -0.5


def test_codal_rag_retrieval():
    df = pd.DataFrame([
        {"ticker": "فولاد", "title": "مجوز افزایش نرخ فروش ورق گرم", "summary": "افزایش ۱۵ درصدی نرخ محصولات"},
        {"ticker": "خودرو", "title": "اطلاعیه توقف خط تولید به دلیل قطعی برق", "summary": "کاهش تولید روزانه"},
        {"ticker": "فملی", "title": "افزایش سرمایه از محل سود انباشته", "summary": "تصویب افزایش سرمایه ۲۰۰ درصدی"},
    ])
    rag = CodalRAGEngine().fit(df)
    hits = rag.query("کدام شرکت افزایش سرمایه از سود انباشته دارد؟", top_k=2)

    assert not hits.empty
    assert hits.iloc[0]["ticker"] == "فملی"