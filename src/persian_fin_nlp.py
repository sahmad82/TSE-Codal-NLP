import re
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# واژگان تخصصی مثبت در اطلاعیه‌های رسمی کدال و پیام‌های ناظر
BULLISH_PHRASES: Dict[str, float] = {
    "افزایش نرخ فروش": 2.5,
    "نرخ فروش": 1.6,
    "تغییر بیش از ۱۰ درصد": 1.8,
    "تغییر بیش از 10 درصد": 1.8,
    "تعدیل مثبت": 2.5,
    "رشد درآمد": 2.0,
    "رشد فروش": 2.0,
    "رکورد فروش": 2.5,
    "انعقاد قرارداد": 1.9,
    "قرارداد جدید": 1.9,
    "برنده مناقصه": 2.2,
    "سود انباشته": 1.7,
    "افزایش سرمایه": 1.5,
    "مجوز افزایش سرمایه": 2.0,
    "ثبت افزایش سرمایه": 1.6,
    "بهره برداری": 1.7,
    "طرح توسعه": 1.5,
    "تصمیمات مجمع عمومی عادی": 1.4,
    "تقسیم سود": 1.6,
    "صادرات": 1.3,
    "بازگشایی نماد": 1.4,
    "رفع تعلیق": 1.9,
}

# واژگان تخصصی منفی و هشداردهنده در اطلاعیه‌های کدال و پیام‌های ناظر
BEARISH_PHRASES: Dict[str, float] = {
    "توقف خط تولید": -2.5,
    "توقف تولید": -2.3,
    "کاهش نرخ فروش": -2.5,
    "کاهش نرخ": -2.0,
    "تعدیل منفی": -2.5,
    "زیان انباشته": -2.2,
    "زیان خالص": -2.0,
    "کاهش فروش": -1.8,
    "کاهش درآمد": -1.8,
    "محدودیت برق": -1.8,
    "محدودیت گاز": -1.8,
    "قطعی برق": -1.8,
    "جریمه": -1.7,
    "دعوی حقوقی": -1.6,
    "فسخ قرارداد": -2.2,
    "تعلیق": -2.1,
    "توقف نماد": -1.8,
    "شفاف سازی": -1.4,
    "نوسان قیمت": -1.3,
    "عدم امکان برگزاری مجمع": -1.9,
}


def normalize_persian_text(text: str) -> str:
    """پاک‌سازی تگ‌های HTML و یکسان‌سازی حروف و نیم‌فاصله‌های فارسی."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # حذف تگ‌های HTML احتمالی
    cleaned = re.sub(r"<[^>]+>", " ", text)

    # یکسان‌سازی حروف عربی به فارسی استاندارد
    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "آ": "ا",
        "\u200c": " ",  # تبدیل نیم‌فاصله به فاصله برای تطابق بهتر عبارات
        "\u0640": "",   # حذف کشیدگی حروف
    }
    for src_char, target_char in replacements.items():
        cleaned = cleaned.replace(src_char, target_char)

    # حذف اعراب و نشانه‌های غیرضروری
    cleaned = re.sub(r"[\u064B-\u065F\u0670]", "", cleaned)
    cleaned = re.sub(r"[^\w\s\u0600-\u06FF٪%0-9]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def score_financial_sentiment(text: str) -> Dict[str, object]:
    """
    تحلیل احساسات و جهت‌گیری بنیادی متن گزارش کدال.
    خروجی شامل امتیاز نرمال‌شده بین -1 تا +1، برچسب و عبارات کلیدی یافت‌شده است.
    """
    norm_text = normalize_persian_text(text)
    if not norm_text:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "matched_drivers": [],
        }

    raw_score = 0.0
    matched: List[str] = []

    for phrase, weight in BULLISH_PHRASES.items():
        norm_phrase = normalize_persian_text(phrase)
        if norm_phrase in norm_text:
            raw_score += weight
            matched.append(f"+{phrase}")

    for phrase, weight in BEARISH_PHRASES.items():
        norm_phrase = normalize_persian_text(phrase)
        if norm_phrase in norm_text:
            raw_score += weight
            matched.append(f"-{phrase}")

    # نرمال‌سازی غیرخطی امتیاز در بازه [-1, 1] با تابع tanh
    normalized_score = float(np.tanh(raw_score / 3.0))

    if normalized_score >= 0.20:
        label = "Bullish"
    elif normalized_score <= -0.20:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "sentiment_score": round(normalized_score, 4),
        "sentiment_label": label,
        "matched_drivers": matched,
    }


class CodalRAGEngine:
    """
    موتور جستجوی معنایی و بازیابی اسناد مالی کدال (Retrieval Engine)
    برای پاسخ به پرسش‌های تحلیلی روی مجموعه اطلاعیه‌های بورس.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.docs_df: pd.DataFrame = pd.DataFrame()
        self.doc_matrix = None

    def fit(self, df: pd.DataFrame, text_columns: List[str] | None = None) -> "CodalRAGEngine":
        """ایندکس‌گذاری متن اطلاعیه‌های کدال در فضای برداری TF-IDF."""
        cols = text_columns or ["ticker", "title", "summary"]
        self.docs_df = df.copy().reset_index(drop=True)

        combined_texts = []
        for _, row in self.docs_df.iterrows():
            parts = [str(row.get(c, "")) for c in cols if pd.notna(row.get(c, ""))]
            combined_texts.append(normalize_persian_text(" ".join(parts)))

        self.docs_df["_normalized_corpus"] = combined_texts
        self.doc_matrix = self.vectorizer.fit_transform(combined_texts)
        return self

    def query(self, question: str, top_k: int = 5) -> pd.DataFrame:
        """بازیابی مرتبط‌ترین اطلاعیه‌های کدال برای پرسش کاربر."""
        if self.doc_matrix is None or self.docs_df.empty:
            raise RuntimeError("CodalRAGEngine must be fitted before querying.")

        norm_q = normalize_persian_text(question)
        if not norm_q:
            return pd.DataFrame()

        q_vec = self.vectorizer.transform([norm_q])
        sims = cosine_similarity(q_vec, self.doc_matrix).ravel()

        res = self.docs_df.drop(columns=["_normalized_corpus"]).copy()
        res["relevance_score"] = np.round(sims * 100.0, 2)
        res = res[res["relevance_score"] > 0].sort_values("relevance_score", ascending=False)
        return res.head(top_k).reset_index(drop=True)