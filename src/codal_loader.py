import datetime
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from src.persian_fin_nlp import normalize_persian_text, score_financial_sentiment

CODAL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://codal.ir/",
}

TSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

TARGET_TICKERS: Dict[str, str] = {
    "فولاد": "46348559193224090",
    "فملی": "35425587644337450",
    "خودرو": "65883838195688438",
    "خساپا": "44891482026867833",
    "شستا": "2400322364771558",
    "شپنا": "7745894403636165",
    "وبملت": "778253364357513",
    "فارس": "14079693677610396",
}


def jalali_to_gregorian_int(jalali_str: str) -> Optional[Tuple[int, bool]]:
    """
    تبدیل تاریخ و ساعت انتشار شمسی کدال (مثلاً '1405/08/12 14:30:00')
    به عدد میلادی YYYYMMDD و تشخیص انتشار بعد از بسته شدن بازار (ساعت 12:30).
    """
    match = re.search(r"(\d{4})/(\d{2})/(\d{2})(?:\s+(\d{2}):(\d{2}))?", str(jalali_str))
    if not match:
        return None

    jy, jm, jd = int(match.group(1)), int(match.group(2)), int(match.group(3))
    hour = int(match.group(4)) if match.group(4) else 10
    minute = int(match.group(5)) if match.group(5) else 0
    after_market_close = (hour > 12) or (hour == 12 and minute >= 30)

    jy += 1595
    days = (
        -355668
        + (365 * jy)
        + ((jy // 33) * 8)
        + (((jy % 33) + 3) // 4)
        + jd
        + ((jm - 1) * 31 if jm < 7 else ((jm - 7) * 30) + 186)
    )
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    sal_a = [
        0,
        31,
        29 if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)) else 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ]
    gm = 0
    while gm < 13 and gd > sal_a[gm]:
        gd -= sal_a[gm]
        gm += 1

    dt = datetime.date(gy, gm, gd)
    if after_market_close:
        dt = dt + datetime.timedelta(days=1)

    return int(dt.strftime("%Y%m%d")), after_market_close


def _fetch_adjusted_prices(ins_code: str, session: requests.Session) -> pd.DataFrame:
    """دریافت سابقه قیمت تعدیل‌شده از TSETMC برای محاسبه بازده پس از انتشار خبر."""
    url = f"https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceDailyList/{ins_code}/0"
    resp = session.get(url, headers=TSE_HEADERS, timeout=20)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json().get("closingPriceDaily", []))
    if df.empty:
        return pd.DataFrame()

    df = df[df["qTotTran5J"] > 0].sort_values("dEven").drop_duplicates("dEven").reset_index(drop=True)
    p_close = df["pClosing"].astype(float).to_numpy()
    p_yest = df["priceYesterday"].astype(float).to_numpy()

    adj_factors = np.ones(len(df), dtype=float)
    for t in range(1, len(df)):
        if p_close[t - 1] > 0 and p_yest[t] > 0:
            adj_factors[t - 1] = p_yest[t] / p_close[t - 1]

    df["adj_close"] = p_close * np.cumprod(adj_factors[::-1])[::-1]
    return df[["dEven", "adj_close"]].rename(columns={"dEven": "trade_date"})


def _fetch_codal_letters_for_ticker(
    ticker: str, ins_code: str, session: requests.Session, pages: int = 3
) -> List[Dict]:
    """استخراج اطلاعیه‌های کدال (و در صورت نیاز پیام‌های ناظر TSETMC) برای یک نماد."""
    records: List[Dict] = []

    for page in range(1, pages + 1):
        url = (
            f"https://search.codal.ir/api/search/v2/q"
            f"?Symbol={ticker}&PageNumber={page}&Audited=true&NotAudited=true"
            f"&IsNotAudited=false&Childs=true&Mains=true&Publisher=false"
            f"&CompanyState=-1&Category=-1&CompanyType=-1"
            f"&Consolidatable=true&NotConsolidatable=true"
        )
        try:
            resp = session.get(url, headers=CODAL_HEADERS, timeout=15)
            if resp.ok:
                letters = resp.json().get("Letters", [])
                if not letters:
                    break
                for item in letters:
                    pub_dt = str(item.get("PublishDateTime", ""))
                    parsed_dt = jalali_to_gregorian_int(pub_dt)
                    if not parsed_dt:
                        continue
                    eff_date_int, after_close = parsed_dt
                    title = normalize_persian_text(str(item.get("Title", "")))
                    letter_code = str(item.get("LetterCode", ""))
                    rel_url = str(item.get("Url", ""))
                    full_url = f"https://codal.ir{rel_url}" if rel_url.startswith("/") else rel_url

                    records.append({
                        "ticker": ticker,
                        "publish_jalali": pub_dt,
                        "effective_date": eff_date_int,
                        "after_close": after_close,
                        "letter_code": letter_code,
                        "title": title,
                        "summary": f"{title} ({letter_code})",
                        "url": full_url,
                        "source": "Codal",
                    })
        except Exception:
            break

    # منبع پشتیبان: پیام‌های ناظر بازار در TSETMC در صورت عدم پاسخ‌دهی کدال
    if len(records) < 10:
        try:
            msg_url = f"https://cdn.tsetmc.com/api/Msg/GetMsgByInsCode/{ins_code}"
            m_resp = session.get(msg_url, headers=TSE_HEADERS, timeout=15)
            if m_resp.ok:
                for msg in m_resp.json().get("msg", [])[:60]:
                    d_even = int(msg.get("dEven", 0))
                    if d_even <= 20200101:
                        continue
                    tse_title = normalize_persian_text(str(msg.get("tseTitle", "")))
                    tse_desc = normalize_persian_text(str(msg.get("tseDesc", "")))
                    records.append({
                        "ticker": ticker,
                        "publish_jalali": str(d_even),
                        "effective_date": d_even,
                        "after_close": False,
                        "letter_code": "TSE-MSG",
                        "title": tse_title,
                        "summary": tse_desc[:240],
                        "url": "https://tsetmc.com",
                        "source": "TSETMC-Msg",
                    })
        except Exception:
            pass

    return records


def _align_news_with_returns(news_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """اتصال هر اطلاعیه به اولین روز معاملاتی مؤثر و محاسبه بازده ۱، ۳ و ۵ روز بعد."""
    if news_df.empty or price_df.empty:
        return pd.DataFrame()

    trade_dates = price_df["trade_date"].to_numpy()
    adj_closes = price_df["adj_close"].to_numpy()

    aligned_rows = []
    for _, row in news_df.iterrows():
        eff_dt = int(row["effective_date"])
        idx = int(np.searchsorted(trade_dates, eff_dt, side="left"))

        # نیاز به حداقل ۱ روز قبل از خبر (قیمت مبنا) و ۵ روز بعد از خبر برای سنجش بازده
        if idx < 1 or (idx + 4) >= len(adj_closes):
            continue

        pre_close = adj_closes[idx - 1]
        if pre_close <= 0:
            continue

        sent = score_financial_sentiment(f"{row['title']} {row['summary']}")
        aligned_rows.append({
            **row.to_dict(),
            "matched_trade_date": int(trade_dates[idx]),
            "sentiment_score": sent["sentiment_score"],
            "sentiment_label": sent["sentiment_label"],
            "matched_drivers": ", ".join(sent["matched_drivers"]) if sent["matched_drivers"] else "None",
            "ret_1d": (adj_closes[idx] / pre_close) - 1.0,
            "ret_3d": (adj_closes[idx + 2] / pre_close) - 1.0,
            "ret_5d": (adj_closes[idx + 4] / pre_close) - 1.0,
        })

    return pd.DataFrame(aligned_rows)


def build_codal_dataset(
    tickers: Optional[List[str]] = None,
    cache_path: str = "data/codal_announcements.csv",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """ساخت و ذخیره‌سازی دیتاست کامل اطلاعیه‌های کدال به همراه امتیاز سنتیمنت و بازده سهام."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and not force_refresh:
        return pd.read_csv(cache_path)

    target_list = tickers or list(TARGET_TICKERS.keys())
    all_frames = []

    with requests.Session() as session:
        for ticker in target_list:
            ins_code = TARGET_TICKERS[ticker]
            print(f"Fetching Codal disclosures & price history for: {ticker} ...")
            try:
                price_df = _fetch_adjusted_prices(ins_code, session)
                raw_letters = _fetch_codal_letters_for_ticker(ticker, ins_code, session, pages=3)
                news_df = pd.DataFrame(raw_letters).drop_duplicates(subset=["title", "effective_date"])
                merged = _align_news_with_returns(news_df, price_df)
                if not merged.empty:
                    all_frames.append(merged)
                    print(f"  [OK] {ticker}: {len(merged)} disclosures aligned with market returns")
            except Exception as exc:
                print(f"  [Warning] {ticker} skipped: {exc}")

    if not all_frames:
        raise RuntimeError("No Codal disclosures could be loaded. Check internet connection and turn off VPN.")

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df = final_df.sort_values("matched_trade_date", ascending=False).reset_index(drop=True)
    final_df.to_csv(cache_path, index=False)
    print(f"\nSaved {len(final_df)} total disclosures -> {cache_path}")
    return final_df


if __name__ == "__main__":
    df_codal = build_codal_dataset(force_refresh=True)
    print("\n=== Sentiment Distribution Across Codal Announcements ===")
    print(df_codal["sentiment_label"].value_counts().to_string())