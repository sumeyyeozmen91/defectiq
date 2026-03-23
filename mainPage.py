import streamlit as st
import pandas as pd
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DEFECTIQ", layout="wide")
st.title("DEFECTIQ")
st.caption("Jira-based Defect Analysis, Prioritization & Similarity Detection")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DEFAULT_JIRA_DOMAIN = "https://jira.turkcell.com.tr"
DEFAULT_PLATFORM_FIELD = "customfield_24721"

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------
jira_domain = st.text_input("Jira Domain", value=DEFAULT_JIRA_DOMAIN)
jsessionid = st.text_input("JSESSIONID", type="password")
jql = st.text_area(
    "JQL",
    height=140,
    placeholder='project in (BIPC, TF) AND issuetype in (Bug, Defect) AND status not in (Cancelled, Done)'
)
platform_field = st.text_input("Platform Field", value=DEFAULT_PLATFORM_FIELD)

# İstersen debug kapatılabilir
debug_mode = st.checkbox("Debug Mode", value=True)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def normalize(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_platform(val):
    if val is None:
        return ""

    if isinstance(val, str):
        return val

    if isinstance(val, dict):
        return val.get("value") or val.get("name") or str(val)

    if isinstance(val, list):
        out = []
        for item in val:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(item.get("value") or item.get("name") or str(item))
            else:
                out.append(str(item))
        return ", ".join([x for x in out if x])

    return str(val)


def extract_description(desc):
    if desc is None:
        return ""

    if isinstance(desc, str):
        return desc

    if isinstance(desc, dict):
        return str(desc)

    if isinstance(desc, list):
        return " ".join([str(x) for x in desc])

    return str(desc)


# ---------------------------------------------------
# JIRA FETCH
# ---------------------------------------------------
def fetch_all_issues(jira_domain, jsessionid, jql, platform_field):
    all_rows = []
    start_at = 0
    page_size = 100

    headers = {
        "Cookie": f"JSESSIONID={jsessionid}",
        "Accept": "application/json"
    }

    while True:
        url = f"{jira_domain.rstrip('/')}/rest/api/2/search"
        params = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": page_size,
            "fields": f"summary,description,priority,{platform_field}"
        }

        response = requests.get(url, headers=headers, params=params, timeout=90)

        if debug_mode:
            st.write("STATUS:", response.status_code)
            st.code(response.text[:500])

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
        except Exception:
            raise Exception(
                "Response JSON parse edilemedi. Büyük ihtimalle Jira JSON yerine HTML/login sayfası döndü. "
                f"Status: {response.status_code} | First 500 chars: {response.text[:500]}"
            )

        issues = data.get("issues", [])
        total = data.get("total", 0)

        if not issues:
            break

        for issue in issues:
            fields = issue.get("fields", {}) or {}
            priority_obj = fields.get("priority", {}) or {}

            all_rows.append({
                "Issue Key": issue.get("key", ""),
                "Summary": fields.get("summary", ""),
                "Description": extract_description(fields.get("description")),
                "Custom field (Platform)": extract_platform(fields.get(platform_field)),
                "Priority": priority_obj.get("name", "") if isinstance(priority_obj, dict) else str(priority_obj)
            })

        start_at += len(issues)

        if start_at >= total:
            break

    return pd.DataFrame(all_rows)


# ---------------------------------------------------
# SEMANTIC PRIORITY ENGINE
# ---------------------------------------------------
def semantic_priority(summary, description):
    text = normalize(f"{summary} {description}")

    # GATING - blocking / stuck / loop
    if any(k in text for k in [
        "cannot select",
        "does not allow",
        "unable to select",
        "cannot proceed",
        "unable to continue",
        "blocks user",
        "blocks user interaction",
        "prevents user",
        "prevents selecting",
        "stuck",
        "loop",
        "keeps opening",
        "opens repeatedly",
        "keeps appearing",
        "popup keeps appearing",
        "page keeps opening"
    ]):
        return "Gating", "BLOCKING_FLOW"

    # GATING - core function failure
    if any(k in text for k in [
        "cannot send",
        "unable to send",
        "message not sent",
        "cannot receive",
        "unable to receive",
        "login failed",
        "cannot login",
        "unable to login",
        "cannot sign in",
        "call not connecting",
        "cannot start call",
        "unable to start call",
        "crash",
        "application crashes",
        "app crashes"
    ]):
        return "Gating", "CORE_FUNCTION_FAILURE"

    # HIGH - interaction failure
    if any(k in text for k in [
        "not clickable",
        "cannot click",
        "tap does not work",
        "not tappable",
        "link is not clickable",
        "button does not work",
        "interaction does not work"
    ]):
        return "High", "INTERACTION_FAILURE"

    # HIGH - state inconsistency
    if any(k in text for k in [
        "appears even though",
        "even though",
        "although",
        "should not be visible",
        "not registered but",
        "shows as",
        "indicates they are not",
        "incorrect state",
        "inconsistent state",
        "confusion",
        "misleading",
        "can communicate but",
        "already registered but"
    ]):
        return "High", "STATE_INCONSISTENCY"

    # MEDIUM - visual artifact
    if any(k in text for k in [
        "black line",
        "visual artifact",
        "render issue",
        "thick black line",
        "line appears",
        "artifact persists"
    ]):
        return "Medium", "VISUAL_ARTIFACT"

    # MEDIUM - performance only
    if any(k in text for k in [
        "lag",
        "lags",
        "slow",
        "delay",
        "scrolling lags"
    ]):
        return "Medium", "PERFORMANCE_ONLY"

    # LOW - pure UI/cosmetic
    if any(k in text for k in [
        "alignment",
        "misalignment",
        "font",
        "color",
        "spacing",
        "icon",
        "cosmetic",
        "typo",
        "misspelling"
    ]):
        return "Low", "UI_ONLY"

    return "Medium", "DEFAULT"


# ---------------------------------------------------
# DUPLICATE ENGINE
# ---------------------------------------------------
def build_text_for_similarity(summary, description):
    return normalize(f"{summary} {description}")


def run_duplicate_analysis(df):
    if df.empty:
        df["Has_Duplicate"] = False
        df["Max_Similarity"] = 0.0
        df["Duplicate_With"] = ""
        df["Duplicate_Type"] = ""
        df["Action"] = ""
        return df

    df = df.copy()
    df["TEXT"] = df.apply(
        lambda r: build_text_for_similarity(r.get("Summary", ""), r.get("Description", "")),
        axis=1
    )

    if df["TEXT"].fillna("").str.strip().eq("").all():
        df["Has_Duplicate"] = False
        df["Max_Similarity"] = 0.0
        df["Duplicate_With"] = ""
        df["Duplicate_Type"] = ""
        df["Action"] = ""
        return df

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(df["TEXT"])
    sim = cosine_similarity(tfidf)

    df["Has_Duplicate"] = False
    df["Max_Similarity"] = 0.0
    df["Duplicate_With"] = ""
    df["Duplicate_Type"] = ""
    df["Action"] = ""

    for i in range(len(df)):
        best_sim = 0.0
        best_j = None

        for j in range(len(df)):
            if i == j:
                continue

            current_sim = float(sim[i, j])
            if current_sim > best_sim and current_sim > 0.75:
                best_sim = current_sim
                best_j = j

        if best_j is not None:
            p1 = normalize(df.iloc[i].get("Custom field (Platform)", ""))
            p2 = normalize(df.iloc[best_j].get("Custom field (Platform)", ""))
            same_platform = (p1 == p2)

            if best_sim > 0.95:
                dtype = "EXACT_DUPLICATE"
            elif best_sim > 0.90:
                dtype = "HIGH_SEMANTIC_SIMILARITY"
            else:
                dtype = "POSSIBLE_SEMANTIC_DUPLICATE"

            if not same_platform:
                action = "KEEP_BOTH_DIFFERENT_PLATFORM"
            else:
                action = "SAFE_DELETE" if best_sim > 0.90 else "QA_REVIEW"

            df.at[i, "Has_Duplicate"] = True
            df.at[i, "Max_Similarity"] = round(best_sim, 3)
            df.at[i, "Duplicate_With"] = df.iloc[best_j].get("Issue Key", "")
            df.at[i, "Duplicate_Type"] = dtype
            df.at[i, "Action"] = action

    return df


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def run_pipeline():
    raw_df = fetch_all_issues(
        jira_domain=jira_domain,
        jsessionid=jsessionid,
        jql=jql,
        platform_field=platform_field
    )

    if raw_df.empty:
        return raw_df

    prio_results = raw_df.apply(
        lambda r: semantic_priority(r.get("Summary", ""), r.get("Description", "")),
        axis=1
    )
    raw_df["STP_Priority"] = [x[0] for x in prio_results]
    raw_df["Reason"] = [x[1] for x in prio_results]

    out_df = run_duplicate_analysis(raw_df)

    final_cols = [
        "Issue Key",
        "Summary",
        "Description",
        "Custom field (Platform)",
        "Priority",
        "STP_Priority",
        "Reason",
        "Has_Duplicate",
        "Max_Similarity",
        "Duplicate_With",
        "Duplicate_Type",
        "Action"
    ]

    return out_df[final_cols]


# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if st.button("Fetch + Analyze"):
    if not jira_domain or not jsessionid or not jql:
        st.error("Jira Domain, JSESSIONID ve JQL zorunlu.")
    else:
        try:
            with st.spinner("Jira'dan kayıtlar çekiliyor ve analiz ediliyor..."):
                result_df = run_pipeline()

            st.success(f"{len(result_df)} kayıt işlendi.")
            st.dataframe(result_df, use_container_width=True, height=600)

            csv_data = result_df.to_csv(index=False, sep=";")
            st.download_button(
                "Download Final CSV",
                data=csv_data,
                file_name="DEFECTIQ_JIRA_OUTPUT.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Hata: {e}")
