import streamlit as st
import PyPDF2
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# UPGRADE 1: BERT-based similarity (with fallback to TF-IDF)
# ─────────────────────────────────────────────
@st.cache_resource
def load_bert_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except ImportError:
        return None

def bert_similarity(resume_text, jd_text, model):
    if model is None:
        # Fallback to TF-IDF if sentence-transformers not installed
        return tfidf_similarity_fallback(resume_text, jd_text)
    from sentence_transformers import util
    r_emb = model.encode(resume_text, convert_to_tensor=True)
    j_emb = model.encode(jd_text, convert_to_tensor=True)
    score = util.cos_sim(r_emb, j_emb)
    return round(float(score) * 100, 2)

def tfidf_similarity_fallback(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]
        # Normalize: TF-IDF cosine on short docs is naturally low, scale up
        normalized = min(100, score * 300)
        return round(normalized, 2)
    except:
        return 0.0


# ─────────────────────────────────────────────
# UPGRADE 2: SYNONYM + ABBREVIATION EXPANSION
# ─────────────────────────────────────────────
SYNONYMS = {
    r'\bml\b': 'machine learning',
    r'\bai\b': 'artificial intelligence',
    r'\bnlp\b': 'natural language processing',
    r'\bdl\b': 'deep learning',
    r'\bcv\b': 'computer vision',
    r'\bjs\b': 'javascript',
    r'\bts\b': 'typescript',
    r'\bdb\b': 'database',
    r'\boop\b': 'object oriented programming',
    r'\bapi\b': 'application programming interface',
    r'\brest\b': 'restful api',
    r'\bci/cd\b': 'continuous integration continuous deployment',
    r'\baws\b': 'amazon web services',
    r'\bgcp\b': 'google cloud platform',
    r'\bkpi\b': 'key performance indicator',
    r'\brnn\b': 'recurrent neural network',
    r'\bcnn\b': 'convolutional neural network',
    r'\bllm\b': 'large language model',
    r'\bbert\b': 'bidirectional encoder representations transformers',
    r'\bds\b': 'data science',
}

def expand_synonyms(text):
    text = text.lower()
    for pattern, replacement in SYNONYMS.items():
        text = re.sub(pattern, replacement, text)
    return text


# ─────────────────────────────────────────────
# STOPWORDS (includes JD filler words)
# ─────────────────────────────────────────────
STOPWORDS = {
    "and", "or", "the", "a", "an", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "be", "will", "we", "you", "your",
    "our", "as", "this", "that", "it", "by", "from", "have", "has",
    "been", "who", "which", "their", "they", "not", "but", "can",
    "all", "also", "any", "its", "must", "should", "would", "about",
    "more", "than", "other", "into", "such", "both", "through",
    "during", "including", "each", "very", "within", "well",
    # JD filler words
    "looking", "seeking", "required", "responsibilities", "role",
    "candidate", "candidates", "please", "apply", "position", "team",
    "work", "working", "job", "hire", "hiring", "join", "need",
    "needs", "ability", "strong", "good", "great", "excellent",
    "preferred", "plus", "bonus", "etc", "including", "like",
    "using", "use", "used", "help", "make", "ensure", "maintain",
}

def extract_jd_keywords(jd_text):
    tokens = re.findall(r'\b[a-z][a-z0-9+#.\-]{1,}\b', jd_text.lower())
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    seen, unique = set(), []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


# ─────────────────────────────────────────────
# KEYWORD MATCH SCORE
# ─────────────────────────────────────────────
def keyword_score(resume_text, jd_keywords):
    if not jd_keywords:
        return 0.0, [], []
    matched, missing = [], []
    for kw in jd_keywords:
        if kw in resume_text:
            matched.append(kw)
        else:
            missing.append(kw)
    score = (len(matched) / len(jd_keywords)) * 100
    return round(score, 2), matched, missing


# ─────────────────────────────────────────────
# UPGRADE 3A: EXPERIENCE MATCHING
# ─────────────────────────────────────────────
def experience_score(resume_text, jd_text):
    jd_years = re.findall(r'(\d+)\+?\s*years?', jd_text)
    required = int(jd_years[0]) if jd_years else 0

    resume_years = re.findall(r'(\d+)\+?\s*years?', resume_text)
    candidate = int(resume_years[0]) if resume_years else 0

    # Check for internship / fresher signals
    fresher_signals = ["fresher", "intern", "internship", "entry level", "0 years", "no experience"]
    is_fresher_role = any(s in jd_text for s in fresher_signals)

    if required == 0 or is_fresher_role:
        return 100.0  # No experience required = full score

    if candidate == 0:
        return 40.0  # Has education but no stated experience

    return round(min(100, (candidate / required) * 100), 2)


# ─────────────────────────────────────────────
# UPGRADE 3B: EDUCATION MATCHING
# ─────────────────────────────────────────────
DEGREE_LEVELS = {
    "phd": 4, "doctorate": 4, "ph.d": 4,
    "master": 3, "mba": 3, "mtech": 3, "m.tech": 3, "msc": 3, "m.sc": 3, "ms": 3,
    "bachelor": 2, "btech": 2, "b.tech": 2, "bsc": 2, "b.sc": 2, "be": 2, "b.e": 2,
    "diploma": 1, "12th": 1, "hsc": 1,
}

def education_score(resume_text, jd_text):
    jd_level = max(
        (v for k, v in DEGREE_LEVELS.items() if k in jd_text), default=0
    )
    resume_level = max(
        (v for k, v in DEGREE_LEVELS.items() if k in resume_text), default=0
    )
    if jd_level == 0:
        return 100.0
    if resume_level >= jd_level:
        return 100.0
    return round((resume_level / jd_level) * 100, 2)


# ─────────────────────────────────────────────
# UPGRADE 4: RESUME FORMAT / QUALITY SCORE
# ─────────────────────────────────────────────
def format_score(resume_text):
    checks = {
        "Email present":           bool(re.search(r'[\w.\-]+@[\w.\-]+\.\w+', resume_text)),
        "Phone present":           bool(re.search(r'\b[\d\s\-+()]{10,}\b', resume_text)),
        "LinkedIn present":        "linkedin" in resume_text,
        "GitHub present":          "github" in resume_text,
        "Projects present":        "project" in resume_text,
        "Skills section":          "skill" in resume_text,
        "Education section":       any(k in resume_text for k in ["education", "b.tech", "btech", "degree"]),
        "Experience/Internship":   any(k in resume_text for k in ["experience", "internship", "intern"]),
        "Measurable achievements": bool(re.search(r'\d+%|\d+x|\d+ years?|improved|increased|reduced|achieved|built|deployed', resume_text)),
        "Good word count":         400 <= len(resume_text.split()) <= 1000,
    }
    score = sum(10 for v in checks.values() if v)
    return score, checks  # max 100


# ─────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text.lower()


# ─────────────────────────────────────────────
# UPGRADE 5: WEIGHTED FINAL SCORING ENGINE
# ─────────────────────────────────────────────
def compute_ats_score(resume_text, jd_text, bert_model):
    # Expand synonyms on both texts
    r = expand_synonyms(resume_text)
    j = expand_synonyms(jd_text)

    jd_keywords = extract_jd_keywords(j)

    # Individual scores
    sem_score   = bert_similarity(r, j, bert_model)
    kw_score, matched, missing = keyword_score(r, jd_keywords)
    exp_score   = experience_score(r, j)
    edu_score   = education_score(r, j)
    fmt_score, fmt_checks = format_score(r)

    # Weighted combination
    # BERT Similarity  : 35%
    # Keyword Match    : 25%
    # Experience Match : 15%
    # Education Match  : 15%
    # Format Quality   : 10%
    final = round(
        (sem_score  * 0.35) +
        (kw_score   * 0.25) +
        (exp_score  * 0.15) +
        (edu_score  * 0.15) +
        (fmt_score  * 0.10),
        2
    )

    return {
        "final_score":       final,
        "semantic_score":    sem_score,
        "keyword_score":     kw_score,
        "experience_score":  exp_score,
        "education_score":   edu_score,
        "format_score":      fmt_score,
        "matched_keywords":  matched,
        "missing_keywords":  missing[:15],
        "format_checks":     fmt_checks,
        "total_jd_keywords": len(jd_keywords),
    }


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="ATS Resume Ranker v2", layout="wide", page_icon="📄")

st.title("📄 ATS Resume Scanner v2 — Accurate AI-Powered Ranker")
st.markdown(
    "Uses **BERT semantic similarity**, **synonym expansion**, **experience & education matching**, "
    "and **resume quality scoring** for highly accurate ATS results."
)

# Show BERT status
with st.spinner("Loading AI model..."):
    bert_model = load_bert_model()

if bert_model:
    st.success("✅ BERT model loaded — using AI-powered semantic similarity")
else:
    st.warning("⚠️ `sentence-transformers` not installed — using TF-IDF fallback. Run: `pip install sentence-transformers torch`")

st.divider()

# ── Step 1: Job Description ──
st.subheader("📋 Step 1: Paste the Job Description")
jd_text = st.text_area(
    "Job Description",
    height=220,
    placeholder="Paste the full job description here. Include required skills, experience, and qualifications."
)

# ── Step 2: Upload Resumes ──
st.subheader("📁 Step 2: Upload Resumes (PDF)")
uploaded_files = st.file_uploader(
    "Upload one or more resumes",
    type=["pdf"],
    accept_multiple_files=True
)

st.divider()

# ── Scan Button ──
if st.button("🔍 Scan & Rank Resumes", use_container_width=True, type="primary"):

    if not jd_text.strip():
        st.error("⚠️ Please paste a job description before scanning.")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least 1 resume.")
    else:
        resume_texts, resume_names = [], []

        with st.spinner("Extracting text from PDFs..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text.strip():
                    resume_texts.append(text)
                    resume_names.append(file.name)
                else:
                    st.warning(f"⚠️ Could not extract text from `{file.name}`. Skipping.")

        if not resume_texts:
            st.error("No readable resumes found.")
        else:
            results = []
            progress = st.progress(0, text="Scoring resumes...")

            for i, (text, name) in enumerate(zip(resume_texts, resume_names)):
                progress.progress((i + 1) / len(resume_texts), text=f"Scoring {name}...")
                scores = compute_ats_score(text, jd_text, bert_model)
                scores["name"] = name
                results.append(scores)

            progress.empty()
            results.sort(key=lambda x: x["final_score"], reverse=True)

            # ── Results ──
            st.subheader("🏆 Resume Ranking Results")
            st.caption(
                "**Score Weights:** 35% BERT Semantic Similarity | 25% Keyword Match | "
                "15% Experience | 15% Education | 10% Resume Quality"
            )

            for i, r in enumerate(results, start=1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                with st.expander(
                    f"{medal}  {r['name']}  —  ATS Score: **{r['final_score']} / 100**",
                    expanded=(i == 1)
                ):
                    # Score cards
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("🎯 Final ATS Score",     f"{r['final_score']}")
                    c2.metric("🧠 Semantic Match",      f"{r['semantic_score']:.1f}")
                    c3.metric("🔑 Keyword Match",       f"{r['keyword_score']:.1f}")
                    c4.metric("💼 Experience",          f"{r['experience_score']:.1f}")
                    c5.metric("🎓 Education",           f"{r['education_score']:.1f}")

                    st.progress(int(min(r["final_score"], 100)))

                    st.markdown(f"**📋 Resume Format Score: {r['format_score']} / 100**")
                    fmt_cols = st.columns(5)
                    for j, (check, passed) in enumerate(r["format_checks"].items()):
                        fmt_cols[j % 5].markdown(
                            f"{'✅' if passed else '❌'} {check}"
                        )

                    st.divider()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**✅ Matched Keywords**")
                        if r["matched_keywords"]:
                            st.markdown(" ".join([f"`{k}`" for k in r["matched_keywords"][:25]]))
                        else:
                            st.write("None matched")
                    with col_b:
                        st.markdown("**❌ Missing Keywords (Top 15)**")
                        if r["missing_keywords"]:
                            st.markdown(" ".join([f"`{k}`" for k in r["missing_keywords"]]))
                        else:
                            st.write("🎉 All major keywords matched!")

            # ── Score Legend ──
            st.divider()
            st.markdown("### 📊 Score Interpretation")
            l1, l2, l3, l4 = st.columns(4)
            l1.success("**80–100** → Excellent Match")
            l2.info("**60–79** → Good Match")
            l3.warning("**40–59** → Moderate Match")
            l4.error("**0–39** → Poor Match")

            # ── Side-by-side comparison table ──
            if len(results) > 1:
                st.divider()
                st.markdown("### 📊 Side-by-Side Comparison")
                import pandas as pd
                df = pd.DataFrame([{
                    "Resume":        r["name"],
                    "ATS Score":     r["final_score"],
                    "Semantic":      r["semantic_score"],
                    "Keywords":      r["keyword_score"],
                    "Experience":    r["experience_score"],
                    "Education":     r["education_score"],
                    "Format":        r["format_score"],
                } for r in results])
                st.dataframe(df, use_container_width=True, hide_index=True)