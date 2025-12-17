import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------- Resume Text Extraction ----------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# ---------- ATS-like Resume Scoring ----------
KEY_SKILLS = [
    "python", "machine learning", "data science", "ai", "nlp",
    "sql", "projects", "experience", "internship",
    "scikit-learn", "streamlit", "tensorflow", "pandas", "numpy"
]

def keyword_score(resume_text):
    score = 0
    for skill in KEY_SKILLS:
        if skill in resume_text:
            score += 1
    return score

# ---------- Resume Ranking ----------
def rank_resumes(resume_texts):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(resume_texts)

    avg_vector = np.mean(vectors.toarray(), axis=0).reshape(1, -1)
    similarity_scores = cosine_similarity(vectors, avg_vector).flatten()

    final_scores = []
    for i, text in enumerate(resume_texts):
        ks = keyword_score(text)
        final = (similarity_scores[i] * 70) + (ks * 2)
        final_scores.append(round(final, 2))

    return final_scores

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.title("📄 AI Resume Scanner – Best Resume Selector")
st.write("Upload multiple resumes (PDF). The system will rank and select the best resume.")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🔍 Scan & Rank Resumes"):
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("Please upload at least 2 resumes.")
    else:
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            resume_texts.append(extract_text_from_pdf(file))
            resume_names.append(file.name)

        scores = rank_resumes(resume_texts)

        results = list(zip(resume_names, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("🏆 Resume Ranking Results")

        for i, (name, score) in enumerate(results, start=1):
            if i == 1:
                st.success(f"🥇 BEST RESUME: {name} — Score: {score}")
            else:
                st.write(f"{i}. {name} — Score: {score}")
