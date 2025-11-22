#__________    _____  .______________
#\______   \  /  _  \ |   \__    ___/
# |    |  _/ /  /_\  \|   | |    |   
# |    |   \/    |    \   | |    |   
# |______  /\____|__  /___| |____|   
#        \/         \/ 

import streamlit as st
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import matplotlib.pyplot as plt
from langchain_community.vectorstores import FAISS


# configuration
genai.configure(api_key="ENTER_API_KEY_HERE")
llm = genai.GenerativeModel("gemini-2.0-flash")

DATA_ROOT = Path("courses")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


embedder = load_embedder()


# classify topics
def classify_topics(qbank, topics):
    topic_vecs = embedder.encode(topics, convert_to_numpy=True)
    topic_vecs /= np.linalg.norm(topic_vecs, axis=1, keepdims=True)

    mapping = {t: [] for t in topics}

    for q in qbank:
        q_vec = embedder.encode([q["question_text"]], convert_to_numpy=True)[0]
        q_vec /= np.linalg.norm(q_vec)

        sims = topic_vecs @ q_vec
        best_topic = topics[int(np.argmax(sims))]

        mapping[best_topic].append(q)

    return mapping


#load courses+faiss(faiss gye)
def load_course(course):
    folder = DATA_ROOT / course

    # topics.json
    with open(folder / "topics.json", "r", encoding="utf-8") as f:
        topics = json.load(f)["topics"]

    # question_bank.json
    with open(folder / "question_bank.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    topic_map = classify_topics(questions, topics)

    # Load FAISS (slides)
    def emb_fn(texts):
        return embedder.encode(texts, convert_to_numpy=True)

    retriever = FAISS.load_local(
        str(folder / "faiss_store"),
        embeddings=emb_fn,
        index_name="index",
        allow_dangerous_deserialization=True
    ).as_retriever(search_kwargs={"k": 5})

    # Precompute embeddings for question similarity
    pyq_emb = embedder.encode(
        [q["question_text"] for q in questions],
        convert_to_numpy=True
    )
    pyq_emb /= np.linalg.norm(pyq_emb, axis=1, keepdims=True)

    return topics, questions, topic_map, retriever, pyq_emb


# rag answering using slides
def rag_answer(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    resp = llm.generate_content(prompt)
    return resp.text.strip(), docs


# ---------------------------------------------------
# FIND RELATED PYQs
# ---------------------------------------------------
def get_relevant_pyqs(query, questions, pyq_embs, top_k=3):
    q_vec = embedder.encode([query], convert_to_numpy=True)[0]
    q_vec /= np.linalg.norm(q_vec)

    sims = pyq_embs @ q_vec
    idxs = sims.argsort()[-top_k:][::-1]

    return [(questions[i], float(sims[i])) for i in idxs]


# ---------------------------------------------------
# PIE CHART
# ---------------------------------------------------
def draw_pie(mapping):
    labels = list(mapping.keys())
    values = [len(mapping[t]) for t in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, autopct="%1.1f%%")
    st.pyplot(fig)


# STREAMLIT UI
#
st.title("🤖 BAIT — BITS AI Tutor 🤖")

courses = [d.name for d in DATA_ROOT.iterdir() if d.is_dir()]
course = st.selectbox("Select Course", courses)

topics, qbank, topic_map, retriever, pyq_embs = load_course(course)

left, right = st.columns(2)


# sorter
with left:
    st.header("📚 Topics & Questions")

    selected_topic = st.selectbox("Choose a topic", topics)

    st.subheader("Questions:")
    qs = topic_map[selected_topic]

    if qs:
        for q in qs:
            st.markdown(f"### Q{q['question_number']}")
            st.write(q["question_text"])
            st.write("---")
    else:
        st.write("No questions found for this topic.")


# Chatbot+pie
with right:
    st.header("BAIT🤖")

    user_q = st.text_area("Ask a question")

    if st.button("Ask"):
        if user_q.strip():
            answer, retrieved = rag_answer(user_q, retriever)

            st.write("### Answer")
            st.write(answer)

            st.write("### 📌 Related PYQs")
            rel = get_relevant_pyqs(user_q, qbank, pyq_embs)

            for q, score in rel:
                st.markdown(f"**Q{q['question_number']}** (match: {score:.2f})")
                st.write(q["question_text"])
                st.write("---")

    st.subheader("📊 Topic Distribution")
    draw_pie(topic_map)
