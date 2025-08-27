import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- Load data ---
@st.cache_data
def load_data():
    products_df = pd.read_csv("data/products.csv")
    accessories_df = pd.read_csv("data/accessories.csv")
    satisfaction_df = pd.read_csv("data/smartphone_reviews.csv")
    return products_df, accessories_df, satisfaction_df

products_df, accessories_df, satisfaction_df = load_data()

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

# --- Precompute embeddings for products ---
@st.cache_data
def compute_embeddings(descriptions):
    return embedder.encode(descriptions, convert_to_tensor=False)

product_embeddings = compute_embeddings(products_df["description"].tolist())

# --- Semantic search ---
def semantic_search(query, embeddings, top_k=3):
    query_emb = embedder.encode([query], convert_to_tensor=False)
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]

# --- AI summarizer (OpenAI or Ollama) ---
def generate_ai_summary(results, reviews, query):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        summary_prompt = f"""The user searched for: '{query}'.
Here are the top product matches, their accessories, and real user reviews with average ratings:

"""
        for idx, (device, accs) in enumerate(results, start=1):
            summary_prompt += f"{idx}. {device['name']} (${device['price']})\n"
            summary_prompt += f"Description: {device['description']}\n"
            summary_prompt += f"Average Rating: {device['avg_rating']} ⭐\n"
            if accs:
                summary_prompt += "Accessories: " + ", ".join(a['name'] for a in accs) + "\n"
            summary_prompt += f"Reviews: {reviews[idx-1]}\n\n"

        summary_prompt += "Write a user-friendly comparison and recommend the best device."

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI product advisor."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"OpenAI not available ({e}), trying Ollama...")

    # --- Ollama fallback ---
    try:
        import requests, json
        summary_prompt = f"User search: '{query}'.\n\n"
        for idx, (device, accs) in enumerate(results, start=1):
            summary_prompt += f"{idx}. {device['name']} (${device['price']})\n"
            summary_prompt += f"Description: {device['description']}\n"
            summary_prompt += f"Average Rating: {device['avg_rating']} ⭐\n"
            if accs:
                summary_prompt += "Accessories: " + ", ".join(a['name'] for a in accs) + "\n"
            summary_prompt += f"Reviews: {reviews[idx-1]}\n\n"
        summary_prompt += "Write a friendly comparative summary."

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": summary_prompt},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        summary_chunks = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        summary_chunks.append(data["response"])
                except json.JSONDecodeError:
                    continue
        return "".join(summary_chunks).strip()
    except Exception as e:
        return f"⚠️ AI summary skipped — no OpenAI or Ollama available ({e})."

# --- Streamlit UI ---
st.title("📱 AI Mobile Device & Accessories Recommender")

query = st.text_input("Search for a device (e.g., 'gaming phone under $700')")

if query:
    top_indices, top_scores = semantic_search(query, product_embeddings, top_k=3)

    results = []
    review_texts = []

    st.subheader("🔎 Recommendations by Company AI BOT")

    for idx in top_indices:
        device = products_df.iloc[idx].to_dict()
        accessories = accessories_df[accessories_df["compatible_skus"].str.contains(device["sku"], na=False)]
        accessories_list = accessories.to_dict(orient="records")

        # --- Calculate average rating ---
        model_reviews_df = satisfaction_df[satisfaction_df["model_name"] == device["name"]]
        if not model_reviews_df.empty:
            avg_rating = round(model_reviews_df["star_rating"].mean(), 1)
        else:
            avg_rating = "N/A"
        device["avg_rating"] = avg_rating

        # Collect reviews for AI
        model_reviews = model_reviews_df["review_text"].tolist()
        review_texts.append(" ".join(model_reviews) if model_reviews else "No reviews available.")

        results.append((device, accessories_list))

        # --- Display in UI ---
        st.markdown(f"### {device['name']} (${device['price']}) — ⭐ {avg_rating}")
        st.write(device["description"])

        if not accessories.empty:
            st.write("**Matching Accessories:**")
            for _, acc in accessories.iterrows():
                st.markdown(f"- {acc['name']} (${acc['price']})")
        else:
            st.write("_No accessories found._")

        if model_reviews:
            st.write("**Sample User Reviews:**")
            for r in model_reviews[:3]:
                st.markdown(f"- {r}")

    # --- AI Summary ---
    st.subheader("🤖 AI Summary")
    summary = generate_ai_summary(results, review_texts, query)
    st.write(summary)