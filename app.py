import os
import random
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Get Ollama API from environment variable (set in docker-compose.yml)
API_URL = os.getenv("OLLAMA_API", "http://ollama:11434")

# --- Session state setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "role" not in st.session_state:
    st.session_state.role = None

if "awaiting_order_confirmation" not in st.session_state:
    st.session_state.awaiting_order_confirmation = False

if "awaiting_order_city" not in st.session_state:
    st.session_state.awaiting_order_city = False

if "last_selected_product" not in st.session_state:
    st.session_state.last_selected_product = None

if "recommended_products" not in st.session_state:
    st.session_state.recommended_products = []

# --- Load data ---
@st.cache_data
def load_data():
    products_df = pd.read_csv("data/products.csv")
    accessories_df = pd.read_csv("data/accessories.csv")
    satisfaction_df = pd.read_csv("data/smartphone_reviews.csv")
    policies_df = pd.read_csv("data/company_policies.csv")
    return products_df, accessories_df, satisfaction_df, policies_df

products_df, accessories_df, satisfaction_df, policies_df = load_data()

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

# --- Precompute embeddings ---
@st.cache_data
def compute_embeddings(descriptions):
    return embedder.encode(descriptions, convert_to_tensor=False)

product_embeddings = compute_embeddings(products_df["description"].tolist())

# --- AI summarizer ---
def generate_ai_summary(results, reviews, query, policy_notes=None):
    """
    Returns:
        ai_output: str -> The AI-generated sales-agent style summary
        policy_notice: str -> Short message about policy restrictions (if any)
    """
    policy_notice = ""
    recommendation_text = ""
    ai_output = ""  # Ensure always defined
 
    if not results:
        if policy_notes:
            policy_notice = (
                f"😔 Unfortunately, due to company restrictions, there are no products that match your request. "
                f"Restrictions applied: {'; '.join(policy_notes)}"
            )
        else:
            policy_notice = "😔 Unfortunately, no products were found matching your search."
 
        recommendation_text = (
            "As your friendly sales agent, I suggest trying other devices allowed for your role, "
            "adjusting your search filters, or exploring alternative features you might be interested in."
        )
    else:
        if policy_notes:
            policy_notice = f"⚠️ Policy restrictions applied: {'; '.join(policy_notes)}"
 
        recommendation_text = f"Hello! Based on your search '{query}', here are some great options I recommend:\n\n"
        for idx, (device, accs) in enumerate(results, start=1):
            recommendation_text += f"{idx}. **{device['name']}** — ${device['price']} — ⭐ {device['avg_rating']}\n"
            recommendation_text += f"Description: {device['description']}\n"
            if accs:
                recommendation_text += "Accessories you might love: " + ", ".join(a['name'] for a in accs) + "\n"
            reviews_text = reviews[idx-1] if reviews else "No reviews available."
            recommendation_text += f"Customer Feedback: {reviews_text[:200]}...\n\n"
 
        # Add sales-agent order suggestion
        recommendation_text += "I’m here to help you pick the perfect device and accessories! Shall I order the selected phone for you?"
 
    # Build prompt for AI
    summary_prompt = ""
    if policy_notice:
        summary_prompt += f"POLICY NOTICE: {policy_notice}\n\n"
    summary_prompt += recommendation_text
 
    # --- AI call with OpenAI + Llama fallback ---
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly, persuasive sales agent for tech devices and accessories."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=400
        )
        ai_output = resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"OpenAI not available ({e}), falling back to local Llama/Mistral model...")
        try:
            import requests, json
            response = requests.post(
                f"{API_URL}/api/generate",
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
            ai_output = "".join(summary_chunks).strip()
        except Exception as e:
            ai_output = f"⚠️ AI summary skipped — no OpenAI or Llama available ({e})."
 
    return ai_output, policy_notice
    
# --- Helper: OpenAI / Llama for Order AI ---
def call_order_ai(prompt_text, system_role="You are a friendly Order AI assistant."):
    ai_output = ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_role},
                      {"role": "user", "content": prompt_text}],
            max_tokens=200
        )
        ai_output = resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"OpenAI not available ({e}), falling back to local Llama model...")
        try:
            import requests, json
            response = requests.post(
                f"{API_URL}/api/generate",
                json={"model": "mistral", "prompt": prompt_text},
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
            chunks = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            chunks.append(data["response"])
                    except json.JSONDecodeError:
                        continue
            ai_output = "".join(chunks).strip()
        except Exception as e:
            ai_output = f"⚠️ Order AI unavailable ({e})"
    return ai_output

# --- UI ---
st.title("📱 AI Mobile Device & Accessories Recommender (Chat)")

# --- Ask for role if not set ---
# --- Role selection ---
available_roles = policies_df["role"].dropna().unique().tolist()

# Case 1: No role selected yet → allow choosing from dropdown
if "role" not in st.session_state or st.session_state.role is None:
    default_role = available_roles[0] if available_roles else "Unknown"
    selected_role = st.selectbox("Select your role", available_roles, index=0)
    if selected_role:
        st.session_state.role = selected_role
        st.chat_message("assistant").write(
            f"✅ Role set to **{selected_role}**. You can now ask about devices."
        )

# Case 2: Role is already selected
else:
    # If chat has not started yet → allow changing role
    if not st.session_state.chat_history:
        default_index = available_roles.index(st.session_state.role) if st.session_state.role in available_roles else 0
        selected_role = st.selectbox("Select your role", available_roles, index=default_index)
        if selected_role != st.session_state.role:
            st.session_state.role = selected_role
            st.chat_message("assistant").write(
                f"🔄 Role updated to **{selected_role}**. You can now ask about devices."
            )
    else:
        # Chat already started → lock the role
        st.info(f"🔒 Role locked as **{st.session_state.role}** for this conversation. Reload the page to change it.")
        
role = st.session_state.role

# --- Extract brand ---
def extract_brand(name: str) -> str:
    tokens = name.split()
    first = tokens[0].lower()
    mapping = {
        "iphone": "Apple", "ipad": "Apple", "galaxy": "Samsung", "samsung": "Samsung",
        "pixel": "Google", "oneplus": "OnePlus", "xiaomi": "Xiaomi", "redmi": "Xiaomi",
        "poco": "Xiaomi", "oppo": "Oppo", "vivo": "Vivo", "lenovo": "Lenovo",
        "huawei": "Huawei", "amazon": "Amazon", "nokia": "Nokia", "motorola": "Motorola"
    }
    return mapping.get(first, tokens[0])
products_df["brand"] = products_df["name"].apply(extract_brand)

# --- Filter by policy ---
def filter_by_policy(products_df, policies_df, role):
    policy = policies_df[policies_df["role"] == role]
    if policy.empty:
        return products_df, ["⚠️ No policy found for this role. Showing all products."]
    policy_notes = []
    filtered = products_df.copy()
    # Allowed brands
    allowed_brands = []
    if "allowed_brands" in policy.columns:
        allowed_brands = [b.strip() for b in str(policy["allowed_brands"].iloc[0]).split(",") if b.strip()]
    if allowed_brands:
        before = len(filtered)
        filtered = filtered[filtered["brand"].isin(allowed_brands)]
        removed = before - len(filtered)
        if removed > 0:
            policy_notes.append(f"{removed} products removed due to brand restrictions (allowed: {', '.join(allowed_brands)})")
    # Price range
    if "allowed_price_range" in policy.columns and pd.notna(policy["allowed_price_range"].iloc[0]):
        try:
            min_price, max_price = map(int, policy["allowed_price_range"].iloc[0].split("-"))
            before = len(filtered)
            filtered = filtered[(filtered["price"] >= min_price) & (filtered["price"] <= max_price)]
            removed = before - len(filtered)
            if removed > 0:
                policy_notes.append(f"{removed} products removed due to price restriction (${min_price}–${max_price})")
        except Exception:
            policy_notes.append("⚠️ Price range format invalid in policy file")
    return filtered, policy_notes

# --- Semantic search ---
def semantic_search(query, df, top_k=3):
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    df_embeddings = embedder.encode(df["description"].tolist(), convert_to_tensor=False)
    query_emb = embedder.encode([query], convert_to_tensor=False)
    scores = cosine_similarity(query_emb, df_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return df.iloc[top_indices]

# --- Replay chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- Chat input ---
if prompt := st.chat_input("Ask about a device..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"💬 **You:** {prompt}")

    with st.chat_message("assistant"):
        user_input = prompt.lower().strip()

        # --- Order AI waiting for city input ---
        if st.session_state.get("awaiting_order_city", False):
            user_city = prompt.strip()
            product_name = st.session_state.last_selected_product['name']
            order_number = random.randint(100000, 999999)
            tracking_url = f"https://www.example.com/track/order/{order_number}"

            order_prompt = f"""
            The user has ordered {product_name} to be delivered in {user_city}.
            Generate a friendly, playful confirmation message that includes the product name,
            the delivery city, the tracking link {tracking_url}, and thanks the user.
            """
            ai_confirmation = call_order_ai(order_prompt)
            # --- Add balloon animation ---
            st.balloons()
            st.markdown(ai_confirmation)

            # Reset state after order confirmation
            st.session_state.awaiting_order_city = False
            st.session_state.last_selected_product = None
            st.session_state.recommended_products = []
            st.session_state.awaiting_order_confirmation = False
            st.stop()

        # --- Order AI waiting for order confirmation (handoff from Sales AI) ---
        elif user_input.startswith("order ") and st.session_state.awaiting_order_confirmation:
            matched_product = None
            for device in st.session_state.recommended_products:
                if device['name'].lower() in user_input:
                    matched_product = device
                    break

            if matched_product is not None:
                st.session_state.last_selected_product = matched_product
                st.session_state.awaiting_order_confirmation = False
                st.session_state.awaiting_order_city = True

                # Ask politely for city
                order_prompt = f"""
                The user wants to order {matched_product['name']}.
                Ask politely for the user's city for delivery in a friendly and cheerful tone.
                """
                ai_response = call_order_ai(order_prompt)
                st.markdown(ai_response)
            else:
                st.info("⚠️ Device name not recognized. Please type: 'Order <device name>' from the recommended list.")

        # --- All other input: treat as a new Sales AI query ---
        else:
            # Reset awaiting_order_confirmation to allow multiple searches
            st.session_state.awaiting_order_confirmation = False

            # --- Original Sales AI recommendation logic ---
            filtered_products, policy_reasons = filter_by_policy(products_df, policies_df, role)
            top_products = semantic_search(prompt, filtered_products)

            results = []
            review_texts = []
            if not top_products.empty:
                for _, device in top_products.iterrows():
                    accessories = accessories_df[accessories_df["compatible_skus"].str.contains(device["sku"], na=False)]
                    accessories_list = accessories.to_dict(orient="records")
                    model_reviews_df = satisfaction_df[satisfaction_df["model_name"] == device["name"]]
                    avg_rating = round(model_reviews_df["star_rating"].mean(), 1) if not model_reviews_df.empty else "N/A"
                    device["avg_rating"] = avg_rating
                    reviews = " ".join(model_reviews_df["review_text"].tolist()) if not model_reviews_df.empty else "No reviews available."
                    review_texts.append(reviews)
                    results.append((device, accessories_list))

            ai_summary, policy_notice_text = generate_ai_summary(results, review_texts, prompt, policy_notes=policy_reasons)

            if policy_notice_text:
                st.info(policy_notice_text)

            if results:
                st.markdown("### Recommended devices available in store for sale:")
                st.session_state.recommended_products = []
                for idx, (device, accs) in enumerate(results, start=1):
                    st.markdown(f"{idx}. 📱 **{device['name']}** — ${device['price']} — ⭐ {device['avg_rating']}")
                    st.write(device["description"])
                    if accs:
                        st.markdown("🛍 **Accessories:** " + ", ".join(f"{a['name']} (${a['price']})" for a in accs))
                    st.markdown("---")
                    st.session_state.recommended_products.append(device)
                st.session_state.awaiting_order_confirmation = True

            st.subheader("✨ Personalized Recommendation Summary")
            st.markdown(ai_summary)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_summary})