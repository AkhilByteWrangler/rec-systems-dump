import streamlit as st
import requests
import io
import textwrap
import qrcode
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from openai import OpenAI
import torch
import numpy as np

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Smart Meal Recommender 🍽️", layout="centered")
st.title("🍽️ Smart Meal Recommender")

for key in ["meal_plan", "recs", "rec_index"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "meal_plan" else 0 if key == "rec_index" else []

st.sidebar.title("🧠 AI Enhancements (Optional)")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
use_openai = bool(api_key)

if use_openai:
    client = OpenAI(api_key=api_key)
else:
    st.sidebar.warning("Add an OpenAI key to unlock summaries, tips, and nutrition insights.")

st.sidebar.title("👤 Select User")

try:
    user_list = requests.get(f"{API_URL}/users?limit=50").json()
    user_id = st.sidebar.selectbox("User ID", user_list)
except Exception as e:
    st.sidebar.error(f"Could not fetch users: {e}")
    user_id = None

with st.sidebar.expander("👶 New User? Customize your taste"):
    cuisine = st.multiselect("🍽️ Preferred cuisines", ["Italian", "Indian", "Mexican", "Chinese", "American"])
    diet = st.selectbox("🥦 Diet type", ["Any", "Vegetarian", "Vegan", "Keto", "Paleo", "Gluten-Free"])
    cook_time = st.slider("⏱️ Max cooking time (minutes)", 10, 120, 45)
    favorites = st.text_input("❤️ Favorite ingredients (comma separated)")
    dislikes = st.text_input("🚫 Ingredients to avoid (comma separated)")

    if st.button("🎯 Get Smart Recommendations"):
        payload = {
            "cuisines": cuisine,
            "diet": diet,
            "cook_time": cook_time,
            "favorites": [x.strip() for x in favorites.split(",") if x.strip()],
            "dislikes": [x.strip() for x in dislikes.split(",") if x.strip()]
        }
        try:
            recs = requests.post(f"{API_URL}/coldstart", json=payload).json()
            st.session_state.recs = recs if isinstance(recs, list) else []
            st.session_state.rec_index = 0
        except Exception as e:
            st.error(f"Error fetching recommendations: {e}")

if user_id and st.sidebar.button("🎲 Load Recommendations"):
    try:
        response = requests.get(f"{API_URL}/recommend", params={"user_id": user_id, "top_k": 20})
        st.session_state.recs = response.json()
        st.session_state.rec_index = 0
    except Exception as e:
        st.error(f"Failed to load recommendations: {e}")

def generate_ai_summary_and_nutrition(title, ingredients):
    prompt = f"""
Give a detailed and enthusiastic summary for a recipe titled '{title}', followed by:
1. Three tips to improve or customize it
2. Estimated calorie count per serving
3. A macro estimate (carbs, proteins, fats)
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Could not fetch AI summary: {str(e)}]"

def wrap_text(text, width=95):
    return textwrap.wrap(text, width=width)

def generate_qr_code(url):
    qr_img = qrcode.make(url)
    buf = io.BytesIO()
    qr_img.save(buf)
    buf.seek(0)
    return buf

def replace_emojis(text):
    replacements = {
        "🍽": "Smart Meal Plan",
        "⏱️": "Time:",
        "🥬": "Top Ingredients:",
        "📻": "Watch on YouTube"
    }
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    return text

def generate_pdf(meal_plan, use_openai=False, generate_ai_summary_and_nutrition=None):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    p.setFont("Helvetica-Bold", 18)
    p.drawString(margin, y, "Smart Meal Plan")
    y -= 30

    for i, recipe in enumerate(meal_plan, 1):
        if y < 200:
            p.showPage()
            y = height - margin

        p.setFont("Helvetica-Bold", 14)
        p.setFillColor(colors.darkblue)
        p.drawString(margin, y, f"{i}. {recipe['title'].capitalize()}")
        y -= 20
        p.setFillColor(colors.black)

        if use_openai and generate_ai_summary_and_nutrition:
            summary = generate_ai_summary_and_nutrition(recipe['title'], recipe['ingredients'])
            summary = replace_emojis(summary)
            p.setFont("Helvetica", 10)
            for line in wrap_text(summary):
                p.drawString(margin, y, line)
                y -= 14
            y -= 5

        p.setFont("Helvetica-Bold", 10)
        p.drawString(margin, y, f"Time: {recipe['minutes']} min")
        y -= 14

        p.drawString(margin, y, "Top Ingredients:")
        y -= 14

        p.setFont("Helvetica", 10)
        for ing in recipe['ingredients'][:5]:
            wrapped = wrap_text(ing)
            for line in wrapped:
                p.drawString(margin + 10, y, f"- {line}")
                y -= 12
            y -= 2

        youtube_url = f"https://www.youtube.com/results?search_query={'+'.join(recipe['title'].split())}+recipe"
        qr_buf = generate_qr_code(youtube_url)
        qr_img = ImageReader(qr_buf)
        p.drawImage(qr_img, width - 120, y - 20, width=60, height=60)
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(width - 120, y - 30, "Watch on YouTube")
        y -= 80

        p.setStrokeColor(colors.grey)
        p.line(margin, y, width - margin, y)
        y -= 20

    p.save()
    buffer.seek(0)
    return buffer

if st.session_state.recs:
    while st.session_state.rec_index < len(st.session_state.recs):
        rec = st.session_state.recs[st.session_state.rec_index]
        try:
            recipe = requests.get(f"{API_URL}/recipe/{rec['recipe_id']}").json()
        except:
            st.session_state.rec_index += 1
            continue

        st.subheader(recipe["title"])
        st.markdown(f"⏱️ **Time**: {recipe['minutes']} min")
        st.markdown(f"🥗 **Ingredients**: {len(recipe['ingredients'])}")
        st.markdown(f"🏷️ **Tags**: {', '.join(recipe['tags'][:5])}")

        st.write("### 📋 Ingredients")
        for ing in recipe["ingredients"][:10]:
            st.markdown(f"- {ing}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("❤️ Save to Meal Plan", key=f"save_{rec['recipe_id']}"):
                st.session_state.meal_plan.append(recipe)
                st.session_state.rec_index += 1
        with col2:
            if st.button("❌ Skip", key=f"skip_{rec['recipe_id']}"):
                st.session_state.rec_index += 1

        st.progress((st.session_state.rec_index + 1) / len(st.session_state.recs))
        break
else:
    st.info("👈 Choose a user or cold-start to load some recommendations!")

if st.session_state.meal_plan:
    st.subheader("📄 Export Your Meal Plan")
    if st.button("📟 Generate PDF"):
        pdf_buffer = generate_pdf(
            st.session_state.meal_plan,
            use_openai,
            generate_ai_summary_and_nutrition if use_openai else None
        )
        st.download_button(
            label="📅 Download PDF",
            data=pdf_buffer,
            file_name="smart_meal_plan.pdf",
            mime="application/pdf"
        )