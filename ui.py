import streamlit as st
import requests
import base64
import os
import re

base_dir = os.path.dirname(__file__)

st.set_page_config(
    page_title="E-JUST Sports AI",
    page_icon=os.path.join(base_dir, "ui", "textures", "ej.png"),
    layout="centered" 
)

API_URL = "http://127.0.0.1:8000/query"

def get_base64_image(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_path = os.path.join(base_dir, "ui", "textures", "pitch_field.jpg")
ball_path = os.path.join(base_dir, "ui", "textures", "ejust_ball.png")

user_avatar = os.path.join(base_dir, "ui", "textures", "ej.png") 
bot_avatar = os.path.join(base_dir, "ui", "textures", "ej.png")

bg_image = get_base64_image(bg_path)
ball_image = get_base64_image(ball_path)

st.markdown(f"""
<style>
::-webkit-scrollbar {{ width: 15px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(239, 68, 68, 0.4); border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(239, 68, 68, 0.7); }}

[data-testid="stToolbar"], 
[data-testid="stHeader"], 
[data-testid="stDecoration"], 
header, footer {{
    display: none !important;
}}

.stApp {{
    background: url("data:image/png;base64,{bg_image}") center/cover no-repeat fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.85) 0%, rgba(255, 255, 255, 0.2) 100%);
    z-index: 0;
    pointer-events: none;
}}

.block-container {{
    position: relative;
    z-index: 2;
    padding-top: 2rem !important;
    padding-bottom: 120px !important; 
    max-width: 850px !important;
}}

.header-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 2px solid #ef4444 !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease;
}}

.header-container:hover {{
    transform: translateY(-5px);
}}

@keyframes textGlow {{
    0%, 100% {{ text-shadow: 0 0 15px rgba(0, 0, 0, 0.1); }}
    50% {{ text-shadow: 0 0 25px rgba(0, 0, 0, 0.5); }}
}}

.title {{
    font-size: 34px;
    font-weight: 800;
    color: #000000; 
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0px;
    margin-top: 15px;
    text-align: center;
    animation: textGlow 3s infinite alternate;
}}

.ball {{
    display: block;
    width: 110px; 
    animation: bounce 1.5s infinite cubic-bezier(0.28, 0.84, 0.42, 1);
    filter: drop-shadow(0 15px 10px rgba(0,0,0,0.15));
    transition: filter 0.3s;
}}

.ball:hover {{
    filter: drop-shadow(0 25px 15px rgba(239,68,68,0.3));
}}

@keyframes bounce {{
    0%, 100% {{ transform: translateY(0) scale(1); }}
    50% {{ transform: translateY(-20px) scale(1.03); }} 
}}

@keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* =========================================
   RTL CONFIGURATION (ARABIC FIXES)
========================================= */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {{
    direction: rtl !important;
    text-align: right !important;
}}

[data-testid="stChatMessage"] {{
    direction: rtl !important;
}}

[data-testid="stChatInput"] textarea {{
    direction: rtl !important;
    text-align: right !important;
}}

[data-testid="stExpander"] details summary p {{
    direction: rtl !important;
    text-align: right !important;
}}

/* =========================================
   CHAT BUBBLE STYLES
========================================= */
[data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-radius: 20px !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
    box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.04) !important;
    animation: slideUp 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

[data-testid="stChatMessage"]:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08) !important;
}}

[data-testid="stChatMessage"]:has(img[alt="user avatar"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background: linear-gradient(135deg, #ffffff, #ffffff) !important;
    border: 2px solid #ef4444 !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2) !important;
}}

/* =========================================
   FORCE TEXT COLORS TO BLACK (KEEP LINKS COLORED)
========================================= */
[data-testid="stChatMessage"] {{
    color: #000000 !important;
}}

[data-testid="stChatMessage"] .stMarkdown,
[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] .stMarkdown span,
[data-testid="stChatMessage"] .stMarkdown li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *:not(a) {{
    color: #000000 !important;
}}

/* Explicitly style the links to stand out */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {{
    color: #ef4444 !important;
    text-decoration: none !important;
    font-weight: bold !important;
}}

/* Add an underline when hovering over the link */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a:hover {{
    text-decoration: underline !important;
}}

/* =========================================
   BULLET POINTS AS COLORED RECTANGLES
========================================= */
[data-testid="stChatMessage"] .stMarkdown ul {{
    list-style-type: none !important; 
    padding-right: 0 !important; 
    margin-right: 0 !important;
}}

[data-testid="stChatMessage"] .stMarkdown li {{
    background-color: rgba(239, 68, 68, 0.08) !important; 
    border-right: 4px solid #ef4444 !important; 
    border-radius: 8px !important;
    padding: 12px 15px !important;
    margin-bottom: 10px !important;
    display: block !important;
    color: #000000 !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
}}

/* =========================================
   INPUT & LAYOUT FIXES
========================================= */
[data-testid="stBottom"], 
[data-testid="stBottomBlock"], 
.stAppBottomBlock {{
    background: transparent !important;
    background-color: transparent !important;
}}

[data-testid="stBottom"] > div,
[data-testid="stBottomBlock"] > div {{
    background: transparent !important;
    background-color: transparent !important;
}}

[data-testid="stChatInput"] {{
    background-color: transparent !important;
    padding: 5px 15px !important;
}}

[data-testid="stChatInput"] > div {{
    background-color: #ffffff !important; 
    border: 2px solid #ef4444 !important; 
    border-radius: 30px !important;
    box-shadow: 0 -5px 25px rgba(0, 0, 0, 0.08) !important;
    padding: 5px 15px !important; 
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}}

[data-testid="stChatInput"] > div:focus-within {{
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.25) !important;
    border-color: #dc2626 !important;
}}

textarea[aria-label="اسأل عن أي مباراة..."] {{
    background-color: #ffffff !important; 
    color: #0f172a !important; 
    -webkit-text-fill-color: #0f172a !important; 
    caret-color: #ef4444 !important; 
    padding: 12px 0px 12px 10px !important; 
}}

textarea[aria-label="اسأل عن أي مباراة..."]::placeholder {{
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}}

/* COMPLETELY HIDE THE SEND BUTTON */
[data-testid="stChatInput"] button {{
    display: none !important;
}}

/* =========================================
   CUSTOM EXPANDER STYLES (NO DARK BACKGROUND)
========================================= */
[data-testid="stExpander"] {{
    background-color: transparent !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 12px !important;
}}

/* Fix: Prevent background from turning dark when idle, open, or hovered */
[data-testid="stExpander"] details, 
[data-testid="stExpander"] details[open],
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details summary:hover {{
    background-color: transparent !important;
    color: #ef4444 !important;
}}

[data-testid="stExpander"] details summary p {{
    color: #ef4444 !important;
    font-weight: bold !important;
    direction: rtl !important;
    text-align: right !important;
}}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-container">
    {f'<img class="ball" src="data:image/png;base64,{ball_image}">' if ball_image else ''}
    <div class="title">مساعدك الذكي للرياضة!</div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Persist messages AND their specific sources on re-render
for message in st.session_state.messages:
    # Set the correct avatar on re-render
    avatar_img = user_avatar if message["role"] == "user" else bot_avatar
    
    with st.chat_message(message["role"], avatar=avatar_img):
        st.markdown(message["content"])
        if message.get("urls"):
            with st.expander("📚 المصادر المذكورة"):
                for url in message["urls"]:
                    st.markdown(f"- [{url}]({url})")

prompt = st.chat_input("اسأل عن أي مباراة...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render user message with the new avatar
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)

    # Render assistant message with the new avatar
    with st.chat_message("assistant", avatar=bot_avatar):
        with st.spinner("جارٍ التحليل... ⚽"):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": prompt, "top_k": 15}
                )

                if response.status_code == 200:
                    data = response.json()
                    raw_answer = data["answer"]
                    
                    # 1. Regex to extract only the URLs present in the LLM's response
                    extracted_urls = re.findall(r'(https?://[^\s]+)', raw_answer)
                    
                    # Clean trailing punctuation from URLs and remove duplicates
                    cleaned_urls = list(dict.fromkeys([u.rstrip('.,")') for u in extracted_urls]))

                    # 2. Slice off the LLM's "المصادر:" block to keep the chat bubble clean
                    clean_answer = re.split(r'\*?\*?المصادر:\*?\*?', raw_answer)[0].strip()

                    st.markdown(clean_answer)

                    # 3. Only display the expander if the LLM successfully cited sources
                    if cleaned_urls:
                        with st.expander("📚 المصادر المذكورة"):
                            for url in cleaned_urls:
                                # Render as clickable Markdown links
                                st.markdown(f"- [{url}]({url})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_answer,
                        "urls": cleaned_urls # Store URLs to persist across renders
                    })

                else:
                    st.error("⚠️ حدث خطأ في الاتصال بالخادم.")

            except Exception as e:
                st.error(f"⚠️ خطأ في الاتصال: {e}")