import streamlit as st
import requests

st.set_page_config(page_title="Youm7 News RAG", page_icon="📰")
st.title("أخبار اليوم من موقع اليوم السابع")

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/query"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("إسألني عن أي خبر من أخبار اليوم"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("البحث عن الأخبار والتفكير..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={"query": prompt, "top_k": 5}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)
                    
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources):
                            st.info(f"Source {i+1}: {source}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI: {e}")
