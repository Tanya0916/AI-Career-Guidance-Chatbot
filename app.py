import streamlit as st
from chatbot import get_response

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Career Guidance Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

h1{
    text-align:center;
    color:#1E88E5;
}

.chat-container{
    border-radius:15px;
    padding:10px;
}

.user-message{
    background:#4CAF50;
    color:white;
    padding:12px;
    border-radius:12px;
    margin:8px 0;
    text-align:right;
}

.bot-message{
    background:#F1F1F1;
    color:#222222;
    padding:12px;
    border-radius:12px;
    margin:8px 0;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🤖 AI Career Bot")

st.sidebar.info(
"""
This chatbot provides guidance on:

 AI Careers

 Machine Learning

 Data Science

 Resume Tips

 Interview Preparation

 Certifications

 Internships

 Web Development

Built using:

• Python

• NLTK

• Scikit-learn

• Streamlit
"""
)

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Title
# -----------------------------
st.title("🤖 AI Career Guidance Chatbot")

st.write(
"Ask me anything about careers, resume building, interviews, skills, or certifications!"
)

# -----------------------------
# Display Previous Messages
# -----------------------------
for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
            👤 {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-message">
            🤖 {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    # Store User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_input
        }
    )

    # Get Bot Response
    response = get_response(user_input)

    # Store Bot Response
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )

    st.rerun()

# -----------------------------
# Clear Chat
# -----------------------------
if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# -----------------------------
# Footer
# -----------------------------
st.markdown(
"""
<div class="footer">
Made with  using Python, NLTK and Streamlit
</div>
""",
unsafe_allow_html=True
)