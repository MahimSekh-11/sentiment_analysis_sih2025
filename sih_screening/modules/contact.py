import streamlit as st

def show():
    st.title("📞 Contact Us")

    st.markdown("""
    Have questions, feedback, or collaboration ideas?  
    We'd love to hear from you! Please fill out the form below or connect with us directly.
    """)

    # ---------------- Contact Form ----------------
    with st.form("contact_form", clear_on_submit=True):
        st.text_input("👤 Name", placeholder="Enter your full name")
        st.text_input("📧 Email", placeholder="Enter your email address")
        st.text_area("💬 Message", placeholder="Type your message here...")
        submitted = st.form_submit_button("📨 Send Message")

        if submitted:
            st.success("✅ Thank you! Your message has been submitted successfully.")

    st.divider()

    # ---------------- Direct Contact ----------------
    st.markdown("### 📬 Direct Contact")
    st.markdown("""
    - **Email:** [support@sentimentai.com](mailto:support@sentimentai.com)  
    - **LinkedIn:** [SentimentAI Team](https://linkedin.com)  
    - **GitHub:** [View Project Repository](https://github.com/RahulRagini02/sentiment_analysis)  
    - **Twitter/X:** [Follow us](https://twitter.com)  
    """)

    st.info("We typically respond within **24-48 hours**.")
