import streamlit as st
import pandas as pd
import requests
import os

DB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "database", "virtual_db.json")
)

ANALYZE_API = "http://127.0.0.1:8500/analyze_comment"

def show():
    st.title("💬 Stakeholder Comment Analysis")

    # Load database JSON
    df = pd.read_json(DB_FILE)

    # ---------------- Clause Selection ----------------
    clauses = ["Overall"]
    if "Clause_id" in df.columns:
        clauses += sorted(df["Clause_id"].dropna().unique().tolist())
    selected_clause = st.selectbox("Choose Clause ID", clauses)

    # ---------------- Stakeholder Selection ----------------
    stakeholders = ["Overall"]
    if "Stakeholders" in df.columns:
        stakeholders += sorted(df["Stakeholders"].dropna().unique().tolist())
    selected_stakeholder = st.selectbox("Choose Stakeholder", stakeholders)

    # ---------------- Label Selection ----------------
    labels = ["Overall"]
    if "Label" in df.columns:
        labels += sorted(df["Label"].dropna().unique().tolist())
    selected_label = st.selectbox("Choose Label", labels)

    # ---------------- Analyze Button ----------------
    if st.button("Analyze Comments"):
        with st.spinner("🔍 Analyzing stakeholder comments..."):
            # Filter dataframe based on selections
            df_filtered = df.copy()
            if selected_clause != "Overall":
                df_filtered = df_filtered[df_filtered["Clause_id"] == selected_clause]
            if selected_stakeholder != "Overall":
                df_filtered = df_filtered[df_filtered["Stakeholders"] == selected_stakeholder]
            if selected_label != "Overall":
                df_filtered = df_filtered[df_filtered["Label"] == selected_label]

            if df_filtered.empty:
                st.warning("⚠️ No data found for the selected Clause/Stakeholder/Label.")
                st.stop()

            # Merge all comments into one text
            long_text = ". ".join(df_filtered["Comment"].astype(str))
            if not long_text.strip():
                st.warning("⚠️ No comments available!")
                st.stop()

            # ---------------- Call Backend API ----------------
            payload = {
                "comment": long_text,
                "clause": None if selected_clause=="Overall" else selected_clause,
                "stakeholder": None if selected_stakeholder=="Overall" else selected_stakeholder
            }

            try:
                response = requests.post(ANALYZE_API, json=payload)
                if response.status_code == 200:
                    result = response.json()

                    # ---------------- Display Comments ----------------
                    st.subheader("📄 Combined Comments")
                    st.text_area(
                        label="All comments merged into one text",
                        value=long_text,
                        height=200
                    )

                    # ---------------- Display Analysis ----------------
                    st.subheader("🔎 Analysis Result")
                    st.markdown(f"**Sentiment:** {result.get('sentiment')}")
                    st.markdown(f"**Confidence:** {result.get('confidence')}")
                    st.markdown(f"**Reason:** {result.get('reason')}")

                    # ---------------- Display Suggestion ----------------
                    st.subheader("💡 Suggested Rewrite")
                    st.text_area(
                        label="Improved stakeholder-friendly version",
                        value=result.get("suggestion", ""),
                        height=200
                    )
                else:
                    st.error("❌ Analysis API error. Status code: {}".format(response.status_code))
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
