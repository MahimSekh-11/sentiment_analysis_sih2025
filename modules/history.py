import streamlit as st
import pandas as pd
import os
from datetime import datetime

DB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "database", "virtual_db.json")
)

def show():
    st.title("📜 History & Analysis Trends")

    # ---------------- Load database ----------------
    if not os.path.exists(DB_FILE):
        st.warning("⚠️ Database file not found!")
        return

    df = pd.read_json(DB_FILE)

    # ---------------- Show basic stats ----------------
    st.subheader("Database Info")
    last_updated_time = datetime.fromtimestamp(os.path.getmtime(DB_FILE))
    st.markdown(f"**Last Updated:** {last_updated_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"**Total Entries:** {len(df)} rows")

    # ---------------- Filters ----------------
    with st.expander("Filters"):
        # Date filter (if database has 'Date' column)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            start_date = st.date_input("Start Date", df["Date"].min())
            end_date = st.date_input("End Date", df["Date"].max())
            df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
        else:
            df_filtered = df.copy()

        # Source filter (if database has 'Source' column)
        if "Source" in df.columns:
            source_filter = st.selectbox("Source", ["All"] + df["Source"].dropna().unique().tolist())
            if source_filter != "All":
                df_filtered = df_filtered[df_filtered["Source"] == source_filter]

    # ---------------- Display Table ----------------
    st.subheader("History Table")
    st.dataframe(df_filtered)

    # ---------------- Optional: Summary Stats ----------------
    st.subheader("Summary Stats")
    st.markdown(f"**Filtered Entries:** {len(df_filtered)} rows")
    if "Sentiment" in df_filtered.columns:
        st.line_chart(df_filtered[["Sentiment"]])
