import streamlit as st
import pandas as pd
import time
from main import handle_call  # Import call handling function

# Store call logs in session state
if "call_logs" not in st.session_state:
    st.session_state.call_logs = []

st.title("📞 Call Scam Detector")

# Button to simulate incoming calls
if st.button("📲 Simulate Incoming Call"):
    call_data = handle_call()  # Get call details
    st.session_state.call_logs.append(call_data)

# Display Call Logs
st.subheader("📋 Call Logs")
df = pd.DataFrame(st.session_state.call_logs, columns=["Serial No.", "Phone Number", "Status", "Time", "AI Summary"])
st.table(df)