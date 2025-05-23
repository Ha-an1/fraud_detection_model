import random
import time
import datetime
import streamlit as st
from contacts import get_saved_contacts  # Ensure this function does not auto-execute
from convoWithMeta import chatbot  # Ensure this function does not auto-execute

def detect_incoming_call():
    """Simulates detecting an incoming call."""
    print("📞 Incoming call detected...")
    time.sleep(2)
    
    incoming_number = "+91" + str(random.randint(1000000000, 9999999999))
    print(f"📲 Call from: {incoming_number}")
    
    return incoming_number

test_case_positive = "+919384276021"

def handle_call():
    """Handles call detection and contact verification."""
    # incoming_number = detect_incoming_call()
    incoming_number = test_case_positive  # Using test case for now

    print("🔍 Checking saved contacts...")
    
    # Ensure this function is only called when needed
    saved_contacts = get_saved_contacts() if "saved_contacts" not in st.session_state else st.session_state.saved_contacts
    st.session_state.saved_contacts = saved_contacts

    if incoming_number in saved_contacts:
        status = f"Saved Contact: {saved_contacts[incoming_number]}"
        ai_summary = "Not required"
        print(f"✅ {incoming_number} is saved as '{saved_contacts[incoming_number]}'. No AI agent forwarding needed.")
    else:
        status = "Unknown Contact"
        print(f"⚠ {incoming_number} is unknown! Forwarding to AI agent...")
        if st.button("Forward to AI Agent"):
            ai_summary = chatbot()  # This will only execute when the button is clicked
        else:
            ai_summary = "Pending AI Analysis"

    call_entry = {
        "Serial No.": len(st.session_state.call_logs) + 1,
        "Phone Number": incoming_number,
        "Status": status,
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "AI Summary": ai_summary
    }

    return call_entry

def run_call():
    call_entry = handle_call()
    st.session_state.call_logs.append(call_entry)
    st.write(call_entry)

# Streamlit UI
st.title("AI Call Assistant")

if "call_logs" not in st.session_state:
    st.session_state.call_logs = []

if "saved_contacts" not in st.session_state:
    st.session_state.saved_contacts = {}  # Avoid loading it unnecessarily

if st.button("Detect Incoming Call"):
    run_call()
