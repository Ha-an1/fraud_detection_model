import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/contacts.readonly"]

def authenticate_google():
    """Handles authentication and returns credentials."""
    creds = None

    # Load saved credentials
    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except json.JSONDecodeError:
            os.remove("token.json")

    # Authenticate if no valid credentials
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)

        # Save new credentials
        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())

    return creds

def get_saved_contacts():
    """Fetches user contacts from Google People API."""
    creds = authenticate_google()
    service = build("people", "v1", credentials=creds)

    try:
        result = service.people().connections().list(
            resourceName="people/me",
            pageSize=100,
            personFields="names,phoneNumbers"
        ).execute()
    except Exception as e:
        print(f"❌ Error fetching contacts: {e}")
        return {}

    contacts = result.get("connections", [])
    saved_contacts = {}

    for contact in contacts:
        names = contact.get("names", [])
        phone_numbers = contact.get("phoneNumbers", [])

        if names and phone_numbers:
            name = names[0].get("displayName", "Unknown")
            phone = phone_numbers[0].get("value", "Unknown")
            saved_contacts[phone] = name

    # print("📋 Fetched Contacts:", saved_contacts)

    return saved_contacts