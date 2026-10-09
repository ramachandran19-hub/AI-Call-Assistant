
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def send_call_email(json_path):
    """Email the call summary and attach the full call report."""

    sender = os.getenv("SMTP_EMAIL")
    app_password = os.getenv("SMTP_APP_PASSWORD")
    recipient = os.getenv("SUMMARY_RECIPIENT_EMAIL")

    if not sender or not app_password or not recipient:
        raise ValueError(
            "Email configuration missing. Check SMTP_EMAIL, "
            "SMTP_APP_PASSWORD, and SUMMARY_RECIPIENT_EMAIL in .env."
        )

    json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(f"Call JSON file not found: {json_file}")

    with json_file.open("r", encoding="utf-8") as file:
        call_data = json.load(file)

    summary = call_data.get("summary", {})
    call_id = call_data.get("call_id", "Unknown")
    txt_file = json_file.with_suffix(".txt")

    message = EmailMessage()
    message["Subject"] = f"AI Call Assistant - Call Summary ({call_id})"
    message["From"] = sender
    message["To"] = recipient

    message.set_content(
        "AI Call Assistant - Call Report\n\n"
        f"Call ID: {call_id}\n"
        f"Call End Reason: {call_data.get('end_reason', 'Not recorded')}\n\n"
        "STRUCTURED SUMMARY\n"
        "------------------\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n\nThe complete call report and transcript are attached."
    )

    # Attach JSON containing the summary and transcript data.
    message.add_attachment(
        json_file.read_bytes(),
        maintype="application",
        subtype="json",
        filename=json_file.name,
    )

    # Attach the readable TXT report with the full transcript.
    if txt_file.exists():
        message.add_attachment(
            txt_file.read_bytes(),
            maintype="text",
            subtype="plain",
            filename=txt_file.name,
        )

    # Gmail SMTP over SSL.
    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        timeout=30,
    ) as server:
        server.login(sender, app_password)
        server.send_message(message)

    print(f"\nEmail sent successfully to {recipient}.")
