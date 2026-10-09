
import os
import smtplib
import ssl

from dotenv import load_dotenv

load_dotenv(override=True)

email = os.getenv("SMTP_EMAIL")
password = os.getenv("SMTP_APP_PASSWORD")

if not email or not password:
    print("Missing email configuration in .env")
    raise SystemExit(1)

try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()
        server.login(email, password)

        print("SUCCESS: Gmail authentication worked.")

except Exception as error:
    print("FAILED:", type(error).__name__)
    print("DETAIL:", str(error))
