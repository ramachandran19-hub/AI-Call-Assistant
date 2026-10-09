
import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv(override=True)

client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN"),
)

try:
    call = client.calls.create(
        to=os.getenv("TEST_PHONE_NUMBER"),
        from_=os.getenv("TWILIO_NUMBER"),
        url="https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech",
    )

    print("Call request accepted.")
    print("Call SID:", call.sid)
    print("Status:", call.status)

except TwilioRestException as error:
    print("Twilio error code:", error.code)
    print("Twilio error message:", error.msg)

