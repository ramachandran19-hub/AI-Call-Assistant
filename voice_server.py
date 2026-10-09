
from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI(title="AI Call Assistant Voice Server")


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "AI Call Assistant Voice Server",
    }


@app.post("/twiml")
def twiml():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="en-US">
        Hello! You have reached the AI Call Assistant.
        Our voice server is working.
    </Say>
</Response>"""

    return Response(
        content=xml,
        media_type="application/xml",
    )
