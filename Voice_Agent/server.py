# server.py

from fastapi import FastAPI, WebSocket
import os
import uuid

from transcribe import transcribe_file

app = FastAPI()

UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            # Receive audio bytes
            audio_bytes = await websocket.receive_bytes()

            # Create unique file name
            file_name = f"{uuid.uuid4()}.wav"
            file_path = os.path.join(UPLOAD_DIR, file_name)

            # Save audio file
            with open(file_path, "wb") as f:
                f.write(audio_bytes)

            print(f"Saved file: {file_path}")

            # Run STT
            transcript = transcribe_file(file_path)

            print("Transcript:", transcript)

            # Send transcript back
            await websocket.send_text(transcript)

        except Exception as e:
            print("Error:", e)
            break