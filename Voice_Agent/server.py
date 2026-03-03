# server.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import tempfile
import os
import time

from transcribe_ws import transcribe_file  # 🔥 Clean import

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n===== Streaming Session Started =====")

    audio_buffer = b""

    try:
        while True:
            message = await websocket.receive()

            # 🔹 Client disconnected
            if message["type"] == "websocket.disconnect":
                print("Client disconnected cleanly.")
                break

            # 🔹 Receiving audio chunks
            if message.get("bytes") is not None:
                audio_buffer += message["bytes"]

            # 🔹 Recording finished
            elif message.get("text") == "END":
                print("[INFO] Recording ended. Running STT...")

                total_start = time.time()

                # Save buffer to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(audio_buffer)
                    tmp_path = tmp.name

                # 🔥 Call transcription module
                transcript = transcribe_file(tmp_path)

                total_end = time.time()
                print(f"[TOTAL] End-to-End STT Time: {total_end - total_start:.3f} sec")

                # Send transcript back to frontend
                await websocket.send_text(transcript)

                # Cleanup
                os.remove(tmp_path)
                audio_buffer = b""

    except WebSocketDisconnect:
        print("WebSocket disconnected gracefully.")

    except Exception as e:
        print("Unexpected error:", e)

    finally:
        print("===== Session Closed =====")