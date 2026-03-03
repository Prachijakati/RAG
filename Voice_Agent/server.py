from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import tempfile
import os
import time
from faster_whisper import WhisperModel

app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n===== Streaming Session Started =====")

    audio_buffer = b""

    try:
        while True:
            message = await websocket.receive()

            # 🔹 If client disconnected
            if message["type"] == "websocket.disconnect":
                print("Client disconnected cleanly.")
                break

            # 🔹 If audio chunk
            if "bytes" in message and message["bytes"] is not None:
                audio_buffer += message["bytes"]

            # 🔹 If END signal
            elif "text" in message and message["text"] == "END":
                print("[INFO] Recording ended. Running STT...")

                start_stt = time.time()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(audio_buffer)
                    tmp_path = tmp.name

                segments, info = model.transcribe(tmp_path, task="translate")
                transcript = "".join([seg.text for seg in segments]).strip()

                end_stt = time.time()
                print(f"[STT] Time: {end_stt - start_stt:.3f} sec")

                await websocket.send_text(transcript)

                os.remove(tmp_path)
                audio_buffer = b""

    except WebSocketDisconnect:
        print("WebSocket disconnected gracefully.")

    except Exception as e:
        print("Unexpected error:", e)

    finally:
        print("===== Session Closed =====")