from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from whisper_online import FasterWhisperASR, OnlineASRProcessor
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, status
import numpy as np
from whisper_online_server import Connection, ServerProcessor

# Define constants
SAMPLING_RATE = 16000  # in Hz
FRAME_DURATION = 0.1  # in seconds
SAMPLE_SIZE = 4  # in bytes for float32
MIN_CHUNK_SIZE = 4096

required_frame_size = int(SAMPLING_RATE * FRAME_DURATION * SAMPLE_SIZE)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    # Redirect to the static HTML file
    return RedirectResponse(url="/static/index.html")

src_lan = "yue"

# Initialize the ASR system
asr = FasterWhisperASR(lan=src_lan, modelsize="large-v2")
online = OnlineASRProcessor(asr)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection = WebSocketConnection(websocket)
    processor = ServerProcessor(connection, online)
    await processor.process()
    await websocket.close()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connection accepted")
#     online.init()
#     print("ASR Processor initiated!")
#     buffer = bytearray() 
#     try:
#         while True:
#             data = await websocket.receive_bytes()  # Receive audio chunk as bytes
#             buffer.extend(data)  # Add received data to the buffer
#            # Process only if buffer has enough data to match a full frame
#             while len(buffer) >= MIN_CHUNK_SIZE:
#                 # Process the buffer in chunks of MIN_CHUNK_SIZE
#                 chunk_data = buffer[:MIN_CHUNK_SIZE]
#                 buffer = buffer[MIN_CHUNK_SIZE:]

#                 # Convert bytes to audio format
#                 audio = np.frombuffer(chunk_data, dtype=np.float32)
#                 if audio.size == 0:
#                     continue    
#                 online.insert_audio_chunk(audio)
#                 result = online.process_iter()
#                 if result and result[2]:  # Ensure there is text to send back
#                     await websocket.send_text(f"{result[0]} {result[1]} {result[2]}")
#     except Exception as e:
#         print(f"Error: {e}")
#     except WebSocketDisconnect:
#         print("WebSocket disconnected.")
#         buffer.clear() 
#     finally:
#         final_output = online.finish()
#         if final_output and final_output[2]:
#             try:
#                 await websocket.send_text(f"{final_output[0]} {final_output[1]} {final_output[2]}")
#             except Exception as e:
#                 print("Error sending final output:", e)
#         await websocket.close()
#         buffer.clear() 
#         print("WebSocket connection closed")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
