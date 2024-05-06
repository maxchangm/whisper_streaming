from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from whisper_online import FasterWhisperASR, OnlineASRProcessor, create_tokenizer
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, status
import numpy as np
from whisper_online_server import Connection, ServerProcessor
import numpy as np
import asyncio
import websockets
import soundfile as sf
import io

import torch

torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_properties(0))


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
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Convert bytes to np.float32 array
                # Assuming the incoming audio is in a common format like WAV
                with io.BytesIO(message) as audio_buffer:
                    # Read audio data as float32 directly
                    data, samplerate = sf.read(audio_buffer, dtype='float32')
                    if samplerate != OnlineASRProcessor.SAMPLING_RATE:
                        raise ValueError(f"Expected samplerate {OnlineASRProcessor.SAMPLING_RATE}, but got {samplerate}")

                # Process the float32 audio data
                asr.insert_audio_chunk(data)
                result = asr.process_iter()
                await websocket.send(str(result))  # Send back the partial transcription
    except Exception as e:
        print(f"Error: {e}")
    finally:
        final_output = asr.finish()
        await websocket.send(final_output)  # Send back the final transcription
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
