from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from whisper_online import FasterWhisperASR, OnlineASRProcessor
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, status

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    # Redirect to the static HTML file
    return RedirectResponse(url="/static/index.html")

src_lan = "en"

# Initialize the ASR system
asr = FasterWhisperASR(lan=src_lan, modelsize="large-v2")
online = OnlineASRProcessor(asr)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted") 
    try:
        while True:
            # Receive audio chunk from client
            audio_chunk = await websocket.receive_bytes()
            
            # Check if audio has ended (you might define a specific message type for this)
            if not audio_chunk:
                break
            
            # Insert audio chunk into the processor
            online.insert_audio_chunk(audio_chunk)
            print("Transcribing...")
            # Process the current audio chunk and obtain partial transcription
            transcription = online.process_iter()
            
            # Send the transcription back to the client
            await websocket.send_text(str(transcription))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Handle any remaining audio data
        final_output = online.finish()
        await websocket.send_text(str(final_output))
        await websocket.close()
        # Re-initialize if re-using the object for another connection
        online.init()
        print("WebSocket connection closed") 

    # Re-initialize if re-using the object for another connection
    online.init()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
