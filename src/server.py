
import cv2
import logging
import numpy as np 
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
logging.basicConfig(level=logging.INFO)
models = {}
app = FastAPI()

@app.websocket("/detect/{model}")
async def video_endpoint(websocket: WebSocket, model: str):
    if model not in models:
        models[model] = YOLO(model=model)
    await websocket.accept()
    try:
        while True:
            frame = await websocket.receive_bytes()
            frame = np.frombuffer(frame, np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            if frame is not None:
                results = models[model](frame)
                await websocket.send_json({
                    idx: result.to_json()
                    for idx, result in enumerate(results)
                })
    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception as e:
        logging.exception(f"Connection error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11535, log_level="info")