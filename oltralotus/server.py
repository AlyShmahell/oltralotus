import __info__
import cv2
import logging
import threading
import numpy as np 
import tempfile
import yaml
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.engine.results import Results
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, Request, Response
logging.basicConfig(level=logging.INFO)

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    def __new__(self, text: str, name: str = ''):
        if hasattr(self, name):
            return getattr(self, name) + text + self.ENDC
        return text

class Singleton(type):
    _instances = {}
    _lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class Models(defaultdict, metaclass=Singleton):
    pass
    
app = FastAPI()

class Detector:
    def __init__(self):
        self.metadata = {}
        self.router = APIRouter()
        self.router.add_api_route("/detect", self.detect_tcp, methods=["POST"])
        self.router.add_api_websocket_route("/detect", self.detect_wss)
    async def detect_tcp(self, request: Request):
        metadata = await request.json()
        self.metadata[request.client.host] = metadata
        return Response(status_code=200)
    async def detect_wss(self, websocket: WebSocket):
        model = self.metadata.get(websocket.client.host, {}).get("model", "")
        if not model:
            websocket.close()
            return
        tracker = self.metadata.get(websocket.client.host, {}).get("tracker", {})
        if tracker:
            trk_cnf = tempfile.NamedTemporaryFile('w', suffix='.yml')
            yaml.dump(tracker, trk_cnf)
        else:
            trk_cnf = None
        if websocket.client.host not in Models(dict) or model not in Models(dict)[websocket.client.host]:
            Models(dict)[websocket.client.host][model] = YOLO(model=model)
        await websocket.accept()
        try:
            while True:
                frame = await websocket.receive_bytes()
                frame = np.frombuffer(frame, np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if frame is not None:
                    if isinstance(trk_cnf, tempfile._TemporaryFileWrapper):
                        results : Results = Models(dict)[websocket.client.host][model].track(frame, persist=True, tracker=trk_cnf.name)
                    else:
                        results: Results = Models(dict)[websocket.client.host][model](frame)
                    await websocket.send_json([result.summary(normalize=False, decimals=5) for result in results])
        except WebSocketDisconnect:
            logging.info("Client disconnected")
        except Exception as e:
            logging.exception(f"Connection error: {e}")
        finally:
            await websocket.close()
            if isinstance(trk_cnf, tempfile._TemporaryFileWrapper):
                trk_cnf.close()

if __name__ == "__main__":
    import uvicorn
    detector = Detector()
    app.include_router(detector.router)
    logging.info(colors(f" Oltralotus Version {__info__.__version__}", "HEADER"))
    uvicorn.run(app, host="0.0.0.0", port=11535, log_level="info")