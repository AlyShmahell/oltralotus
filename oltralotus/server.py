import __info__
import cv2
import gc
import logging
import threading
import numpy as np 
import tempfile
import yaml
import colorama
import rich
import torch
import time 
import datetime
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.engine.results import Results
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, Request, Response
logging.basicConfig(level=logging.INFO)

class colors:
    def __new__(self, text: str, name: str = ''):
        if hasattr(colorama.Fore, name):
            return getattr(colorama.Fore, name) + text + colorama.Fore.RESET
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
    def keysets(self, d, path=None, visited=None):
        if path is None:
            path = []
        if visited is None:
            visited = set()
        lines = []
        for key, value in d.items():
            if isinstance(value, (dict, Models)):
                if id(value) in visited:
                    continue
                visited.add(id(value))
                new_path = path + [key]
                lines.extend(self.keysets(value, new_path, visited))
            else:
                lines.append(path + [key])
        return lines
    def background(self):
        while True:
            keysets = self.keysets(self)
            for keys in keysets:
                remaining = int(self.timeout - (datetime.datetime.now() - keys[-1]).total_seconds())
                if remaining < 0:
                    logging.info(colors(f"deleting {' -> '.join(keys[:-1])}", "RED"))
                    del self[*keys[:-1]]
                else:
                    logging.info(f"awaiting {'->'.join(keys[:-1])} {colors(str(remaining), 'RED')}")
            time.sleep(1)
    def __init__(self, timeout=60):
        super().__init__(Models) 
        self.timeout = timeout
        threading.Thread(target=self.background, daemon=True).start()
    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(item, Models):
            return item
        else:
            value = [*item.values()][0]
            super().__setitem__(key, {datetime.datetime.now(): value})
            return value
    def __setitem__(self, key, value):
        if isinstance(value, Models):
            super().__setitem__(key, value)
        else:
            assert isinstance(value, YOLO)
            super().__setitem__(key, {datetime.datetime.now(): value})
    def __delitem__(self, key):
        if isinstance(key, tuple):
            current = self
            for k in key[:-1]:
                current = current[k]
            del current[key[-1]]
        else:
            super().__delitem__(key)
        gc.collect()
        torch.cuda.empty_cache()
    
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
        results = None
        try:
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
            if websocket.client.host not in Models() or model not in Models()[websocket.client.host]:
                Models(timeout=10)[websocket.client.host][model] = YOLO(model=model)
            await websocket.accept()
            while True:
                frame = await websocket.receive_bytes()
                frame = np.frombuffer(frame, np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if frame is not None:
                    if isinstance(trk_cnf, tempfile._TemporaryFileWrapper):
                        results : Results = Models()[websocket.client.host][model].track(frame, persist=True, tracker=trk_cnf.name, stream=True)
                    else:
                        results: Results = Models()[websocket.client.host][model](frame)
                    await websocket.send_json([result.summary(normalize=False, decimals=5) for result in results])
        except WebSocketDisconnect:
            logging.info("Client disconnected")
        except Exception as e:
            await websocket.close()

            logging.exception(f"Connection error: {e}")
        finally:
            del results
            if isinstance(trk_cnf, tempfile._TemporaryFileWrapper):
                trk_cnf.close()
            gc.collect()
            torch.cuda.empty_cache()
            logging.info(f"cleanup complete for {websocket.client.host } {model}")
            

if __name__ == "__main__":
    import uvicorn
    detector = Detector()
    app.include_router(detector.router)
    logging.info(colors(f" Oltralotus Version {__info__.__version__}", "MAGENTA"))
    uvicorn.run(app, host="0.0.0.0", port=11535, log_level="info")