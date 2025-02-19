import cv2
import time
import json
import httpx
import asyncio
import websockets



async def func():
    async with httpx.AsyncClient() as client:
        tracker = {
            'tracker_type': 'bytetrack' ,
            'track_high_thresh': 0.25 ,
            'track_low_thresh': 0.1 ,
            'new_track_thresh': 0.25 ,
            'track_buffer': 30,
            'match_thresh': 0.8 ,
            'fuse_score': True
        }
        resp = await client.post("http://localhost:11535/detect", json={"model": "yolo11n-seg.pt", "tracker": tracker})
        if resp.status_code != 200:
            return
    uri = "ws://localhost:11535/detect"
    cap = cv2.VideoCapture(2)
    W, H = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        break
                    s = time.perf_counter()
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await websocket.send(buffer.tobytes())
                    resp = await websocket.recv()
                    t = time.perf_counter() -s 
                    print(t)
                    try:
                        resp = json.loads(resp)
                    except Exception as e:
                        resp = []
                    if resp:
                        for entries in resp:
                            if entries:
                                for entry in entries:
                                    cv2.rectangle(
                                        frame, 
                                        (
                                            int(entry.get("box", {}).get("x1")), 
                                            int(entry.get("box", {}).get("y1"))
                                        ), 
                                        (
                                            int(entry.get("box", {}).get("x2")), 
                                            int(entry.get("box", {}).get("y2"))
                                        ), 
                                        (0, 255, 0), 
                                        2
                                    )
                                    cv2.putText(
                                        frame, 
                                        entry.get("name", ""), 
                                        (
                                            int(entry.get("box", {}).get("x1")), 
                                            int(entry.get("box", {}).get("y1")) - 10
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, 
                                        (0, 255, 0), 
                                        2, 
                                        cv2.LINE_AA
                                    )
                                    cv2.putText(
                                        frame, 
                                        f'ID #{entry.get("track_id", "")}', 
                                        (
                                            int(entry.get("box", {}).get("x2")), 
                                            int(entry.get("box", {}).get("y1")) - 10
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, 
                                        (0, 255, 0), 
                                        2, 
                                        cv2.LINE_AA
                                    )
                                    cv2.imshow('', frame)
                                    cv2.waitKey(1)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection lost: {e}. Reconnecting...")
            await asyncio.sleep(2) 
            cap.release()

asyncio.run(func())
