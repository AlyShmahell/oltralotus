import cv2
import time
import json
import asyncio
import websockets

async def send_video():
    uri = "ws://localhost:11535/detect/yolo11n.pt"
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
                    try:
                        resp = json.loads(resp)
                    except Exception as e:
                        resp = {}
                    if resp:
                        for entries in resp.values():
                            try:
                                entries = json.loads(entries)
                            except:
                                entries = {}
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
                                    cv2.imshow('', frame)
                                    cv2.waitKey(1)
                    t = time.perf_counter() -s 
                    print(t)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection lost: {e}. Reconnecting...")
            await asyncio.sleep(2) 
            cap.release()

asyncio.run(send_video())
