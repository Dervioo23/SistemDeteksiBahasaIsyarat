
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio    

# Import modules from app (assuming hybrid structure)
from app.core.config import settings
from app.websocket.manager import manager
from app.services.inference import inference_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title=settings.PROJECT_NAME)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "project_name": settings.PROJECT_NAME})

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Create session for this connection
    from app.services.session import SignLanguageSession
    session = SignLanguageSession()
    
    # Store session in manager (optional, or just keep local ref)
    # For simplicity, we keep local ref here, but if we need HTTP endpoints to affect this session
    # we would need to map websocket -> session in the manager.
    # For now, let's assume commands come via WebSocket too or we make a simple session map.
    
    try:
        while True:
            # Receive data (bytes or text)
            # data = await websocket.receive_bytes() # This only accepts bytes
            
            # Hybrid receiver: check type
            message = await websocket.receive()
            
            if "bytes" in message:
                # Video Frame
                data = message["bytes"]
                loop = asyncio.get_event_loop()
                try:
                    # Run session processing
                    result = await loop.run_in_executor(None, session.process_frame_data, data)
                    await manager.send_json(result, websocket)
                except WebSocketDisconnect:
                    # Normal disconnect - user closed connection
                    raise  # Re-raise to exit the while loop
                except Exception as frame_error:
                    logger.error(f"Frame processing error: {frame_error}")
                    try:
                        await manager.send_json({"error": str(frame_error)}, websocket)
                    except:
                        pass  # Connection already closed
            
            elif "text" in message:
                # Text Command - Keyboard shortcuts and controls
                text = message["text"]
                response = None
                
                # Handle raw text commands from frontend
                # Also support JSON format as fallback: {"text": "command"}
                command = text
                if text.startswith("{"):
                    try:
                        import json
                        parsed = json.loads(text)
                        command = parsed.get("text", text)
                    except json.JSONDecodeError:
                        command = text
                
                # --- COMMAND HANDLERS (matching run_inference_multiclass.py) ---
                
                if command == "toggle_mode":  # TAB key
                    response = session.toggle_mode()
                    
                elif command == "commit_spelling":  # SPACE key
                    response = session.commit_spelling()
                    
                elif command == "local_brain":  # 8 key
                    response = await asyncio.to_thread(session.ask_local_brain)
                    
                elif command == "finish_sentence":  # ENTER key
                    response = await asyncio.to_thread(session.finish_sentence)
                    
                elif command == "backspace":  # BACKSPACE key
                    response = session.backspace()
                    
                elif command == "clear_session":  # 9 key
                    response = session.clear()
                    
                elif command == "speaking_done":  # Called when TTS finishes
                    response = session.set_speaking_done()
                
                if response:
                    await manager.send_json(response, websocket)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        import traceback
        logger.error(f"WebSocket error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"ERROR in WebSocket: {e}")
        print(traceback.format_exc())
        try:
            manager.disconnect(websocket)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        ws_ping_timeout=60,      # Increase from default 20s
        ws_ping_interval=30      # Increase ping interval
    )
