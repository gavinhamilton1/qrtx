from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import uuid
import os
from pathlib import Path
from typing import Dict
from fountain_code import LTEncoder, LTDecoder

app = FastAPI()

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for the session state
# In a real production app, use Redis/Memcached
active_encoders: Dict[str, LTEncoder] = {}
active_decoders: Dict[str, LTDecoder] = {}
# Store file metadata (filename, content_type) for each session
file_metadata: Dict[str, Dict[str, str]] = {}

# --- Pydantic Models for Validation ---

class InitReceiverRequest(BaseModel):
    session_id: str
    num_blocks: int
    block_size: int = 256

class DropletPayload(BaseModel):
    seed: int
    data: str
    num_blocks: int
    file_size: int
    block_size: int = 256

class SubmitDropletRequest(BaseModel):
    session_id: str
    droplet: DropletPayload

# Public directory for static files
public_dir = BASE_DIR / "public"

app.mount("/public", StaticFiles(directory=os.path.join(BASE_DIR, ".", "public")), name="public")

# --- Routes ---

@app.get("/")
async def index():
    # FastAPI doesn't use 'render_template' by default, 
    # we just serve the static HTML file directly.
    return FileResponse(str(BASE_DIR / 'templates' / 'index.html'))


@app.post("/upload")
async def upload(request: Request):
    """
    Handle file upload and initialize the Fountain Code Encoder.
    """
    try:
        # Get form data
        form = await request.form()
        
        print(f"=== UPLOAD DEBUG ===")
        print(f"Form keys: {list(form.keys())}")
        print(f"Form items: {[(k, type(v).__name__) for k, v in form.items()]}")
        
        # Get file from form
        if "file" not in form:
            raise HTTPException(status_code=400, detail="No file provided")
        file = form["file"]
        
        # Get original filename and content type
        original_filename = file.filename if hasattr(file, 'filename') else "file"
        content_type = file.content_type if hasattr(file, 'content_type') else "application/octet-stream"
        
        # Get block_size from form data - try multiple ways
        block_size_str = form.get("block_size")
        if not block_size_str:
            # Try alternative key names
            block_size_str = form.get("block-size") or form.get("blockSize") or "256"
            print(f"block_size not found with 'block_size', trying alternatives...")
        
        print(f"Received block_size from form: '{block_size_str}' (type: {type(block_size_str)})")
        print(f"All form keys: {list(form.keys())}")
        
        # If block_size is still default, check if it's in the form at all
        if block_size_str == "256" or not block_size_str:
            print(f"WARNING: Using default block_size. All form values: {[(k, str(v)[:50] if hasattr(v, 'read') else v) for k, v in form.items()]}")
        
        # Validate and convert block size
        try:
            block_size = int(block_size_str)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert block_size '{block_size_str}' to int, using default 256")
            block_size = 256  # Default if invalid
        
        # Block size validation removed - allow any positive integer
        if block_size <= 0:
            raise HTTPException(status_code=400, detail="block_size must be a positive integer")
        
        # Read the file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        if len(content) > 52428800:  # 50MB
            raise HTTPException(status_code=400, detail="File too large for demo (Limit 50MB)")
        
        session_id = str(uuid.uuid4())[:8]
        # Initialize encoder with specified block size
        try:
            print(f"About to create encoder: block_size={block_size}, file_size={len(content)} bytes")
            expected_blocks = (len(content) + block_size - 1) // block_size
            print(f"Expected num_blocks for {len(content)} bytes with block_size={block_size}: {expected_blocks}")
            
            encoder = LTEncoder(content, block_size=block_size)
            print(f"Encoder created: block_size={encoder.block_size}, num_blocks={encoder.num_blocks}")
            
            # Verify the encoder actually used the correct block size
            if encoder.block_size != block_size:
                print(f"ERROR: Encoder block_size ({encoder.block_size}) doesn't match requested ({block_size})!")
                print(f"FIXING: Recreating encoder with correct block_size")
                encoder = LTEncoder(content, block_size=block_size)
                print(f"After fix: block_size={encoder.block_size}, num_blocks={encoder.num_blocks}")
            
            if encoder.num_blocks == (len(content) + 64 - 1) // 64:
                print(f"CRITICAL WARNING: num_blocks ({encoder.num_blocks}) matches block_size=64 calculation!")
                print(f"This means the encoder is using block_size=64 instead of {block_size}!")
                print(f"Encoder.block_size attribute: {encoder.block_size}")
                print(f"Recreating with explicit block_size={block_size}...")
                encoder = LTEncoder(content, block_size=block_size)
                print(f"After explicit recreation: block_size={encoder.block_size}, num_blocks={encoder.num_blocks}")
            
            active_encoders[session_id] = encoder
            # Store block size with session for decoder initialization
            active_encoders[session_id]._block_size = block_size
            # Store file metadata for reconstruction
            file_metadata[session_id] = {
                "filename": original_filename,
                "content_type": content_type
            }
            print(f"Final: block_size={block_size}, num_blocks={encoder.num_blocks}, file_size={len(content)}")
            print(f"File metadata: filename={original_filename}, content_type={content_type}")
        except Exception as e:
            print(f"Exception creating encoder: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error initializing encoder: {str(e)}")
        
        response_data = {
            "session_id": session_id, 
            "mode": "sender", 
            "block_size": int(block_size),  # Ensure it's an int, not a string
            "num_blocks": int(encoder.num_blocks),
            "file_size": int(len(content)),
            "filename": original_filename,
            "content_type": content_type
        }
        print(f"Returning response: {response_data}")
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.get("/next_droplet/{session_id}")
async def next_droplet(session_id: str):
    """
    Generate the next 'droplet' (Fountain Code packet) for the QR stream.
    """
    encoder = active_encoders.get(session_id)
    if not encoder:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # generate_droplet returns a dict, FastAPI automatically converts to JSON
    return encoder.generate_droplet()

@app.post("/init_receiver")
async def init_receiver(request: InitReceiverRequest):
    """
    Initialize a decoder session on the receiver side.
    """
    if request.session_id not in active_decoders:
        active_decoders[request.session_id] = LTDecoder(request.num_blocks, block_size=request.block_size)
        # For test transfers, sender and receiver use the same session_id
        # So file metadata should already be available from the upload
        # In production, metadata would be shared separately (e.g., in first QR code or separate channel)
        if request.session_id not in file_metadata:
            print(f"Warning: No file metadata found for session {request.session_id}")
    return {"status": "ready"}

@app.post("/submit_droplet")
async def submit_droplet(request: SubmitDropletRequest):
    """
    Receive a scanned QR packet and attempt to solve the file.
    """
    decoder = active_decoders.get(request.session_id)
    if not decoder:
        raise HTTPException(status_code=404, detail="No decoder found")
    
    # Pass the dict representation of the droplet to the decoder logic
    # model_dump() is preferred in Pydantic v2, dict() for v1 compatibility
    try:
        droplet_data = request.droplet.model_dump() 
    except AttributeError:
        droplet_data = request.droplet.dict()

    is_complete = decoder.add_droplet(droplet_data)
    
    # If this is the first droplet and we have metadata from encoder, copy it to decoder session
    # This ensures the download endpoint has the correct file metadata
    if decoder.num_solved == 1 and request.session_id not in file_metadata:
        # Try to find metadata from encoder (for test transfer, sender and receiver use same session_id)
        # In production, metadata would be shared separately
        pass
    
    progress = 0.0
    if decoder.num_blocks > 0:
        progress = (decoder.num_solved / decoder.num_blocks) * 100
    
    result_url = None
    if is_complete:
        result_url = f"/download/{request.session_id}"
        
    return {
        "progress": progress,
        "is_complete": is_complete,
        "result_url": result_url
    }

@app.get("/download/{session_id}")
async def download(session_id: str):
    """
    Stream the reconstructed file back to the user.
    """
    decoder = active_decoders.get(session_id)
    if not decoder or not decoder.is_complete():
        raise HTTPException(status_code=404, detail="File not ready")
        
    result_bytes = decoder.get_result()
    
    # Get file metadata if available
    metadata = file_metadata.get(session_id, {})
    filename = metadata.get("filename", "transferred_file.bin")
    content_type = metadata.get("content_type", "application/octet-stream")
    
    # StreamingResponse is more efficient for file downloads
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

if __name__ == '__main__':
    import uvicorn
    # Run with: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)