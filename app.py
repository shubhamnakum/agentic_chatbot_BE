from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import logging
from workflow import process_message
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import asyncio
import json
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
import jwt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent RAG Chatbot")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 180


# Update Token model
class Token(BaseModel):
    access_token: str
    token_type: str
    isDev: bool


# User credentials
CREDENTIALS = {
    "username": "admin",
    "password": "password123"
}

DEV_CREDENTIALS = {
    "username": "dev",
    "password": "password123"
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    domain: List[str]


class ChatResponse(BaseModel):
    response: str
    conversation_history: List[ChatMessage]


@app.get("/")
async def root():
    return {"status": "ok", "message": "Chat API is running"}


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(
            status_code=401, detail="Could not validate credentials")
    return username


@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Check for dev credentials
    if (form_data.username == DEV_CREDENTIALS["username"] and
            form_data.password == DEV_CREDENTIALS["password"]):
        user_type = "dev"
        is_dev = True

    # Check for admin credentials
    elif (form_data.username == CREDENTIALS["username"] and
          form_data.password == CREDENTIALS["password"]):
        user_type = "admin"
        is_dev = False

    else:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token with user_type
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username, "user_type": user_type},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "isDev": is_dev
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, current_user: str = Depends(get_current_user)):

    try:
        logger.info(f"Received chat request: {request.message}")

        # Convert conversation history to the format expected by agents
        messages = [{"role": msg.role, "content": msg.content}
                    for msg in request.conversation_history]

        # Initialize state for the agents
        state = {
            "messages": messages,
            "current_input": request.message,
            "retrieved_docs": "",
            "draft_response": "",
            "final_response": "",
            "domain": request.domain
        }

        # Process the message through our multi-agent system
        try:
            result = process_message(state, request.message, request.domain)
            logger.info("Successfully processed message through agents")

            # Get the final response
            response = result.get(
                "final_response", "I apologize, but I couldn't generate a response at this time.")

            # Update conversation history
            conversation_history = request.conversation_history
            conversation_history.append(ChatMessage(
                role="user", content=request.message))
            conversation_history.append(ChatMessage(
                role="assistant", content=response))

            return ChatResponse(
                response=response,
                conversation_history=conversation_history
            )

        except Exception as e:
            logger.error(f"Error in processing message: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your message"
            )

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: str = Depends(get_current_user)):
    try:
        logger.info(f"Received streaming chat request: {request.message}")

        # Convert conversation history to the format expected by agents
        messages = [{"role": msg.role, "content": msg.content}
                    for msg in request.conversation_history]

        # Initialize state for the agents
        state = {
            "messages": messages,
            "current_input": request.message,
            "retrieved_docs": "",
            "draft_response": "",
            "final_response": "",
            "domain": request.domain
        }

        # Process the message through our multi-agent system
        try:
            result = process_message(state, request.message, request.domain)
            cleaned_docs = result.get("cleaned_docs", [])
            logger.info("Successfully processed message through agents")

            # Get the final response
            response = result.get(
                "final_response", "I apologize, but I couldn't generate a response at this time.")

            # Split the response into words for more natural streaming
            words = response.split()

            async def stream_response():
                # ✅ Step 1: Send metadata with cleaned_docs
                meta_payload = {
                    "meta": {
                        "cleaned_docs": cleaned_docs
                    }
                }
                yield f"data: {json.dumps(meta_payload)}\n\n"
                await asyncio.sleep(0.1)

                # ✅ Step 2: Stream word by word for more natural reading
                for i, word in enumerate(words):
                    # Add space before words except the first one
                    prefix = " " if i > 0 else ""
                    chunk = prefix + word
                    is_last = i == len(words) - 1

                    # Send each chunk as a separate event with proper formatting
                    yield f"data: {json.dumps({'chunk': chunk, 'done': is_last})}\n\n"
                    await asyncio.sleep(0.05)  # Consistent delay between words

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )

        except Exception as e:
            logger.error(f"Error in processing message: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your message"
            )

    except Exception as e:
        logger.error(f"Unexpected error in chat stream endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )

if __name__ == "__main__":
    import uvicorn
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set!")
        exit(1)

    HOST = "0.0.0.0"  # Default to 0.0.0.0 for Docker
    PORT = 8000
    DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

    logger.info("Starting chat server...")

    if DEBUG_MODE:
        logger.info("Development mode on..")
        # Use import string format when reload is True
        uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
    else:
        logger.info("Production mode on..")
        # Can use direct object reference when reload is False
        uvicorn.run(app, host=HOST, port=PORT)
