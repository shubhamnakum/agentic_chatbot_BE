# Multi-Agent RAG Chatbot

This is a sophisticated chatbot application that uses LangGraph for multi-agent orchestration and RAG (Retrieval Augmented Generation) capabilities. The system employs multiple specialized agents working together to provide informative and accurate responses.

## Features

- Multi-agent system with specialized roles:
  - Researcher Agent: Analyzes queries and retrieves relevant information
  - Writer Agent: Composes coherent and contextual responses
  - Critic Agent: Reviews and improves responses
- RAG capabilities for enhanced knowledge retrieval
- FastAPI backend for easy integration
- Scalable architecture using LangGraph

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key
   - Adjust other settings as needed

## Running the Application

1. Start the FastAPI server:
```bash
python app.py
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### POST /chat
Send a message to the chatbot:
```json
{
    "message": "Your question here",
    "conversation_history": [
        {
            "role": "user",
            "content": "Previous message"
        },
        {
            "role": "assistant",
            "content": "Previous response"
        }
    ]
}
```

## Architecture

The system uses a directed graph of specialized agents:
1. The Researcher Agent analyzes the query and retrieves relevant information
2. The Writer Agent composes an initial response using the retrieved information
3. The Critic Agent reviews and improves the response

Each agent is powered by LangChain and uses the OpenAI API for natural language processing.

## Contributing

Feel free to submit issues and enhancement requests! 