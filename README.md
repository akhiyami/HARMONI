# Context-aware conversation

A simple interface to chat with a virtual assistant with a visual memory of users.

## Installation

1. Clone the repository

```bash
git clone git@github.com:MalecotJeanne/Context-aware_Conversation.git
cd Context-aware_Conversation
```

2. Install dependecies

```bash
pip install requirements.txt
```

## Configuration

### Environment variables:

You will need an *OpenAI API key* and a *HuggingFace token* :

```bash
touch .env
echo "API_KEY=your_api_key_here" >> .env
echo "SECRET_KEY=your_secret_key_here" >> .env
```

## Execution

To run the conversation app :

```bash
uvicorn app:app --reload
```

> [!WARNING]  
> You have to chose a user image before starting interracting wth the system