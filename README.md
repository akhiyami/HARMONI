# Context-aware conversation

A framework for context-aware, multi-turn conversations with robots.  
Its goal is to enable robots to dynamically adapt to individuals and groups while respecting ethical and social constraints.

![System Architecture](documentation/architecture.png)

### Main features

- **Perception Module:** Interprets video input by extracting spoken questions and identifying the active speaker.
- **User Modeling Module:** Retrieves and maintains user-specific profiles and long-term memory.
- **World Modeling Module:** Maintains short-term conversational sessions and retrieves relevant memories linked to the user profiles present in the environment.
- **Generation Module:** Generates responses conditioned on both the user profile and associated memory, ensuring contextually appropriate and personalized outputs.

> [!NOTE]  
> This is a working version, only a subset of features is implemented

### Work interfaces

The framework provides several web applications that simulate different aspects of interaction between users and the assistant.  
These interfaces are useful both for testing and for developing new features.

- **Chatbox App** (Text-based): 
  - Users interact with the system through a text chatbox.
  - Each message is linked to a user, selected by uploading/choosing a picture.
  - Users are stored in a database that can be modified directly through the app.
  - Context and session management are supported (start new sessions, reset database).
    
- **Video Interaction App** (Speaker-based):
  - Interaction is driven by video input rather than text.
  - The system identifies the active speaker in the video (assumed to contain one interaction).
  - The spoken question is transcribed, and both the transcription and system response are shown in a chatbox-like interface.
  - The same database, context, and session management tools as in the text app are available.
    
## Installation
  
1. Clone the repository
```bash
git clone git@github.com:MalecotJeanne/Context-aware_Conversation.git
cd Context-aware_Conversation
```
2. Install dependencies
```bash
pip install -r requirements.txt
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
> You have to choose a user image before starting interracting with the system

### Chatbox App

### Video Interaction App 

## Database management

Some python scripts are available in the folder `scripts` to perform basic operations on the user database (`users.db`)

### Usage in command line:

- Forget the profile of an user

```bash
python scripts/forget_profile.py -u [user_id]   
```

>The features will be erased, but the facial encoding for the user will remain
- Erase an user from the database

```bash
python scripts/remove_user.py -u [user_id]   
```

>All the information about the user will be erased (facial encoding and user profile)
- Clear the database

```bash
python scripts/clean_db.py
```

>Every user and its associated informations will be deleted.



