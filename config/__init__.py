from .models import (
    USER_RETRIEVER_MODEL,
    USER_RETRIEVER_PROCESSOR,
    INSIGHTFACE_MODEL,
)

def get_face_embedding_model(name: str):
    """
    Returns the appropriate model configuration based on the given name.
    
    Args:
        name (str): Model identifier, e.g., "ULIP-p16", "CVLFace"

    Returns:
        dict: Dictionary containing model and related components
    """
    if name == "ULIP-p16":
        return {
            "name": "ULIP-p16",
            "model": USER_RETRIEVER_MODEL,
            "processor": USER_RETRIEVER_PROCESSOR
        }
    
    elif name == "INSIGHTFACE":
        return {
            "name": "INSIGHTFACE",
            "model": INSIGHTFACE_MODEL,
            "processor": None  # InsightFace does not require a separate processor
        }
    else:
        raise ValueError(f"Unknown embedding model name: {name}")
