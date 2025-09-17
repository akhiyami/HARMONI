import torch
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor



def prob_emotions(image, model, processor):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError("Input must be a NumPy array or PIL.Image")
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        return logits.softmax(dim=1).detach().cpu().numpy()
    
def detect_emotions(imgs, model, processor):
    with ThreadPoolExecutor() as executor:
        probs_list = list(executor.map(lambda img: prob_emotions(img, model, processor), imgs))
    probs_list = np.array(probs_list)

    emotions = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
    emotion_probs = np.mean(probs_list, axis=0)
    emotion_index = np.argmax(emotion_probs)
    return emotions[emotion_index], emotion_probs[0][emotion_index], emotions, emotion_probs[0].tolist()