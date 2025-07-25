import torch
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def generate_description(image, model, processor):
    prompt = "<image> "
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded

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
    return emotions[emotion_index], emotion_probs[0][emotion_index]