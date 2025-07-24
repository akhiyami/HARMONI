from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch

from transformers.image_utils import load_image


def generate_description(image, model, processor):
    prompt = "<image> "
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded

if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_id = "ACIDE/User-VLM-10B-base"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

    url = "https://media.istockphoto.com/id/1403196779/fr/photo/une-joyeuse-famille-m%C3%A9tisse-de-trois-personnes-se-relaxant-dans-le-salon-et-jouant-ensemble.webp?s=2048x2048&w=is&k=20&c=vNqwOIBiOXOTaapic91TOZiOSpnqFymTiT99b4Vgb4c="
    image = load_image(url)

    description = generate_description(image, model, processor)
    print(description)
