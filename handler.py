import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

# Modell und Prozessor laden
model_id = "mPLUG/DocOwl2"
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    revision="main",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id, revision="main")


def handler(event, context):
    try:
        data = event.get("body", {})

        # Sprache (optional)
        lang = data.get("lang", "de")

        # Prompt in der gewünschten Sprache
        default_prompts = {
            "de": "Beschreibe das Dokument detailliert.",
            "en": "Describe the document in detail."
        }
        prompt = data.get("prompt", default_prompts.get(lang, default_prompts["de"]))

        # Bild decodieren
        image_b64 = data.get("image")
        if not image_b64:
            return {"statusCode": 400, "body": "Fehler: Kein Bild im Base64-Format gesendet."}

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Eingaben vorbereiten
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        # Modellvorhersage
        outputs = model.generate(**inputs)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return {
            "statusCode": 200,
            "body": {
                "prompt": prompt,
                "result": generated_text
            }
        }
    except Exception as e:
        return {"statusCode": 500, "body": f"Interner Fehler: {str(e)}"}
