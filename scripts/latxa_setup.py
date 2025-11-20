from transformers import pipeline
pipe = pipeline("text-generation", model="HiTZ/latxa-7b-v1.1")
text = "Euskara adimen artifizialera iritsi da!"
pipe(text, max_new_tokens=50, num_beams=5)