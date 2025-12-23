# from transformers import pipeline
# pipe = pipeline("text-generation", model="HiTZ/latxa-7b-v1.1")
# text = "Euskara adimen artifizialera iritsi da!"
# pipe(text, max_new_tokens=50, num_beams=5)

from transformers import AutoTokenizer, AutoModelForCausalLM
latxa7B_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
latxa7B_model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
latxa70B_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-70b-v1.2")
latxa70B_model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-70b-v1.2")
llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
llama2_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")