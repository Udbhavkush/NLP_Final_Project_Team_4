from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-3B")


prompt = "I am depressed. Can you please tell me what to do?"

inputs = tokenizer(prompt, return_tensors='pt')

res = model.generate(**inputs)

print(tokenizer.decode(res[0]))