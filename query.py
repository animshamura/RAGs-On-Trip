# Input query
query = "What is the role of mitochondria in a cell?"

# Tokenize
inputs = tokenizer(query, return_tensors="pt")

# Generate
with torch.no_grad():
    generated = model.generate(input_ids=inputs["input_ids"])

# Decode and print answer
answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print("Answer:", answer)
