from transformers import MT5Tokenizer

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

text = "आजचे हवामान खूप छान आहे. मुले बागेत खेळत आहेत."
  # Telugu text
tokens = tokenizer.encode(text, add_special_tokens=True)

print("Tokenized Output:", tokens)
actual_tokens = tokenizer.convert_ids_to_tokens(tokens)
print(actual_tokens)