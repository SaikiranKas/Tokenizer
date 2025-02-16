from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
#model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

input_ids = tokenizer(  "आज का मौसम बहुत सुहाना है। बच्चे पार्क में खेल रहे हैं।"
, return_tensors="pt").input_ids[0].tolist()
actual_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(actual_tokens)