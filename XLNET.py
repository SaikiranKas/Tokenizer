from transformers import XLNetTokenizer

# Load pre-trained XLNet tokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# Example text to tokenize

inp=["Photosynthesis","Nonetheless","Wow! 🚀 AI is evolving fast. #MachineLearning #AI🤖","Check out https://example.com, it's awesome! Also, email me at test@example.org.","The company’s revenue jumped from $5M to ₹40Cr in FY’23—an astounding 800% growth!","आज का मौसम बहुत सुहाना है। बच्चे पार्क में खेल रहे हैं।","ఈ రోజు వాతావరణం చాలా చల్లగా ఉంది. పిల్లలు పార్క్‌లో ఆడుతున్నారు.","आजचे हवामान खूप छान आहे. मुले बागेत खेळत आहेत."]
for i in range(0,len(inp)):
    text=inp[i]

# Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Convert token IDs back to words
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist())

# Print tokenized output
    print("Tokenized Input IDs:", tokens["input_ids"])
    print("Attention Mask:", tokens["attention_mask"])
    print("Decoded Tokens:", decoded_tokens)