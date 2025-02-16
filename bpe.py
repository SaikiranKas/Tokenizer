from transformers import AutoTokenizer

# Load a pretrained tokenizer (GPT-2 uses BPE)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Example text
text = '''తెలుగు ఒక అందమైన భాషగా ప్రసిద్ధి చెందింది. ఇది భారతదేశంలోని దక్షిణ భారత రాష్ట్రం ఆంధ్రప్రదేశ్ మరియు తెలంగాణలో ప్రధాన భాషగా మాట్లాడబడుతుంది. తెలుగును "భాషా సముద్రం" అని కూడా అంటారు, ఎందుకంటే దీని వ్యాకరణం మరియు పదసంపత్తి విస్తృతంగా ఉంది. ఈ భాషకు గొప్ప సాహిత్య పరంపర ఉంది, అందులో పద్యాలు, కథలు, మరియు నాటకాలు ప్రత్యేకమైన స్థానం కలిగి ఉన్నాయి. తెలుగు ప్రజలు సంస్కృతి, సంప్రదాయాలను గౌరవిస్తారు మరియు ఉత్సవాలను వేడుకగా జరుపుకుంటారు. "మాతృభాష" తెలుగును కాపాడటం ప్రతి తెలుగువారి కర్తవ్యంగా భావించవచ్చు. తెలుగు అక్షరాల అందం, సంగీతం లాంటి మాధుర్యం తెలుగు భాషకు ప్రత్యేకతను కలిగించాయి,'''

# Tokenize the text
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)

Decode back to text
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)
