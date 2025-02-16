import sentencepiece as spm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
bpe_tokenizer = Tokenizer(models.BPE())
sample_text = '''తెలుగు ఒక అందమైన భాషగా ప్రసిద్ధి చెందింది. ఇది భారతదేశంలోని దక్షిణ భారత రాష్ట్రం ఆంధ్రప్రదేశ్ మరియు తెలంగాణలో ప్రధాన భాషగా మాట్లాడబడుతుంది. తెలుగును "భాషా సముద్రం" అని కూడా అంటారు, ఎందుకంటే దీని వ్యాకరణం మరియు పదసంపత్తి విస్తృతంగా ఉంది. ఈ భాషకు గొప్ప సాహిత్య పరంపర ఉంది, అందులో పద్యాలు, కథలు, మరియు నాటకాలు ప్రత్యేకమైన స్థానం కలిగి ఉన్నాయి. తెలుగు ప్రజలు సంస్కృతి, సంప్రదాయాలను గౌరవిస్తారు మరియు ఉత్సవాలను వేడుకగా జరుపుకుంటారు. "మాతృభాష" తెలుగును కాపాడటం ప్రతి తెలుగువారి కర్తవ్యంగా భావించవచ్చు. తెలుగు అక్షరాల అందం, సంగీతం లాంటి మాధుర్యం తెలుగు భాషకు ప్రత్యేకతను కలిగించాయి.'''
# --------------------- SentencePiece Tokenizer ---------------------
# Save sample text to a file for SentencePiece training
with open("sample.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)

# Use a pre-tokenizer (whitespace-based)
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train the BPE tokenizer
trainer = trainers.BpeTrainer(vocab_size=1310, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
bpe_tokenizer.train(["sample.txt"], trainer)

# Tokenize text using BPE
bpe_tokens = bpe_tokenizer.encode(sample_text).tokens
print("BPE Tokens:", bpe_tokens)