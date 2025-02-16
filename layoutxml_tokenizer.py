from transformers import LayoutXLMTokenizer

# Load LayoutXLM tokenizer
tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")

# Input text as a list of words
text = [ '''e Aadhaar linkage would continue to be mandatory for availing the subvention for short term
loans in 2021-22, 2022-23 and 2023-24.e All the short term loans processed in 2021-22, 2022-23 and 2023-24, which are eligible for
subvention, are required to be brought on the ISS portal/DBT platform. Lending institutions
have to capture and submit category-wise data of beneficiaries under the Scheme and report
the same on the ISS portal, individual farmer-wise, to settle the audited claims arising from
2021-22 onwards.''','''All the short term loans processed in 2021-22, 2022-23 and 2023-24, which are eligible for
subvention, are required to be brought on the ISS portal/DBT platform. Lending institutions
have to capture and submit category-wise data of beneficiaries under the Scheme and report
the same on the ISS portal, individual farmer-wise, to settle the audited claims arising from
2021-22 onwards.'''
]

# Dummy bounding boxes (each word must have a corresponding bounding box)
# Format: [x_min, y_min, x_max, y_max]
bboxes = [[312, 1028, 2224, 1392],[315 ,1150, 2223, 1392]]
# Tokenize with bounding boxes
tokens = tokenizer(text,boxes=bboxes,is_split_into_words=True,return_tensors="pt")
input_ids = tokens["input_ids"][0].tolist()

# Convert token IDs back to tokenized words
actual_tokens = tokenizer.convert_ids_to_tokens(input_ids)
freq=len(actual_tokens)
print("Total no of tokens generated:",freq)
text_string = " ".join(text)

# Split text into words based on whitespace
words = text_string.split()

# Count number of words
word_count = len(words)

print("Total number of words:", word_count)
fertility=freq/word_count
print("Fertility_Score",fertility)


#print(actual_tokens)
