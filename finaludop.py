import os
from bs4 import BeautifulSoup

# Corrected path using raw string (best approach)
folder_path = r"C:\Users\Saikiran Kasturi\OneDrive\Desktop\Tokenizer\data\hindi"

# Lists to store extracted paragraphs and bounding boxes
paragraphs = []
bounding_boxes = []

# Loop through all HOCR files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".hocr"):  # Process only HOCR files
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "lxml")  # Parse HOCR file

            # Find all <p> tags
            for p_tag in soup.find_all("p"):
                # Get paragraph text
                paragraph_text = p_tag.get_text(strip=True)
                
                # Extract bounding box from class attribute
                title_attr = p_tag.get("title", "")
                bbox = None
                if "bbox" in title_attr:
                    bbox_values = title_attr.split("bbox")[-1].strip().split(";")[0].strip()
                    bbox = [int(x) for x in bbox_values.split()]  # Convert to list of integers
                
                # Store results if paragraph exists and bbox is found
                if paragraph_text and bbox:
                    paragraphs.append(paragraph_text)
                    bounding_boxes.append(bbox)
                    
import json
from transformers import AutoTokenizer

# Load the UDOP tokenizer
tokenizer_name = "microsoft/udop-large"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Sample Data (Replace these lists with your actual data)


# Dictionary to store results
results = {tokenizer_name: {}}

# Variables to track total words and tokens for entire document
total_words = 0
total_tokens = 0

# Tokenization, word counting, and fertility score calculation
for i, (paragraph, bbox) in enumerate(zip(paragraphs, bounding_boxes)):
    # Pre-tokenizing words (UDOP requires List[str] format)
    words = paragraph.split()  # Splitting into words
    num_words = len(words)  # Counting words
    
    # Update document-level word count
    total_words += num_words
    
    # Assign the same bounding box to every word in the paragraph
    word_bboxes = [bbox] * num_words  # Duplicate the bounding box for each word
    
    # Tokenizing paragraph with duplicated bounding boxes
    encoded_input = tokenizer(words, boxes=word_bboxes, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

    # Extracting tokenized output
    tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
    num_tokens = len(tokens)  # Counting generated tokens
    
    # Update document-level token count
    total_tokens += num_tokens

    # Calculating paragraph-level fertility score
    fertility_score = num_tokens / num_words if num_words > 0 else 0

    # Store paragraph-level results
    results[tokenizer_name][f"Paragraph_{i+1}"] = {
        "Tokens": tokens,
        "Fertility_Score": fertility_score
    }

# Compute overall document fertility score
document_fertility_score = total_tokens / total_words if total_words > 0 else 0

# Store document-level results
results[tokenizer_name]["Document_Level"] = {
    "Total_Words": total_words,
    "Total_Tokens": total_tokens,
    "Fertility_Score": document_fertility_score
}

# Save results to JSON file
json_filename = "tokenization_results.json"
with open(json_filename, "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, indent=4)

# Print results
print(json.dumps(results, indent=4))
print(f"Results saved to {json_filename}")
