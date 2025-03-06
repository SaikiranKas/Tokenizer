import os
import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

def process_hocr_folder(folder_path):
    # Load the Galactica tokenizer
    tokenizer_name = "facebook/galactica-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Fix: Add a custom PAD token explicitly
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Define a new PAD token

    # Dictionary to store results
    results = {tokenizer_name: {}}

    # Variables to track total words and tokens for the entire document
    total_doc_words = 0
    total_doc_tokens = 0

    # Loop through all HOCR files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".hocr"):  # Process only HOCR files
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "lxml")  # Parse HOCR file

                # Lists to store extracted paragraphs per page
                paragraphs = []

                # Find all <p> tags
                for p_tag in soup.find_all("p"):
                    # Get paragraph text
                    paragraph_text = p_tag.get_text(strip=True)
                    
                    # Store results if paragraph exists
                    if paragraph_text:
                        paragraphs.append(paragraph_text)

            # Tokenization, word counting, and fertility score calculation per page
            total_words = 0
            total_tokens = 0
            
            for paragraph in paragraphs:
                words = paragraph.split()  # Splitting into words
                num_words = len(words)
                total_words += num_words
                total_doc_words += num_words
                
                # Tokenizing paragraph
                encoded_input = tokenizer(
                    paragraph, 
                    return_tensors="pt", 
                    padding="longest",  # Ensures longest sequence is padded
                    truncation=True, 
                    max_length=512 # Prevents overly long sequences
                )

                # Extracting tokenized output
                tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
                num_tokens = len(tokens)
                total_tokens += num_tokens
                total_doc_tokens += num_tokens

            # Compute page-level fertility score
            page_fertility_score = total_tokens / total_words if total_words > 0 else 0

            # Store page-level results
            results[tokenizer_name][filename] = {
                "Total_Words": total_words,
                "Total_Tokens": total_tokens,
                "Fertility_Score": page_fertility_score
            }

    # Compute document-level fertility score
    document_fertility_score = total_doc_tokens / total_doc_words if total_doc_words > 0 else 0
    results[tokenizer_name]["Document_Level"] = {
        "Total_Words": total_doc_words,
        "Total_Tokens": total_doc_tokens,
        "Fertility_Score": document_fertility_score
    }

    # Ensure the output directory exists
    output_dir = r"output_tokenizer/results_telugu"
    os.makedirs(output_dir, exist_ok=True)  # Create directories if they don’t exist

    # Save results to JSON file
    json_filename = os.path.join(output_dir, "Galactica_tokenization_results.json")
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)

    print(json.dumps(results, indent=4))
    print(f"Results saved to {json_filename}")

# Example usage
if __name__ == "__main__":
    folder_path = r"data/Telugu"
    
    if os.path.isdir(folder_path):
        process_hocr_folder(folder_path)
    else:
        print(f"Error: The folder path '{folder_path}' does not exist. Please check the path and try again.")
