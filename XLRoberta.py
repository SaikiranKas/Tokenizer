import os
import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

def process_hocr_folder(folder_path):
    # Load the XLM-Roberta tokenizer
    tokenizer_name = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Dictionary to store results
    results = {tokenizer_name: {}}

    # Variables to track total words and tokens for entire document
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
                encoded_input = tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

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

    # Save results to JSON file
    json_filename = os.path.join(r"output_tokenizer\results_telugu", "Kosmostokenization_results.json")
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)

    print(json.dumps(results, indent=4))
    print(f"Results saved to {json_filename}")

# Example usage
if __name__ == "__main__":
    folder_path = r"C:\Users\Saikiran Kasturi\OneDrive\Desktop\Tokenizers\data\Telugu"
    if os.path.isdir(folder_path):
        process_hocr_folder(folder_path)
    else:
        print("Invalid folder path. Please try again.")
