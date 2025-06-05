import os
import pandas as pd
import json
import glob

def load_dataset(base_path, dataset):
    file_path = os.path.join(base_path, f"./data/{dataset}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return None

def load_annotation_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize: if data is a dict of dicts, convert it to a list of dicts
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            data = list(data.values())
        else:
            raise ValueError(f"Unexpected dict structure in {file_path}")

    elif not isinstance(data, list):
        raise ValueError(f"Unexpected data format in {file_path}: {type(data)}")

    return data

def extract_bias_words(annotation_data, bias_name):
    """Extract words from bias spans when label is 1"""
    words = []
    
    if bias_name in annotation_data and isinstance(annotation_data[bias_name], dict):
        bias_data = annotation_data[bias_name]
        
        # Check if the bias is labeled as present (label=1)
        if bias_data.get("label") == 1 and "spans" in bias_data:
            # Extract text from each span
            for span in bias_data["spans"]:
                if "text" in span:
                    words.append(span["text"].strip())
    
    return words

def normalize_text_for_comparison(text):
    """Normalize text by removing all quotes and extra whitespace for comparison"""
    if not isinstance(text, str):
        return str(text).strip()
    
    # Remove all types of quotes and strip whitespace
    normalized = text.strip()
    
    # Remove all quote characters: ", ", ', "
    quote_chars = ['"', '"', '"', "'", "“", "”"]
    for quote in quote_chars:
        normalized = normalized.replace(quote, '')
    
    return normalized.strip()

def merge_words_with_substring_check(existing_words, new_words):
    """Merge word lists, keeping only the longest text when substring relationships exist"""
    # Create initial word dictionary from existing words
    word_dict = {}
    
    # Add existing words to dictionary
    if existing_words:
        for existing_word in existing_words:
            existing_word = existing_word.strip()
            normalized = normalize_text_for_comparison(existing_word)
            if normalized:
                word_dict[normalized] = (existing_word, 1)
    
    # Process new words
    for word in new_words:
        word = word.strip()
        normalized_new = normalize_text_for_comparison(word)
        
        if not normalized_new:  # Skip empty strings
            continue
        
        # Check if this word has substring relationship with existing words
        found_match = False
        keys_to_remove = []
        
        for existing_normalized, (existing_word, count) in list(word_dict.items()):
            # Check if one is a substring of the other
            if normalized_new in existing_normalized:
                # New word is substring of existing - keep existing, increment count
                word_dict[existing_normalized] = (existing_word, count + 1)
                found_match = True
                break
            elif existing_normalized in normalized_new:
                # Existing word is substring of new - replace with new (longer)
                keys_to_remove.append(existing_normalized)
                word_dict[normalized_new] = (word, count + 1)
                found_match = True
                break
        
        # Remove the shorter entries that were replaced
        for key in keys_to_remove:
            del word_dict[key]
        
        # If no substring relationship found, add as new entry
        if not found_match:
            word_dict[normalized_new] = (word, 1)
    
    # Convert back to list of words (without counts)
    return [word_info[0] for word_info in word_dict.values()]

base_path = os.getcwd()
data = "MC_merged"  # Changed from MC_train to MC_merged as per instructions

# Load the main dataset
df = load_dataset(base_path, data)
if df is None:
    print(f"Could not load dataset {data}")
    exit(1)

# Define bias types to process
bias_types = ["loadedLanguage", "appealToFear", "nameCalling"]

# Initialize columns for each bias type in the dataframe
for bias in bias_types:
    col_name = f"word_{bias}"
    df[col_name] = None

# Find all annotation files across languages and rounds
annotation_files = []
languages = ["SPA", "ITA", "PT"]  # Updated languages as specified
for language in languages:
    for round_num in range(1, 5):  # Rounds 1-5 as specified
        # Use the correct path format based on the example provided
        pattern = f"/home/michele/Documenti/HYBRIDS/Experiment/Dataset_creation_MC/ANNOTATION/{language}/ROUND_{round_num}/DEF/r{round_num}_*.json"
        files = glob.glob(pattern)
        annotation_files.extend(files)

print(f"Found {len(annotation_files)} annotation files")

# Process each annotation file
for file_path in annotation_files:
    print(f"Processing {file_path}")
    
    # Extract language, round number, and annotator name from file path
    file_parts = os.path.basename(file_path).split('_')
    if len(file_parts) < 2:
        print(f"Skipping file with unexpected name format: {file_path}")
        continue
    
    annotator_name = file_parts[-1].split('.')[0]
    
    # Load the annotation data
    annotations = load_annotation_file(file_path)
    
    # Match annotations with MC_merged dataset and extract bias words
    for annotation in annotations:
        if "id" not in annotation:
            continue
            
        article_id = annotation["id"]
        
        # Find the corresponding row in the dataframe
        matches = df[df['id'] == article_id]
        
        if not matches.empty:
            row_idx = matches.index[0]
            
            # Process each bias type
            for bias in bias_types:
                words = extract_bias_words(annotation, bias)
                if words:
                    # Column name format: word_biasName
                    col_name = f"word_{bias}"
                    
                    # Get current value
                    current_value = df.at[row_idx, col_name]
                    
                    # Prepare existing words list
                    existing_words = []
                    if isinstance(current_value, list):
                        existing_words = current_value
                    
                    # Merge with new words using substring check
                    updated_words = merge_words_with_substring_check(existing_words, words)
                    
                    # Update the dataframe
                    df.at[row_idx, col_name] = updated_words

# Save the updated dataframe
output_path = os.path.join(base_path, f"./data/{data}_with_bias_words.csv")
df.to_csv(output_path, index=False)
print(f"Updated dataset saved to {output_path}")

# Let's also update the prompt generation code to use the new bias word columns
def generate_prompts(df):
    prompt_template = []

    for i, row in df.iterrows():
        texts = []

        # Hyperpartisan
        if row['hyperpartisan_majority'] == 1:
            texts.append("This news headline is Hyperpartisan.")
        else:
            texts.append("This news headline is Neutral.")

        # Great Replacement Theory
        if row['prct_majority'] == 1:
            texts.append("The headline contains Great Replacement Theory information.")
        else:
            texts.append("The headline does not contain Great Replacement Theory information.")

        # Stance
        stance = row['stance_majority']
        if pd.notna(stance):
            texts.append(f"The headline stance is {stance} immigration.")

        # Loaded Language
        if row['loadedLanguage_majority'] == 1:
            words_LL = row.get('word_loadedLanguage', [])
            if isinstance(words_LL, list) and words_LL:
                # Remove duplicates based on normalized text for display
                unique_words = []
                seen_normalized = set()
                for word in words_LL:
                    normalized = normalize_text_for_comparison(word)
                    if normalized and normalized not in seen_normalized:
                        unique_words.append(word)
                        seen_normalized.add(normalized)
                
                if unique_words:
                    word_list = ', '.join(f'"{word}"' for word in unique_words)
                    texts.append(f'The headline used loaded language: {word_list}.')
                else:
                    texts.append("The headline used loaded language.")
            else:
                texts.append("The headline used loaded language.")
        else:
            texts.append("The headline didn't use loaded language.")

        # Appeal to Fear
        if row['appealToFear_majority'] == 1:
            words_AF = row.get('word_appealToFear', [])
            if isinstance(words_AF, list) and words_AF:
                # Remove duplicates based on normalized text for display
                unique_words = []
                seen_normalized = set()
                for word in words_AF:
                    normalized = normalize_text_for_comparison(word)
                    if normalized and normalized not in seen_normalized:
                        unique_words.append(word)
                        seen_normalized.add(normalized)
                
                if unique_words:
                    word_list = ', '.join(f'"{word}"' for word in unique_words)
                    texts.append(f'The headline used appeal to fear: {word_list}.')
                else:
                    texts.append("The headline used appeal to fear.")
            else:
                texts.append("The headline used appeal to fear.")
        else:
            texts.append("The headline didn't use appeal to fear.")

        # Name Calling
        if row['nameCalling_majority'] == 1:
            words_NC = row.get('word_nameCalling', [])
            if isinstance(words_NC, list) and words_NC:
                # Remove duplicates based on normalized text for display
                unique_words = []
                seen_normalized = set()
                for word in words_NC:
                    normalized = normalize_text_for_comparison(word)
                    if normalized and normalized not in seen_normalized:
                        unique_words.append(word)
                        seen_normalized.add(normalized)
                
                if unique_words:
                    word_list = ', '.join(f'"{word}"' for word in unique_words)
                    texts.append(f'The headline used name calling: {word_list}.')
                else:
                    texts.append("The headline used name calling.")
            else:
                texts.append("The headline didn't use name calling.")

        finalized_prompt = " ".join(texts)
        prompt_template.append(finalized_prompt)
    
    return prompt_template

# Generate prompts with the updated dataframe
prompts = generate_prompts(df)

df['prompts'] = prompts
df.to_csv('prompts.csv')