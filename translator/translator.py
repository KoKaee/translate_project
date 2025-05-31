import os
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Disable parallelism warning and set framework preferences
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

# Replace illegal characters in filenames with underscores
def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

# Translate text using M2M100 model from one language to another
def translate_text(text, tokenizer, model, src_lang="en", tgt_lang="fr"):
    if not text.strip():
        return ""  # Skip empty or whitespace-only strings

    tokenizer.src_lang = src_lang  # Set source language
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Generate translation
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_length=512
    )

    # Decode the result and remove special tokens
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text

# Load the translation model and tokenizer
def charge_model():
    print("loeding the translation model (M2M100 en → fr)...")
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    print("Model loaded.")
    return tokenizer, model

# Locate input and output file paths for translation
def search_srt_files(filename):
    # Build full path to the input subtitle file
    input_path = os.path.join(os.path.dirname(__file__), "..", "enhanced_srt_files", filename)

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "..", "translated_srt_files")
    os.makedirs(output_dir, exist_ok=True)

    # Generate safe output file name
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    safe_base = safe_filename(base_name)
    output_path = os.path.join(output_dir, f"{safe_base}_fr.srt")

    return input_path, output_path

# Process and translate an entire .srt subtitle file
def translate_srt_file(filename):
    tokenizer, model = charge_model()

    result = search_srt_files(filename)
    if not result:
        return
    input_path, output_path = result

    if not input_path or not output_path or not tokenizer or not model:
        return

    # Read the subtitle file and split into lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    translated_lines = []
    buffer = []

    # Loop through lines and identify subtitle blocks
    for line in lines:
        if line.strip() == "":
            if len(buffer) >= 3:
                index = buffer[0]  # Subtitle number
                timestamp = buffer[1]  # Time range
                text_lines = buffer[2:]  # Actual text lines
                full_text = " ".join(text_lines)  # Merge text lines for translation

                # Translate and build new subtitle block
                translated_text = translate_text(full_text, tokenizer, model)
                new_block = [index, timestamp] + [translated_text, ""]
                translated_lines.extend(new_block)
            else:
                # Preserve improperly formatted blocks
                translated_lines.extend(buffer + [""])
            buffer = []
        else:
            buffer.append(line)

    # Write translated content to new .srt file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))

    print(f"file translated into : {output_path}")
