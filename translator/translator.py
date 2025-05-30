import os
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

def translate_text(text, tokenizer, model, src_lang="en", tgt_lang="fr"):
    """Traduire un texte avec M2M100 (ex: anglais vers français)."""
    if not text.strip():
        return ""

    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_length=512
    )
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text


def charge_model():
    print("Chargement du modèle de traduction (M2M100 en → fr)...")
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    print("Modèle chargé.")
    return tokenizer, model

def search_srt_files(filename):
    input_path = os.path.join(os.path.dirname(__file__), "..", "enhanced_srt_files", filename)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "translated_srt_files")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    safe_base = safe_filename(base_name)
    output_path = os.path.join(output_dir, f"{safe_base}_fr.srt")
    
    return input_path, output_path

def translate_srt_file(filename):
    tokenizer, model = charge_model()

    result = search_srt_files(filename)
    if not result:
        return
    input_path, output_path = result

    if not input_path or not output_path or not tokenizer or not model:
        return
    
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    translated_lines = []
    buffer = []

    for line in lines:
        if line.strip() == "":
            if len(buffer) >= 3:
                index = buffer[0]
                timestamp = buffer[1]
                text_lines = buffer[2:]
                full_text = " ".join(text_lines)

                translated_text = translate_text(full_text, tokenizer, model)

                new_block = [index, timestamp] + [translated_text, ""]
                translated_lines.extend(new_block)
            else:
                translated_lines.extend(buffer + [""])
            buffer = []
        else:
            buffer.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))

    print(f"Fichier traduit enregistré : {output_path}")

