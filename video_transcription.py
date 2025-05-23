import os
import glob
import re
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

def translate_text(text, translator):
    """Translate text from English to French."""
    if not text.strip():
        return ""
    translated = translator(text, max_length=512)
    return translated[0]["translation_text"]

def translate_srt_file(input_path, output_path, translator):
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
                translated_text = translate_text(full_text, translator)

                new_block = [index, timestamp] + [translated_text, ""]
                translated_lines.extend(new_block)
            else:
                translated_lines.extend(buffer + [""])
            buffer = []
        else:
            buffer.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))

    print(f"✅ Fichier traduit enregistré : {output_path}")

def main():
    print("🚀 Chargement du modèle de traduction (en → fr)...")
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    print("✅ Modèle chargé.")

    folder = os.path.join(os.path.dirname(__file__), "enhanced_srt_files")
    if not os.path.isdir(folder):
        print(f"❌ Dossier introuvable : {folder}")
        return

    srt_files = glob.glob(os.path.join(folder, "*.srt"))
    if not srt_files:
        print("❌ Aucun fichier SRT trouvé.")
        return

    print(f"✅ {len(srt_files)} fichier(s) trouvé(s) :")
    for i, file in enumerate(srt_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    try:
        choice = int(input("Quel fichier voulez-vous traduire ? (numéro) ").strip()) - 1
        if not (0 <= choice < len(srt_files)):
            raise ValueError
    except ValueError:
        print("❌ Choix invalide.")
        return

    input_srt = srt_files[choice]
    output_dir = os.path.join(os.path.dirname(__file__), "translated_srt_files")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_srt))[0]
    safe_base = safe_filename(base_name)
    output_path = os.path.join(output_dir, f"{safe_base}_fr.srt")

    print(f"📂 Fichier à traduire : {input_srt}")
    print(f"📄 Fichier traduit : {output_path}")
    print(f"📂 Dossier de sortie : {output_dir}")

    translate_srt_file(input_srt, output_path, translator)

if __name__ == "__main__":
    main()
