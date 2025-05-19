import os
import glob
import re
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
os.environ["USE_TORCH"] = "1"  # Force use of PyTorch
os.environ["USE_TF"] = "0"  # Disable TensorFlow

def translate_text(text, translator):
    """Translate text from English to Chinese."""
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
            # Traiter un bloc complet
            if len(buffer) >= 3:
                index = buffer[0]
                timestamp = buffer[1]
                text_lines = buffer[2:]
                full_text = " ".join(text_lines)
                translated_text = translate_text(full_text, translator)

                # Concaténer original + traduction
                new_block = [index, timestamp] + text_lines + [translated_text, ""]
                translated_lines.extend(new_block)
            else:
                translated_lines.extend(buffer + [""])
            buffer = []
        else:
            buffer.append(line)

    # Sauvegarder dans un nouveau fichier
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))

    print(f"✅ Fichier traduit enregistré : {output_path}")

def main():
    print("🚀 Chargement du modèle de traduction...")
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    print("✅ Modèle de traduction chargé.")
    print("🔍 Recherche de fichiers SRT dans le répertoire courant...")

    folder = os.path.join(os.path.dirname(__file__), "enhanced_srt_files")
    # Vérifie que le dossier existe
    if not os.path.isdir(folder):
        print(f"❌ Dossier introuvable : {folder}")
        exit()

    # Recherche les fichiers .srt
    srt_files = glob.glob(os.path.join(folder, "*.srt"))

    if not srt_files:
        print("❌ Aucun fichier SRT trouvé dans le dossier.")
        exit()

    print(f"✅ {len(srt_files)} fichier(s) SRT trouvé(s) :")
    
    for i, file in enumerate(srt_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    # Choisir un fichier
    choice = int(input("Quel fichier voulez-vous traduire ? (numéro) ").strip()) - 1

    if not (0 <= choice < len(srt_files)):
        print("❌ Numéro invalide.")
        exit()
        
    input_srt = srt_files[choice]

    # Dossier de sortie
    output_dir = os.path.join(os.path.dirname(__file__), "translated_srt_files")
    os.makedirs(output_dir, exist_ok=True)

    # Nom de fichier propre
    base_name = os.path.splitext(os.path.basename(input_srt))[0]
    safe_base = safe_filename(base_name)
    output_srt = os.path.join(output_dir, f"{safe_base}_fr.srt")
    output_srt = os.path.join(os.path.dirname(__file__),"translated_srt_files")

    print(f"📂 Fichier à traduire : {input_srt}")
    print(f"📄 Fichier traduit : {input_srt.replace(".srt", "_fr.srt")}")
    print(f"📂 Dossier de sortie : {output_srt}")

    translate_srt_file(input_srt, output_srt, translator)


if __name__ == "__main__":
    main()