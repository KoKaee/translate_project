from translator.subtitle_splitter import enhance
from translator.translator import translate_srt_file
from translator.subtitle_enhancer_ai import enhance_using_AI
import os


def main():
    filename = input("Enter the name of the srt file :").strip()
    input_path = os.path.join(os.path.dirname(__file__), "srt_files", filename)

    print("Nettoyage des fichiers SRT...")
    enhance(input_path)
    enhance_using_AI(filename)

    print("Traduction des fichiers SRT...")
    translate_srt_file(filename)
    

if __name__ == "__main__":
    main()