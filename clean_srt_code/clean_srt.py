import os
import re

def clean_srt_preserve_timing(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    block = []


    for line in lines:
        if line.strip() == "":
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(line.rstrip('\n'))
    if block:
        blocks.append(block)

    cleaned_blocks = []
    previous_text = ""

    def clean_text_lines(text_lines):
        cleaned_lines = []
        for line in text_lines:
            line_clean = re.sub(r'\[.*?\]', '', line).strip()
            if line_clean:
                cleaned_lines.append(line_clean)
        return cleaned_lines

    for block in blocks:
        if len(block) < 3:
            continue

        time_line = block[1]
        text_lines = block[2:]

        cleaned_text_lines = clean_text_lines(text_lines)
        if not cleaned_text_lines:
            continue

        combined_text = " ".join(cleaned_text_lines).lower()

        if combined_text == previous_text:
            continue

        previous_text = combined_text
        cleaned_blocks.append((time_line, cleaned_text_lines))

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (time_line, texts) in enumerate(cleaned_blocks, 1):
            f.write(f"{i}\n")
            f.write(f"{time_line}\n")
            for text_line in texts:
                f.write(f"{text_line}\n")
            f.write("\n\n")

def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

def batch_clean_srts(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".srt"):
            input_path = os.path.join(input_folder, filename)
            base, _ = os.path.splitext(filename)
            safe_base = safe_filename(base)
            output_path = os.path.join(output_folder, f"{safe_base}.cleaned.srt")
            print(f"Cleaning: {filename}")
            try:
                clean_srt_preserve_timing(input_path, output_path)
            except Exception as e:
                print(f"❌ Error cleaning {filename}: {e}")


batch_clean_srts("srt_files","enhanced_srt_files" )
