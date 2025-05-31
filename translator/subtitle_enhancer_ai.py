import os
import re
from llama_cpp import Llama

# Path to the quantized Mistral model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Initialize the LLaMA model with specific context size and threading
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,       # Context length (number of tokens that can be handled)
    n_threads=4       # Number of CPU threads to use
)

# Loads subtitle blocks from an SRT file and parses them into a structured list
def load_srt_blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = []
    for block in content.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            index = lines[0]
            timestamp = lines[1]
            text = " ".join(lines[2:]).strip() if len(lines) > 2 else ""
            blocks.append((index, timestamp, text))
    return blocks

# Prepares the subtitle text with markers like <S1>, <S2> to preserve block alignment during LLM editing
def annotate_blocks(blocks):
    annotated_text = ""
    for i, (_, _, text) in enumerate(blocks, 1):
        if not text:
            # Keep empty subtitle blocks for re-alignment
            annotated_text += f"<S{i}>\n"
        else:
            annotated_text += f"<S{i}> {text.strip()}\n"
    return annotated_text.strip()

# Sends the annotated subtitle content to the LLM and returns the improved version
def call_llm_for_improvement(annotated_text):
    prompt = f"""### Instruction:
You are improving subtitle text. Merge broken sentences, add proper punctuation.
Don't add new words or change the meaning.
Keep the <S1>, <S2>, ... markers unchanged so we can split the text after correction.
Do not add any new markers, only use the existing ones.
Split multiple independent sentences into separate lines within each tag, do not merge them into a single line.
Do not merge short interjections like "wow", "um", or "bou" with other lines — keep them isolated.
Do not alter or remove empty blocks — preserve them.

### Input:
{annotated_text}

### Response:"""

    # Call the model with a token limit and stop sequence
    response = llm(prompt, max_tokens=2048, stop=["###"])
    return response["choices"][0]["text"].strip()

# Extract the cleaned-up subtitle lines paired with their respective markers (e.g. S1, S2, ...)
def extract_sentences_with_markers(improved_text):
    pattern = r"<S(\d+)>([^<]*)"
    matches = re.findall(pattern, improved_text)
    numbered_sentences = {int(num): sentence.strip() for num, sentence in matches}
    return numbered_sentences

# Rebuild the SRT structure with improved text, keeping original index and timestamp
def rebuild_srt(blocks, improved_sentences):
    output_lines = []
    for i, (index, timestamp, _) in enumerate(blocks, 1):
        text = improved_sentences.get(i, "")
        output_lines.extend([index, timestamp, text, ""])
    return "\n".join(output_lines)

# The main function to apply AI enhancement to an SRT file using the local LLM
def enhance_using_AI(filename):    
    input_path = os.path.join(os.path.dirname(__file__), "..", "enhanced_srt_files", filename)
    blocks = load_srt_blocks(input_path)
    annotated_text = annotate_blocks(blocks)
    improved_text = call_llm_for_improvement(annotated_text)
    improved_sentences = extract_sentences_with_markers(improved_text)
    final_srt = rebuild_srt(blocks, improved_sentences)
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(final_srt)
