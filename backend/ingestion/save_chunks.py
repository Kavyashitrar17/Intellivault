import json

def save_chunks(chunks, output_file="chunks.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

