from utils.chunking import chunk_text

MAX_CHARS = 500
OVERLAP = 100

# with open("data/books/the_great_gatsby.txt", "r", encoding="utf-8") as f:
#     text = f.read()

with open("data/books/why_sports_is_necessary.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("File length:", len(text))

chunks = chunk_text(text, max_chars=MAX_CHARS, overlap=OVERLAP)

print("Number of chunks generated:", len(chunks))

# Print all chunks completely
for i, c in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(c)   
    print()   
