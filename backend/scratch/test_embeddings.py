from langchain_huggingface import HuggingFaceEmbeddings
print("Loading model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Model loaded.")
text = "This is a test."
vec = embeddings.embed_query(text)
print(f"Vector length: {len(vec)}")
print("Success!")
