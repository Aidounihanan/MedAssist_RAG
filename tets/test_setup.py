print("=== Test setup MedAssist RAG ===\n")

# Test 1 : LangChain
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✓ LangChain OK")
except Exception as e:
    print(f"✗ LangChain : {e}")

# Test 2 : ChromaDB
try:
    import chromadb
    print("✓ ChromaDB OK")
except Exception as e:
    print(f"✗ ChromaDB : {e}")

# Test 3 : Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode("test médical")
    print(f"✓ Embeddings OK — dimension : {len(vec)}")
except Exception as e:
    print(f"✗ Sentence Transformers : {e}")

# Test 4 : PDF
try:
    from pypdf import PdfReader
    print("✓ pypdf OK")
except Exception as e:
    print(f"✗ pypdf : {e}")

# Test 5 : pandas
try:
    import pandas as pd
    print("✓ pandas OK")
except Exception as e:
    print(f"✗ pandas : {e}")

print("\n=== Setup terminé ===")