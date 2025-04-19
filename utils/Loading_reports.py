import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.extract_text import extract_text
from utils.embed_store import store_embeddings


def process_and_store(file_path):
    text = extract_text(file_path)
    store_embeddings(text)
    return text