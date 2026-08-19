import sys
import torch
import transformers
import sentence_transformers

from sentence_transformers import SentenceTransformer


print("=== Environment ===")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Sentence-Transformers:", sentence_transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")


print("\n=== Loading Model ===")

model_name = "sentence-transformers/bert-base-nli-mean-tokens"

model = SentenceTransformer(model_name)

print("Model loaded successfully:", model_name)


print("\n=== Generating Embeddings ===")

sentences = [
    "A man is playing a guitar.",
    "A person is performing music on a guitar.",
    "A woman is driving a car.",
    "The company announced its annual profit."
]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)


print("\n=== Similarity Test ===")

similarities = model.similarity(embeddings, embeddings)

similar_pair = similarities[0][1].item()
unrelated_pair = similarities[0][3].item()

print("Similar pair score:", similar_pair)
print("Unrelated pair score:", unrelated_pair)

assert similar_pair > unrelated_pair

print("\nFEASIBILITY TEST PASSED")