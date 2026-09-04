import main
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch    

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t30_150M_UR50D", device_map="auto")


model_name: str = "esm2_t30_150M_UR50D"
model, alphabet = getattr(esm.pretrained, model_name)()
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_converter = alphabet.get_batch_converter()

tok_to_idx = alphabet.tok_to_idx
idx_to_tok = {idx: tok for tok, idx in tok_to_idx.items()}
idx_to_tok 

#masking positions
"""for proteins, each amino acid is a token, 
   so we can mask a specific amino acid by its position in the sequence
   The positions are 0-based, meaning the first amino acid is at position 0, 
   the second at position 1, and so on.
"""
sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGAG"

data = [("protein1", sequence)]
_, _, tokens = batch_converter(data)
tokens = tokens.to(device)

print(f"Sequence: {sequence}")
print(f"Tokens: {tokens}")

print(f"Masking token ID: {alphabet.mask_idx}")
print(f"CLS token index: {alphabet.cls_idx}")
print(f"EOS token index: {alphabet.eos_idx}\n")

masked_positions_0based = 0

masked_tokens = tokens.clone()
masked_tokens[0, masked_positions_0based] = alphabet.mask_idx
print(tokens[0])
print(masked_tokens[0])

