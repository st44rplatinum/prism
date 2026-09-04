import torch
import esm2_model
import pandas as pd

with torch.no_grad():
    out = esm2_model.model(esm2_model.masked_tokens, repr_layers=[], return_contacts=False)
    logits = out["logits"][0]

    print(logits.shape) # (L, V)

    logits

pos = 0 
raw_logits = logits[pos + 1] # +1 to skip the start token
predicted_probablity = torch.softmax(raw_logits, dim=-1)
print(torch.round(predicted_probablity, decimals=2))

for aa in esm2_model.alphabet.standard_toks:
    print(f" {aa}: {predicted_probablity[esm2_model.tok_to_idx[aa]]*100:.1f}")


# compute log probability score
pos = 0
wt_AA = "M"
mut_AA = "A"

raw_logits = logits[pos + 1] # +1 to skip the start token
predicted_probablity = torch.softmax(raw_logits, dim=-1)

wt_prob = predicted_probablity[esm2_model.tok_to_idx[wt_AA]]
mut_prob = predicted_probablity[esm2_model.tok_to_idx[mut_AA]]

log_prob_score = torch.log(mut_prob / wt_prob).item()
print(f"Log probability score for mutation {wt_AA} -> {mut_AA} at position {pos}: {log_prob_score:.2f}")

# compute log probability score for the GRB2 dataset

ind = 1995
sequence = pd.DataFrame["sequence"].loc[ind]
variant = pd.DataFrame["variant"].loc[ind]

mutations = variant.split(",")

wt_AAs, mut_AAs, mut_positions = [], [], []
for mut in mutations:
    wt_AAs.append(mut[0]),
    mut_AAs.append(mut[-1]),
    mut_positions.append(int(mut[1:-1]) - 1) # convert to 0-based index

_, _, tokens = esm2_model.batch_converter([("protein", sequence)])
tokens = tokens.to(esm2_model.device)

log_prob_score = 0
for wt_AA, mut_AA, pos in zip(wt_AAs, mut_AAs, mut_positions):
    raw_logits = logits[pos + 1] # +1 to skip the start token
    predicted_probablity = torch.softmax(raw_logits, dim=-1)

    wt_prob = predicted_probablity[esm2_model.tok_to_idx[wt_AA]]
    mut_prob = predicted_probablity[esm2_model.tok_to_idx[mut_AA]]

    log_prob_score += torch.log(mut_prob / wt_prob).item()
print(f"Total log probability score for all mutations: {log_prob_score:.2f}")