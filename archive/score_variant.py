import torch
import predict_prob
import esm2_model
import numpy as np

def score_variant(sequence, variant):
    log_prob_score = 0.0

    for m in variant.split(","):
        wt_AA = m[0]
        mut_AA = m[-1]
        pos0 = int(m[1:-1]) # convert to 0-based index

        _, _, tokens = esm2_model.batch_converter([("protein", sequence)])
        masked_tokens = tokens.to(device=esm2_model.device).clone()
        masked_tokens[0, pos0 + 1] = esm2_model.alphabet.mask_idx # +1 to skip the start token

        with torch.no_grad():
            logits = esm2_model.model(masked_tokens, repr_layers=[], return_contacts=False)["logits"][0]

        probs = torch.softmax(logits, dim=-1)
        log_prob_score += torch.log(probs[esm2_model.tok_to_idx[mut_AA]] / probs[esm2_model.tok_to_idx[wt_AA]]).item()

    return log_prob_score

sampled["ESM2_score"] = np.nan

for ind in tqdm(range(len(sampled)), desc="Scoring variants"):
    sampled.loc[ind, "ESM2_score"] = score_variant(
        sampled.loc[ind, "sequence"], sampled.loc[ind, "variant"])

# add visualization later

