# Evaluation functions
import torch


# Dynamic tokenization function
def dynamic_tokenize_texts(texts, dynamic_bpe, batch_size=128, max_merges=10):
    """
    texts: list[str]
    returns: list[list[str]]
    """
    all_tokens = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_examples = [{"text": t} for t in batch_texts]

        dyn_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
            batch_examples=batch_examples,
            max_nr_merges=max_merges,
            mlm=True
        )

        all_tokens.extend(dyn_tokens)

    return all_tokens

# dynamic tokens → text
def dynamic_tokens_to_text(dynamic_tokens):
    """  list[str] → str  """
    text = ""
    for tok in dynamic_tokens:
        if tok.startswith("▁") or tok.startswith("Ġ"):
            text += " " + tok[1:]
        else:
            text += tok
    return text.strip()
print("Running Dynamic BPE tokenization...")


# dynamic tokens → latxa IDs
def dynamic_tokens_to_latxa_ids(dynamic_tokens_batch, tokenizer):
    """
    list[list[str]] → list[list[int]]
    """
    texts = [
        dynamic_tokens_to_text(tokens)
        for tokens in dynamic_tokens_batch
    ]

    enc = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False
    )

    return enc["input_ids"]


# Batch builder
def build_batch_tensors(batch_ids, pad_id, device):
    max_len = max(len(seq) for seq in batch_ids)
    input_ids = torch.full(
        (len(batch_ids), max_len),
        pad_id,
        dtype=torch.long,
        device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batch_ids):
        seq = torch.tensor(seq, device=device)
        input_ids[i, :len(seq)] = seq
        attention_mask[i, :len(seq)] = 1

    return input_ids, attention_mask

# Scoring
@torch.no_grad()
def score_choices(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]

    scores = []
    for i in range(input_ids.size(0)):
        score = 0.0
        for t in range(targets.size(1)):
            if attention_mask[i, t + 1]:
                score += log_probs[i, t, targets[i, t]]
        scores.append(score)

    return torch.stack(scores)