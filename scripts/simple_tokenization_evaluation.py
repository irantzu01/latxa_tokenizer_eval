from collections import Counter
import sys
from datasets import load_dataset

# Helper function to flatten list of lists
def flatten(list_of_lists):
    for sub in list_of_lists:
        for item in sub:
            yield item


# 1. Average tokens per word and per sentence
def avg_tokens_per_word(raw_examples, token_lists):
    total_tokens = sum(len(toks) for toks in token_lists)
    total_words = sum(len(ex["text"].split()) for ex in raw_examples)

    avg_word = total_tokens / total_words
    avg_sentence = total_tokens / len(raw_examples)

    return avg_word, avg_sentence

# 2. Token length distribution
def token_length_distribution(token_lists):
    token_lengths = [len(tok) for tok in flatten(token_lists)]
    return Counter(token_lengths)

def plot_token_length_distribution(latxa_tokens, dyn_tokens):
    latxa_lengths = [len(tok) for tok in flatten(latxa_tokens)]
    dyn_lengths = [len(tok) for tok in flatten(dyn_tokens)]

    max_length = max(max(latxa_lengths), max(dyn_lengths))
    bins = list(range(1, max_length + 1))

    latxa_freq = [latxa_lengths.count(i) for i in bins]
    dyn_freq = [dyn_lengths.count(i) for i in bins]

    return bins, latxa_freq, dyn_freq


# 3. Overlap of token sets
def token_set_overlap(token_lists1, token_lists2):
    set1 = set(flatten(token_lists1))
    set2 = set(flatten(token_lists2))

    intersection = set1 & set2
    union = set1 | set2

    jaccard_index = len(intersection) / len(union) if union else 0
    return jaccard_index, intersection, union


# 4. word coverage
# convert HF BatchEncoding list -> list[list[token_str]]
def prepare_latxa_token_strings(encoded_latxa, tokenizer):
    """
    encoded_latxa: list of BatchEncoding dicts (each has 'input_ids' possibly nested)
    tokenizer: HF tokenizer object (to convert ids->tokens)
    returns: list of token-string lists, one per example
    """
    latxa_tokens = []
    for enc in encoded_latxa:
        # enc may be a BatchEncoding or dict. Try to extract input_ids
        if isinstance(enc, dict) and "input_ids" in enc:
            ids = enc["input_ids"]
            # sometimes input_ids is nested list (e.g., pair encoding), handle that
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            latxa_tokens.append(tokenizer.convert_ids_to_tokens(ids))
        else:
            # if it's already a list of token strings, keep as-is
            latxa_tokens.append(enc)
    return latxa_tokens


def word_coverage_from_token_strings(raw_examples, token_lists, token_type="token"):
    """
    raw_examples: list of dicts with "text" key (your raw_examples)
    token_lists: list of token-lists (strings) aligned with raw_examples
    token_type: just a label for returns
    returns: dict {total, single, coverage_pct}
    """
    total = 0
    single = 0

    for ex, toks in zip(raw_examples, token_lists):
        sent = ex["text"]
        words = sent.split()
        # For speed: build tokenization of each word by tokenizing the word itself using the token list alignment
        # Simpler approach: for each word, ask tokenizer to encode the single word,
        # but since we already have token lists per sentence, we use a heuristic:
        # count single-token words by checking if that word appears as a single token when tokenizing the word
        # (we will instead re-tokenize the word using the same tokenizer if needed; but here token_lists are strings)
        for w in words:
            # Naive but accurate approach: re-tokenize the word using the sentence's tokenization tokenizer is unknown.
            # So we conservatively check whether the word equals the token after stripping common markers.
            # First try to find tokens in the sentence that match the word when stripping markers.
            # If any token equals the whole word, count as single-token word.
            found_single = False
            for tok in toks:
                # normalize token to compare to word
                tnorm = tok.lstrip("Ġ▁")  # common markers
                if tnorm == w:
                    found_single = True
                    break
            total += 1
            if found_single:
                single += 1

    return {"total": total, "single": single, "coverage_pct": single / total if total else 0.0}


# 5. Most common tokens
def most_common_subwords(token_lists, n=50):
    return Counter(flatten(token_lists)).most_common(n)


# 6. Tokenization consistency
families = {
    "etxe (house)": ["etxe", "etxea", "etxeak", "etxeko", "etxeetako", "etxearen",
                    "etxeen", "etxean","etxeetan", "etxetik", "etxeetatik", "etxera",
                    "etxeetara", "etxerantz", "etxeetarantz"],
    "eskola (school)": ["eskola", "eskolak", "eskolako", "eskoletako", "eskolaren",
                       "eskolen", "eskolan", "eskoletan", "eskolatik", "eskoletatik",
                       "eskolara", "eskoletara", "eskolarantz", "eskoletarantz"],
    "neska (girl)": ["neska", "neskak", "neskek", "neskaren", "nesken", "neskari",
                     "neskei", "neskarekin", "neskekin", "neskarengana", "neskengana"],
    "ume (kid)": ["ume", "umea", "umeak", "umeek", "umearen", "umeen", "umeari", 
                  "umeei", "umearekin", "umeekin", "umearengana", "umeengana"],
    "osasun (health)": ["osasun-zerbitzu", "osasun-sistema", 
                        "osasun-politika", "osasun-langile", "osasun-zentro", 
                        "osasun-etxe", "osasun-krisi", "osasun-arazo"]
}
def analyze_family(words, tokenizer):
    results = {}
    for w in words:
        results[w] = tokenizer.tokenize(w)
    return results


# 7. Morphscore evaluation
# sys.path.append("/home/irantzu/MASTER/WiSe25/Lab Rotation/morphscore")

# ds = load_dataset("catherinearnett/morphscore")['train']
# df = ds.filter(lambda x: x["language"] == "eus_latn").to_pandas()

# from morphscore import MorphScore
# morph_score = MorphScore(
#     data_dir="../data/",
#     language_subset=['eus_latn'],
#     by_split=False,
#     freq_scale=True,
#     exclude_single_tok=False
# )

# result, df = morph_score.eval(tokenizer, return_df=True)

# metrics = result['eus_latn']
# for key, value in metrics.items():
#     print(f"{key}: {value}")

