from collections import Counter
import sys
from datasets import load_dataset

# Helper function to flatten list of lists
def flatten(list_of_lists):
    for sub in list_of_lists:
        for item in sub:
            yield item


# 1. Average tokens per word and per sentence
def avg_tokens_per_word(sentences, token_lists):
    total_tokens = sum(len(toks) for toks in token_lists)
    total_words = sum(len(sent.split()) for sent in sentences)

    avg_word = total_tokens / total_words
    avg_sentence = total_tokens / len(sentences)

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
def word_coverage(sentences, token_lists):
    uncovered_counts = []

    for sent, tokens in zip(sentences, token_lists):
        words = sent.split()
        joined = "".join(tok.lstrip("Ġ") for tok in tokens)

        uncovered = [w for w in words if w not in joined]
        uncovered_counts.append(len(uncovered))

    avg_uncovered = sum(uncovered_counts) / len(sentences)
    return avg_uncovered


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

