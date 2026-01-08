# Extract instances where Latxa and BPE performed differently
import json
import sys

diff_ids = set()
with open("cache/eusproficiency_dynamic_eval_results.jsonl") as fbpe, \
     open("cache/eusproficiency_latxa_eval_results.jsonl") as flatxa:
     for bpe_line, latxa_line in zip(fbpe, flatxa):
        bpe = json.loads(bpe_line)
        latxa = json.loads(latxa_line)
        if bpe["correct"] != latxa["correct"]:
            id_diff = bpe["id"]
            diff_ids.add(id_diff)
print(f"Found {len(diff_ids)} differing instances")


with open("cache/eusproficiency_dynamic_fewshot_tokenized.jsonl") as fbpe_tok, \
     open("cache/eusproficiency_latxa_fewshot_tokenized.jsonl") as flatxa_tok, \
     open("eusproficiency_different_results.jsonl", "w") as fout:
     for bpe_line, latxa_line in zip(fbpe, flatxa):
        bpe_entry = json.loads(fbpe_tok.readline())
        latxa_entry = json.loads(flatxa_tok.readline())
        if bpe_entry["id"] in id_list:
            combined_entry = {
                "id": bpe_entry["id"],
                "bpe": bpe_entry,
                "latxa": latxa_entry
            }
            fout.write(json.dumps(combined_entry) + "\n")
            
    
        