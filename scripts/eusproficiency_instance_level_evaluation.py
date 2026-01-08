# Extract instances where Latxa and BPE performed differently
import json
import sys

id_list = set()
with open("cache/eusproficiency_dynamic_eval_results.jsonl") as fbpe, \
     open("cache/eusproficiency_latxa_eval_results.jsonl") as flatxa, \
     for line in fbpe, flatxa:
        bpe_result = json.loads(fbpe.readline())
        latxa_result = json.loads(flatxa.readline())
        if bpe_result["correct"] != latxa_result["correct"]:
            id_diff = bpe_result["id"]
            id_list.add(id_diff)

with open("cache/eusproficiency_dynamic_fewshot_tokenized.jsonl") as fbpe_tok, \
     open("cache/eusproficiency_latxa_fewshot_tokenized.jsonl") as flatxa_tok, \
     open("eusproficiency_different_results.jsonl", "w") as fout:
     for line in fbpe_tok, flatxa_tok:
        bpe_entry = json.loads(fbpe_tok.readline())
        latxa_entry = json.loads(flatxa_tok.readline())
        if bpe_entry["id"] in id_list:
            combined_entry = {
                "id": bpe_entry["id"],
                "bpe": bpe_entry,
                "latxa": latxa_entry
            }
            fout.write(json.dumps(combined_entry) + "\n")
            
    
        