"""Tokenize FineWeb-Edu (sample-10BT parquet shards) into token files (uint16, or uint32 for vocabularies > 65536) for
Phase-3 LM training. The dtype is recorded in meta.json and read back by token_dtype().

  python prep_data.py --out ~/data/fwe --shards 3 --train_tokens 2_000_000_000 --val_tokens 20_000_000 --workers 8

Downloads whole parquet shards with huggingface_hub (much faster than HF streaming), reads them with pyarrow,
tokenizes with GPT-2 BPE (MIT licence, ungated) in a process pool, joins documents with EOS (50256).
Val = the first documents of shard 0, train = everything after (disjoint). Output: train.bin, val.bin, meta.json.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

EOS = 50256
TOKENIZER = "openai-community/gpt2"
_tok = None


def _init(name=None, eos=None):
    global _tok, EOS
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(name or TOKENIZER)
    _tok.model_max_length = 10 ** 9
    if eos is not None:
        EOS = eos


def _encode(texts):
    return [ids + [EOS] for ids in _tok(texts, add_special_tokens=False)["input_ids"]]


def text_batches(paths, n=512):
    import pyarrow.parquet as pq
    for path in paths:
        f = pq.ParquetFile(path)
        for rb in f.iter_batches(batch_size=n, columns=["text"]):
            yield rb.column(0).to_pylist()


DTYPE = np.uint16


def write(path, docs_iter, budget):
    arr = np.memmap(path, dtype=DTYPE, mode="w+", shape=(budget,))
    pos, report = 0, 0
    for docs in docs_iter:
        for ids in docs:
            n = min(len(ids), budget - pos)
            arr[pos:pos + n] = ids[:n]
            pos += n
            if pos >= budget:
                arr.flush()
                return pos
        if pos >= report:
            print(f"  {os.path.basename(path)}: {pos / 1e6:.0f}M tokens", flush=True)
            report += 100_000_000
    arr.flush()
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--train_tokens", type=int, default=2_000_000_000)
    ap.add_argument("--val_tokens", type=int, default=20_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tokenizer", default="openai-community/gpt2", help="e.g. LiquidAI/LFM2-350M, Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(a.tokenizer)
    global DTYPE
    DTYPE = np.uint16 if len(t) <= 65536 else np.uint32
    eos = t.eos_token_id if a.tokenizer != "openai-community/gpt2" else EOS
    os.makedirs(a.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download("HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet", repo_type="dataset")
             for i in range(a.shards)]
    print("shards:", paths, flush=True)
    pool = Pool(a.workers, initializer=_init, initargs=(a.tokenizer, eos))
    docs = pool.imap(_encode, text_batches(paths), chunksize=1)
    n_val = write(os.path.join(a.out, "val.bin"), docs, a.val_tokens)
    n_train = write(os.path.join(a.out, "train.bin"), docs, a.train_tokens)
    if n_train < a.train_tokens:  # shrink the file to what was actually written
        with open(os.path.join(a.out, "train.bin"), "r+b") as fh:
            fh.truncate(n_train * np.dtype(DTYPE).itemsize)
    json.dump({"tokenizer": a.tokenizer, "eos": eos, "dtype": np.dtype(DTYPE).name, "val_tokens": n_val, "train_tokens": n_train,
               "dataset": "HuggingFaceFW/fineweb-edu:sample-10BT", "shards": a.shards},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("done", n_val, n_train, flush=True)
    # no pool.terminate()/join: with unconsumed imap results both hung on the 8x B200 node (`with Pool` exit, then
    # terminate()). Everything is on disk, so kill the workers and leave without waiting on pool threads.
    for w in pool._pool:
        w.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
