"""Reference top-20 logprobs from the model's own implementation (Hugging Face transformers), for comparing
llama.cpp's arithmetic paths against (experiments/path_noise.py).

For every trace reply, one teacher-forced forward pass over prompt + reply in BF16 (gpt-oss: the MXFP4 experts
dequantized to BF16, so the weights are exactly the released ones); logprobs in float32 from the BF16 logits.

    python experiments/hf_reference.py traces.json.gz --model openai/gpt-oss-20b --out hf-bf16.json.gz
"""
import argparse
import gzip
import json

import torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("traces")
ap.add_argument("--model", required=True)
ap.add_argument("--variant", default="honest")
ap.add_argument("--out", required=True)
a = ap.parse_args()

kw = {}
if "gpt-oss" in a.model:
    from transformers import Mxfp4Config
    kw["quantization_config"] = Mxfp4Config(dequantize=True)
try:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, **kw).eval()
except (ValueError, KeyError):  # multimodal checkpoints (Gemma-4) load as image-text models; text input works the same
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, **kw).eval()
data = json.load(gzip.open(a.traces, "rt"))
out = []
with torch.no_grad():
    for rep in data["variants"][a.variant]["replies"]:
        ids = rep["prompt_ids"] + rep["tokens"]
        logits = model(torch.tensor([ids], device="cuda")).logits[0].float()
        lp = torch.log_softmax(logits[len(rep["prompt_ids"]) - 1: len(ids) - 1], dim=-1)
        v, i = lp.topk(20, dim=-1)
        out.append({"i": rep["i"], "top": [[[int(t), float(x)] for t, x in zip(ii, vv)] for ii, vv in
                                           zip(i.tolist(), v.tolist())]})
        print(rep["i"], len(rep["tokens"]), flush=True)
with gzip.open(a.out, "wt") as f:
    json.dump(out, f)
print("wrote", a.out)
