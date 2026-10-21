#!/usr/bin/env python3
"""Benchmark llama.cpp using the exact token inputs exported by validate.py."""
import argparse
import hashlib
import json
import pathlib
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18091")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()
    inputs = json.loads(pathlib.Path(options.inputs).read_text())
    with pathlib.Path(options.out).open("w") as destination:
        for item in inputs:
            payload = {"prompt": item["ids"], "n_predict": 128, "temperature": 0,
                       "seed": 123, "ignore_eos": True, "cache_prompt": False, "stream": True}
            request = urllib.request.Request(options.url + "/completion", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            started = time.monotonic()
            first, final = None, None
            with urllib.request.urlopen(request, timeout=1800) as response:
                for raw in response:
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    if event.get("content") and first is None:
                        first = time.monotonic() - started
                    if event.get("stop"):
                        final = event
            if first is None or final is None:
                raise RuntimeError("incomplete stream")
            timings = final["timings"]
            if timings["predicted_n"] != 128 or timings["prompt_n"] != len(item["ids"]) or timings["cache_n"] != 0:
                raise RuntimeError("benchmark output length or cold-prompt gate failed")
            record = {"kind": "performance", "engine": "llama.cpp-dcu", "gpus": 2,
                      "prompt_tokens": len(item["ids"]), "repeat": item["repeat"], "output_tokens": 128,
                      "input_sha256": hashlib.sha256(json.dumps(item["ids"]).encode()).hexdigest(),
                      "ttft_seconds": first, "elapsed_seconds": time.monotonic() - started, "timings": timings}
            destination.write(json.dumps(record) + "\n")
            destination.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
