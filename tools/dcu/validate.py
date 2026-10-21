#!/usr/bin/env python3
"""Validate a DTK build and record cold-prompt, non-speculative GPU benchmarks."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.server import StrataEngine, request_timings
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--functional-only", action="store_true")
    parser.add_argument("--layout", choices=["layer", "peer", "helper"], default="layer")
    parser.add_argument("--pcie-frac", type=float, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--aux-cpus", default="off")
    parser.add_argument("--chat-bench", action="store_true")
    parser.add_argument("--reserve-mib", type=int, default=4096)
    options = parser.parse_args()
    if not 0 <= options.pcie_frac <= 1 or options.workers < 1 or options.reserve_mib < 1024:
        parser.error("pcie-frac must be in [0,1] and workers must be positive; reserve-mib must be >= 1024")
    out = pathlib.Path(options.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_gguf(options.gguf)
    eos_ids = sorted({tokenizer.ids[name] for name in ["<|endoftext|>", "<|im_end|>"] if name in tokenizer.ids})
    if not eos_ids:
        raise RuntimeError("the model has no expected end-of-turn tokens")
    template = ChatTemplate(pathlib.Path(options.pack) / "tokenizer/chat_template.jinja")
    checks = [
        ("What is 17 + 25? Reply only with the number.", "42"),
        ("Repeat only this identifier: ALPHA_7391", "ALPHA_7391"),
        ("Extract the product SKU and reply only with the SKU: This product has SKU ZX-4826.", "ZX-4826"),
        ("What is the capital of France? Answer with only the city name.", "Paris"),
        ("Calculate 9 times 8. Reply only with the number.", "72"),
        ("Find the identifier in these records. " + "This record is irrelevant. " * 200 + "The identifier is KEY_52719. Reply only with that identifier.", "KEY_52719"),
        ("Explain Python context managers, including __enter__ and __exit__, in detail.", "__exit__"),
    ]
    corpus = tokenizer.encode("The service reads a record, checks its identifier, and returns the requested value. " * 500)
    inputs = []
    for length in [128, 512, 1024, 2048]:
        for repeat in range(3):
            # A different first token defeats last-request prefix reuse as well as parked prompt caches.
            prefix = tokenizer.encode(chr(ord("A") + len(inputs)))
            assert len(prefix) == 1
            inputs.append({"prompt_tokens": length, "repeat": repeat + 1, "ids": prefix + corpus[:length - 1]})
    (out / "benchmark-inputs.json").write_text(json.dumps(inputs))

    def record(data):
        data["layout"] = options.layout if data.get("gpus") == 2 else "single"
        with (out / "requests.jsonl").open("a") as destination:
            destination.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(json.dumps(data, ensure_ascii=False), flush=True)

    for cards in [int(x) for x in options.gpus.split(",")]:
        if cards not in [1, 2]:
            raise ValueError("this validation supports one or two cards")
        previous = None
        for run in [1, 2]:
            args = ["--pack", str(pathlib.Path(options.pack).resolve()), "--native", str(pathlib.Path(options.gguf).resolve()),
                    "--max-context", "4096", "--prefill", "128", "--pool-workers", str(options.workers), "--spec", "2",
                    "--suffix-draft", "0", "--prompt-cache", "0", "--adapt-every", "0", "--pcie-frac", str(options.pcie_frac),
                    "--aux-cpus", options.aux_cpus,
                    "--stop-eos", "--eos-ids", ",".join(str(token) for token in eos_ids),
                    "--expert-profile", str(ROOT / "data/expert-profile.bin"),
                    "--resident-budget-gib", "24", "--expert-cache", "auto", "--serve"]
            args += ["--vram-reserve-mib", str(options.reserve_mib)]
            if cards == 2:
                if options.layout == "layer":
                    args += ["--layer-split", "24", "--split-device", "1"]
                elif options.layout == "peer":
                    args += ["--peer-device", "1", "--peer-reserve-mib", "4096"]
                else:
                    args += ["--expert-cache-device1", "12000", "--remote-expert-opt"]
            print("START_ENGINE", cards, run, flush=True)
            loading_started = time.monotonic()
            child_env = dict(os.environ, HIP_VISIBLE_DEVICES="0" if cards == 1 else "0,1", STRATA_ROUTE_TAIL_SKIP="0")
            engine = StrataEngine(str(pathlib.Path(options.engine).resolve()), args, cwd=str(ROOT),
                                  log=str(out / f"engine-{cards}gpu-{run}.log"), env=child_env)
            print("ENGINE_READY", cards, run, flush=True)
            maps = pathlib.Path(f"/proc/{engine.proc.pid}/maps").read_text()
            libraries = sorted({line.split()[-1] for line in maps.splitlines()
                                if any(name in line for name in ["librocblas", "libhipblas", "libgalaxyhip", "libamdhip", "libcublas", "libcuda.so"])})
            if any(name in str(libraries) for name in ["libhipblas", "libcublas", "libcuda.so", "/opt/rocm"]):
                engine.close()
                raise RuntimeError("unexpected AMD or NVIDIA math/runtime library in DCU process")
            record({"kind": "configuration", "gpus": cards, "run": run, "args": args,
                    "startup_seconds": time.monotonic() - loading_started, "libraries": libraries,
                    "environment": {key: value for key, value in child_env.items() if key.startswith("STRATA_") or key == "HIP_VISIBLE_DEVICES"},
                    "engine_sha256": hashlib.sha256(pathlib.Path(options.engine).read_bytes()).hexdigest(),
                    "engine_info": engine.info})
            resource_stop = threading.Event()
            resources = {"kind": "resources", "gpus": cards, "run": run, "rss_peak_bytes": 0,
                         "mem_available_min_bytes": 2**64, "gpu_peak_bytes": {}}
            def sample_resources():
                while not resource_stop.is_set():
                    try:
                        status = pathlib.Path(f"/proc/{engine.proc.pid}/status").read_text()
                        rss = re.search(r"VmHWM:\s+(\d+)", status)
                        if rss:
                            resources["rss_peak_bytes"] = max(resources["rss_peak_bytes"], int(rss[1]) * 1024)
                        memory = pathlib.Path("/proc/meminfo").read_text()
                        available = int(re.search(r"MemAvailable:\s+(\d+)", memory)[1]) * 1024
                        resources["mem_available_min_bytes"] = min(resources["mem_available_min_bytes"], available)
                        for path in pathlib.Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"):
                            name = path.parent.parent.name
                            resources["gpu_peak_bytes"][name] = max(resources["gpu_peak_bytes"].get(name, 0), int(path.read_text()))
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    resource_stop.wait(1)
            monitor = threading.Thread(target=sample_resources, daemon=True)
            monitor.start()
            try:
                answers = []
                for index, (prompt, expected) in enumerate(checks * (2 if run == 1 else 1)):
                    text = template.render([{"role": "user", "content": prompt}], enable_thinking=False)
                    ids = tokenizer.encode(text, parse_special=True)
                    tokens, first = [], None
                    start = time.monotonic()
                    for token in engine.generate(ids, 256 if expected == "__exit__" else 128,
                                                 {"temperature": 0, "seed": 123}, threading.Event()):
                        if token is not None:
                            if first is None:
                                first = time.monotonic() - start
                            tokens.append(token)
                    answer = tokenizer.decode(tokens)
                    passed = expected.lower() in answer.lower()
                    if expected == "__exit__":
                        passed = passed and "with" in answer.lower() and len(answer) > 250
                    record({"kind": "functional", "gpus": cards, "run": run, "case": index,
                            "expected": expected, "answer": answer, "passed": passed, "ttft_seconds": first,
                            "elapsed_seconds": time.monotonic() - start,
                            "timings": request_timings(len(ids), len(tokens), engine.last), "engine": engine.last})
                    if not passed:
                        raise RuntimeError("functional answer check failed")
                    answers.append(answer)
                short = answers[:6]
                repeated = run != 1 or short == answers[7:13]
                restarted = previous is None or short == previous
                record({"kind": "repeatability", "gpus": cards, "run": run,
                        "repeat_exact": repeated, "restart_exact": restarted})
                if not repeated or not restarted:
                    raise RuntimeError("short answers differ across repetitions or restarts")
                previous = short
                if run != 2 or options.functional_only:
                    continue
                bias = {str(token): -100 for token in eos_ids}
                for item in inputs:
                    start = time.monotonic()
                    first, tokens = None, []
                    for token in engine.generate(item["ids"], 128, {"temperature": 0, "seed": 123, "logit_bias": bias}, threading.Event()):
                        if token is not None:
                            if first is None:
                                first = time.monotonic() - start
                            tokens.append(token)
                    timing = request_timings(len(item["ids"]), len(tokens), engine.last)
                    record({"kind": "performance", "gpus": cards, "repeat": item["repeat"],
                            "prompt_tokens": len(item["ids"]), "output_tokens": len(tokens),
                            "input_sha256": hashlib.sha256(json.dumps(item["ids"]).encode()).hexdigest(),
                            "output_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                            "answer_preview": tokenizer.decode(tokens)[:256],
                            "ttft_seconds": first, "elapsed_seconds": time.monotonic() - start,
                            "timings": timing, "engine": engine.last})
                    if len(tokens) != 128 or not timing or timing["cache_n"] != 0:
                        raise RuntimeError("benchmark output length or cold-prompt gate failed")
                    if int(engine.last.get("drafts_offered") or 0) != 0:
                        raise RuntimeError("unexpected speculative draft tokens in the baseline")
                if options.chat_bench:
                    from chat_bench import run_chat_bench
                    run_chat_bench(engine, tokenizer, template, lambda row: record(dict(row, gpus=cards)))
            finally:
                resource_stop.set()
                monitor.join(timeout=3)
                record(resources)
                engine.close()
    (out / "validation.done").write_text(time.strftime("%F %T") + "\n")
    print("DCU_VALIDATION_DONE", flush=True)


if __name__ == "__main__":
    main()
