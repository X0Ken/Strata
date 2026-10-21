#!/usr/bin/env python3
"""Collect a native DTK hipprof decode trace after prefill and warmup, via serve IPC."""
import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, request_timings
from strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True)
    p.add_argument('--baseline', required=True, help='validate.py directory containing args and inputs')
    p.add_argument('--out', required=True)
    p.add_argument('--profiler', default=shutil.which('hipprof'))
    p.add_argument('--reserve-mib', type=int)
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    if not a.profiler: p.error('source DTK env.sh or pass --profiler')
    out = pathlib.Path(a.out).resolve(); out.mkdir(exist_ok=False, parents=True)
    base = pathlib.Path(a.baseline)
    configs = [json.loads(s) for s in (base / 'requests.jsonl').read_text().splitlines()]
    args = list(next(x['args'] for x in configs if x['kind'] == 'configuration' and x['gpus'] == 2))
    for flag, value in [('--vram-reserve-mib', a.reserve_mib), ('--pool-workers', a.workers)]:
        if value is not None: args[args.index(flag) + 1] = str(value)
    profiler = str(pathlib.Path(a.profiler).resolve())
    engine_path = str(pathlib.Path(a.engine).resolve())
    session = f'strata-dcu-{os.getpid()}'
    wrapper = out / 'engine-prof.sh'
    command = [profiler, '--hip-trace', '--trace-off', '--session', session, '-d', str(out), '-o', 'trace.csv', engine_path]
    wrapper.write_text('#!/bin/bash\nexec ' + shlex.join(command) + ' "$@"\n'); wrapper.chmod(0o755)
    tok = Tokenizer.from_gguf(args[args.index('--native') + 1])
    bias = {str(tok.ids[x]): -100 for x in ['<|endoftext|>', '<|im_end|>']}
    sample = next(x for x in json.loads((base / 'benchmark-inputs.json').read_text()) if x['prompt_tokens'] == 128)
    env = dict(os.environ, HIP_VISIBLE_DEVICES='0,1', STRATA_ROUTE_TAIL_SKIP='0', STRATA_SH_STREAM='0', STRATA_DECODE_TIMING='1')
    def record(row):
        with (out / 'requests.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    engine = StrataEngine(str(wrapper), args, cwd=str(ROOT), log=str(out / 'engine.log'), env=env)
    tracing = False
    def control(action):
        subprocess.run([profiler, '--session', session, action], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=20)
    try:
        record(dict(kind='configuration',gpus=2,args=args,pid=engine.proc.pid,engine_info=engine.info,
                    engine_sha256=hashlib.sha256(pathlib.Path(engine_path).read_bytes()).hexdigest(),
                    optimization_env={k:v for k,v in env.items() if k.startswith('STRATA_')}))
        for index, (n, trace) in enumerate([(16, False), (128, False), (16, True), (128, False)]):
            ids = list(sample['ids']); ids[0] = tok.encode(chr(ord('A') + index))[0]
            start = time.monotonic(); tokens = []; first = None
            for token in engine.generate(ids,n,dict(temperature=0,seed=123,logit_bias=bias),threading.Event()):
                if token is None: continue
                if first is None: first = time.monotonic() - start
                tokens.append(token)
                if trace and not tracing:
                    control('--start'); tracing = True
            if tracing: control('--stop'); tracing = False
            record(dict(kind='trace' if trace else ('warmup' if index == 0 else 'performance'),
                        output_tokens=len(tokens),ttft_seconds=first,elapsed_seconds=time.monotonic()-start,
                        output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        timings=request_timings(len(ids),len(tokens),engine.last),engine=engine.last))
            if len(tokens) != n: raise RuntimeError('unexpected output length')
    finally:
        if tracing: control('--stop')
        engine.close()
    (out / 'profile.done').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__': main()
