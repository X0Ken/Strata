#!/usr/bin/env python3
"""Measure exact-route DCU CPU/PCIe expert splits with the validation inputs."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from serve.server import StrataEngine, request_timings
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer


def thread_affinity(pid):
    result = {}
    for task in pathlib.Path(f'/proc/{pid}/task').iterdir():
        try:
            mask = ','.join(str(cpu) for cpu in sorted(os.sched_getaffinity(int(task.name))))
            result[mask] = result.get(mask, 0) + 1
        except (ProcessLookupError, FileNotFoundError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--baseline', required=True, help='validate.py output directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpus', type=int, choices=[1, 2], default=2)
    parser.add_argument('--shared-stream', choices=['default', '0', '1'], default='default')
    parser.add_argument('--mode', choices=['layer', 'helper', 'peer'], default='layer')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--fractions', default='0,0.25,0.55,1')
    parser.add_argument('--lengths', default='128,2048')
    parser.add_argument('--repeats', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--aux-cpus', default='off', help='upstream auxiliary thread CPU placement')
    parser.add_argument('--chat-bench', action='store_true', help='also measure code, English and Chinese prompts')
    parser.add_argument('--reserve-mib', type=int, help='override saved VRAM reserve')
    parser.add_argument('--warmup', action='store_true', help='record an untimed 128-token prompt before scoring')
    parser.add_argument('--profile', action='store_true', help='diagnostic kernels change execution; not a speed score')
    options = parser.parse_args()
    fractions = [float(x) for x in options.fractions.split(',')]
    if not fractions or any(not 0 <= x <= 1 for x in fractions):
        parser.error('fractions must be between zero and one')
    baseline = pathlib.Path(options.baseline)
    out = pathlib.Path(options.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (baseline / 'requests.jsonl').read_text().splitlines()]
    configuration = next(row for row in rows if row['kind'] == 'configuration' and row['gpus'] == options.gpus)
    args = list(configuration['args'])
    args[args.index('--pool-workers') + 1] = str(options.workers)
    args += ['--aux-cpus', options.aux_cpus]
    if options.reserve_mib is not None:
        if options.reserve_mib < 1024:
            parser.error('reserve-mib must be at least 1024')
        args[args.index('--vram-reserve-mib') + 1] = str(options.reserve_mib)
    if options.mode != 'layer':
        if options.gpus != 2:
            parser.error('helper/peer modes require two visible GPUs')
        for flag in ['--layer-split', '--split-device']:
            if flag in args:
                index = args.index(flag)
                del args[index:index + 2]
        if options.mode == 'helper':
            args += ['--expert-cache-device1', '12000', '--remote-expert-opt']
        else:
            args += ['--peer-device', '1', '--peer-reserve-mib', '4096']
    tokenizer = Tokenizer.from_gguf(args[args.index('--native') + 1])
    template = ChatTemplate(pathlib.Path(args[args.index('--pack') + 1]) / 'tokenizer/chat_template.jinja')
    bias = {str(tokenizer.ids[name]): -100 for name in ['<|endoftext|>', '<|im_end|>']}
    inputs = json.loads((baseline / 'benchmark-inputs.json').read_text())
    lengths = [int(x) for x in options.lengths.split(',')]
    inputs = [x for x in inputs if x['prompt_tokens'] in lengths and x['repeat'] <= options.repeats]
    if len(inputs) != len(lengths) * options.repeats:
        parser.error('lengths must refer to the saved validation inputs')
    env = dict(os.environ, HIP_VISIBLE_DEVICES='0' if options.gpus == 1 else '0,1',
               STRATA_ROUTE_TAIL_SKIP='0', STRATA_DECODE_TIMING='1')
    if options.shared_stream != 'default':
        env['STRATA_SH_STREAM'] = options.shared_stream
    if options.profile:
        env['STRATA_VERIFY_PROFILE'] = '1'
    else:
        env.pop('STRATA_VERIFY_PROFILE', None)

    def record(row):
        row.update(gpus=options.gpus, workers=options.workers, profile=options.profile,
                   aux_cpus=options.aux_cpus, mode=options.mode, shared_stream=options.shared_stream)
        with (out / 'requests.jsonl').open('a') as destination:
            destination.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    engine = StrataEngine(str(pathlib.Path(options.engine).resolve()), args, cwd=str(ROOT),
                          log=str(out / 'engine.log'), env=env)
    try:
        record({'kind': 'configuration', 'args': args, 'engine_info': engine.info,
                'fractions': fractions, 'lengths': lengths, 'repeats': options.repeats,
                'pid': engine.proc.pid, 'cpu_affinity': thread_affinity(engine.proc.pid),
                'optimization_env': {k: v for k, v in env.items() if k.startswith('STRATA_')},
                'engine_sha256': hashlib.sha256(pathlib.Path(options.engine).read_bytes()).hexdigest()})
        # Check readable answers at every split before timing it; routed experts are never dropped.
        for fraction in fractions:
            for prompt, expected in [('What is 17 + 25? Reply only with the number.', '42'),
                                     ('Repeat only this identifier: DCU_TUNE_7391', 'DCU_TUNE_7391')]:
                ids = tokenizer.encode(template.render([{'role': 'user', 'content': prompt}], enable_thinking=False), parse_special=True)
                tokens = list(engine.generate(ids, 128, {'temperature': 0, 'seed': 123,
                    'strata_tune': {'pcie_frac': fraction}}, threading.Event()))
                answer = tokenizer.decode([token for token in tokens if token is not None])
                passed = expected in answer
                record({'kind': 'functional', 'pcie_frac': fraction, 'expected': expected, 'answer': answer, 'passed': passed})
                if not passed:
                    raise RuntimeError('expert split failed answer check')
        if options.warmup:
            warm = next(x for x in json.loads((baseline / 'benchmark-inputs.json').read_text()) if x['prompt_tokens'] == 128)
            ids = tokenizer.encode('Z') + warm['ids'][1:]
            tokens = [token for token in engine.generate(ids, 16, {'temperature': 0, 'seed': 123, 'logit_bias': bias}, threading.Event()) if token is not None]
            record({'kind': 'warmup', 'prompt_tokens': len(ids), 'output_tokens': len(tokens), 'engine': engine.last})
        # Rotate order between repeats so clock/cache drift does not always favor the same split.
        for repeat in range(1, options.repeats + 1):
            order = fractions[repeat - 1:] + fractions[:repeat - 1]
            for fraction in order:
                for item in [x for x in inputs if x['repeat'] == repeat]:
                    if len(lengths) == 1:
                        # Adjacent variants otherwise share the exact last prompt, even with parked caches disabled.
                        list(engine.generate(tokenizer.encode('Z'), 1, {'temperature': 0, 'seed': 123}, threading.Event()))
                    started, first, tokens = time.monotonic(), None, []
                    for token in engine.generate(item['ids'], options.output_tokens,
                        {'temperature': 0, 'seed': 123, 'logit_bias': bias,
                         'strata_tune': {'pcie_frac': fraction}}, threading.Event()):
                        if token is not None:
                            if first is None:
                                first = time.monotonic() - started
                            tokens.append(token)
                    timing = request_timings(len(item['ids']), len(tokens), engine.last)
                    record({'kind': 'performance', 'pcie_frac': fraction, 'repeat': repeat,
                            'prompt_tokens': len(item['ids']), 'output_tokens': len(tokens),
                            'input_sha256': hashlib.sha256(json.dumps(item['ids']).encode()).hexdigest(),
                            'output_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                            'answer_preview': tokenizer.decode(tokens)[:256],
                            'ttft_seconds': first, 'elapsed_seconds': time.monotonic() - started,
                            'timings': timing, 'engine': engine.last})
                    if len(tokens) != options.output_tokens or not timing or timing['cache_n'] != 0:
                        raise RuntimeError('benchmark length or cold-prompt check failed')
                    if int(engine.last.get('drafts_offered') or 0):
                        raise RuntimeError('unexpected speculative tokens')
        if options.chat_bench:
            from chat_bench import run_chat_bench
            run_chat_bench(engine, tokenizer, template, record)
    finally:
        engine.close()
    (out / 'tuning.done').write_text(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
