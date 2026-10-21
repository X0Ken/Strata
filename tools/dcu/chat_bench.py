"""Natural-language/code decode workloads shared by the DCU benchmark runners."""
import hashlib
import json
import threading
import time
from serve.server import request_timings


def run_chat_bench(engine, tokenizer, template, record):
    prompts = [
        ('python', 'Write a Python function that parses timestamped log lines, groups errors by hour, and returns the five most frequent messages. Explain edge cases and include tests.'),
        ('explanation', 'Explain how hash tables handle collisions, resizing, and adversarial input. Compare separate chaining with open addressing and give practical examples.'),
        ('chinese', '请详细说明分布式任务队列如何处理任务重试、幂等、超时和工作节点故障，并给出一个具体例子。'),
    ]
    bias = {str(tokenizer.ids[k]): -100 for k in ['<|endoftext|>', '<|im_end|>']}
    for repeat in range(1, 3):
        for name, prompt in prompts:
            ids = tokenizer.encode(template.render([dict(role='user', content=prompt)], enable_thinking=False), parse_special=True)
            start = time.monotonic(); first = None; tokens = []
            for token in engine.generate(ids, 256, dict(temperature=0, seed=123, logit_bias=bias), threading.Event()):
                if token is None: continue
                if first is None: first = time.monotonic() - start
                tokens.append(token)
            timing = request_timings(len(ids), len(tokens), engine.last)
            record(dict(kind='chat_performance', workload=name, repeat=repeat, prompt=prompt, input_ids=ids,
                        prompt_tokens=len(ids), output_tokens=len(tokens), answer=tokenizer.decode(tokens),
                        input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        ttft_seconds=first, elapsed_seconds=time.monotonic()-start, timings=timing, engine=engine.last))
            if len(tokens) != 256 or not timing or timing['cache_n'] or int(engine.last.get('drafts_offered') or 0):
                raise RuntimeError('chat benchmark length/cache/speculation check failed')
