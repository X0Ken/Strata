#!/usr/bin/env python3
"""Summarize native hipprof SQLite and optional Strata layer counters; export a Chrome trace.

Durations are accumulated device work, not a wall-time percentage. Polling kernels
include CPU service and queue stalls. Kernel copy traffic is distinct from COPY records.
"""
import argparse
import collections
import gzip
import json
import pathlib
import sqlite3


def union_ns(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        if b <= a:
            continue
        total += max(0, b - max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    return total


def analyze(database, trace=None):
    db = sqlite3.connect(f'file:{pathlib.Path(database).resolve()}?mode=ro', uri=True)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    names = {(pid, str(sid)): name for pid, sid, name in db.execute('SELECT PID,STR_ID,STR_NAME FROM STR_TABLE')}
    kernels = collections.defaultdict(lambda: [0, 0])
    intervals = collections.defaultdict(list)
    events = []
    copies = collections.defaultdict(lambda: [0, 0, 0])
    apis = collections.defaultdict(lambda: [0, 0])
    for table in sorted(tables):
        if table.startswith('OPS_'):
            for a, b, dev, queue, sid, pid, tid, idx, duration in db.execute(f'SELECT * FROM "{table}"'):
                name = names.get((pid, str(sid)), str(sid))
                kernels[(dev, name)][0] += 1
                kernels[(dev, name)][1] += duration
                intervals[dev].append((a, b))
                if trace:
                    events.append(dict(name=name, cat='kernel', ph='X', ts=a / 1000, dur=(b-a) / 1000, pid=dev, tid=queue))
        elif table.startswith('COPY_'):
            for a, b, dev, queue, kind, pid, tid, idx, duration, size, memory in db.execute(f'SELECT * FROM "{table}"'):
                row = copies[(dev, kind, size)]
                row[0] += 1; row[1] += duration; row[2] += size
                if trace:
                    events.append(dict(name=f'copy kind={kind} bytes={size}', cat='copy', ph='X', ts=a/1000, dur=(b-a)/1000, pid=dev, tid=queue))
        elif table.startswith('HIP_'):
            for args, duration in db.execute(f'SELECT args,DurationNs FROM "{table}"'):
                row = apis[args.split('(', 1)[0]]
                row[0] += 1; row[1] += duration
    merged = [interval for values in intervals.values() for interval in values]
    span = max(b for a,b in merged) - min(a for a,b in merged)
    result = dict(span_ms=span/1e6, kernel_calls=sum(v[0] for v in kernels.values()),
                  kernel_sum_ms=sum(v[1] for v in kernels.values())/1e6,
                  kernel_union_ms=union_ns(merged)/1e6,
                  device_union_ms={str(k):union_ns(v)/1e6 for k,v in intervals.items()},
                  kernels=[dict(device=dev,name=name,calls=value[0],sum_ms=value[1]/1e6)
                           for (dev,name),value in sorted(kernels.items(),key=lambda x:-x[1][1])],
                  copies=[dict(device=dev,kind=kind,bytes_per_copy=size,calls=v[0],sum_ms=v[1]/1e6,total_bytes=v[2])
                          for (dev,kind,size),v in sorted(copies.items())],
                  apis=[dict(name=name,calls=v[0],sum_ms=v[1]/1e6) for name,v in sorted(apis.items(),key=lambda x:-x[1][1])])
    result['cross_device_overlap_ms'] = sum(result['device_union_ms'].values())-result['kernel_union_ms']
    if trace:
        first = min(e['ts'] for e in events)
        for e in events: e['ts'] -= first
        opener = gzip.open if str(trace).endswith('.gz') else open
        with opener(trace,'wt') as f: json.dump(dict(traceEvents=events,displayTimeUnit='ms'),f,separators=(',',':'))
    db.close()
    return result


def layers(paths):
    result = {}
    for path in paths:
        rows = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines()]
        requests = {}
        for phase, request in sorted({(r['phase'],r['request']) for r in rows}):
            chosen = [r for r in rows if (r['phase'],r['request']) == (phase,request)]
            calls = max((r['host_calls'] for r in chosen),default=0)
            windows = max((r['gpu_windows'] for r in chosen),default=0)
            requests[f'{phase}-{request}'] = dict(
                windows=windows, max_layer_host_calls=calls,
                host_service_ms_per_window=sum(r['host_service_ms'] for r in chosen)/max(calls,1),
                host_wait_ms_per_window=sum(r['host_wait_ms'] for r in chosen)/max(calls,1),
                gpu_segment_ms_per_window=[sum(r['gpu_segment_ns'][i] for r in chosen)/max(windows,1)/1e6 for i in range(33)],
                slowest_host_layers=sorted([dict(layer=r['layer'],device=r['device'],kind=r['kind'],
                    service_ms_per_call=r['host_service_ms']/max(r['host_calls'],1),
                    wait_ms_per_call=r['host_wait_ms']/max(r['host_calls'],1)) for r in chosen],
                    key=lambda r:-r['service_ms_per_call'])[:8])
        result[pathlib.Path(path).parent.name] = requests
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db');p.add_argument('--layers',nargs='*',default=[]);p.add_argument('--trace');p.add_argument('--out',required=True)
    a=p.parse_args()
    if not a.db and not a.layers: p.error('provide --db or --layers')
    if a.trace and not a.db: p.error('--trace requires --db')
    result={}
    if a.db: result['native_trace']=analyze(a.db,a.trace)
    if a.layers: result['layers']=layers(a.layers)
    pathlib.Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
