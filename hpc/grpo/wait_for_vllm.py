from __future__ import annotations
import argparse, json, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:8000/v1/models')
p.add_argument('--timeout', type=float, default=600.0)
a = p.parse_args()
deadline = time.time() + a.timeout
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(a.url, timeout=5) as r:
            payload = json.load(r)
        if payload.get('data'):
            print('vLLM ready:', payload['data'][0].get('id'))
            raise SystemExit(0)
    except Exception as e:
        last = e
    time.sleep(2)
raise SystemExit(f'vLLM did not become ready within {a.timeout}s: {last}')
