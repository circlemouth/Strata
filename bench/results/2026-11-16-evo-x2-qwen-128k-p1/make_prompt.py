"""Recreate the fully synthetic P1 request bytes; never sends an HTTP request."""
import hashlib,json,pathlib,sys
REQUEST = {'chat_template_kwargs': {'enable_thinking': False}, 'effort_position': 'start', 'frequency_penalty': 0, 'max_tokens': 512, 'messages': [{'role': 'user', 'content': None}], 'min_p': 0, 'model': 'strata-qwen3.8-flash-next-iq3s', 'presence_penalty': 0, 'reasoning_effort': 'none', 'repetition_penalty': 1, 'seed': 42, 'stream': True, 'stream_options': {'include_usage': True}, 'temperature': 0, 'top_k': 1, 'top_p': 1}
EXPECTED_SHA256 = 'f3e54c5a3e33068ee014d931c7ad59f7eb63d5af0427bc7b923f910d5a431cdc'
def request_bytes():
    value=dict(REQUEST)
    value['messages']=[{'role':'user','content':' '.join(['green']*130540)}]
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED_SHA256
    return raw
if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Usage: python make_prompt.py OUTPUT.json')
    path=pathlib.Path(sys.argv[1])
    with path.open('xb') as f:f.write(request_bytes())
    print(EXPECTED_SHA256)
