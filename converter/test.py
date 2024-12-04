import ncnn
import numpy as np

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

tokenizer = TRIE_TOKENIZER('../assets/rwkv_vocab_v20230424.txt')

def sample_logits(out, temperature=1.0, top_p=0.8, top_k=128):
    out = out.flatten()
    out -= np.max(out, axis=-1, keepdims=True)
    probs = np.exp(out) / np.sum(np.exp(out), axis=-1, keepdims=True)
    if top_k == 0:
        return np.argmax(probs)
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    cutoff = sorted_probs[top_k]
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        # probs = torch.tensor(probs).pow(1.0 / temperature).numpy()
        probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

net = ncnn.Net()
net.load_param('test.param')
net.load_model('test.bin')

n_layer, n_head, head_size, vocab_size = 24, 32, 64, 65536

prompt = 'User: Hello, how are you?\n\nAssistant:'
prompt_ids = tokenizer.encode(prompt)
print(prompt, end='', flush=True)

states = []
for i in range(n_layer):
    states.append(np.ascontiguousarray(np.zeros(n_head * head_size), dtype=np.float32))
    states.append(np.ascontiguousarray(np.zeros((n_head, head_size, head_size)), dtype=np.float32))
    states.append(np.ascontiguousarray(np.zeros(n_head * head_size), dtype=np.float32))

for id in prompt_ids:
    ex = net.create_extractor()
    input = ncnn.Mat(np.array([id], dtype=np.int32))
    for i in range(n_layer):
        ex.input(f'state_{3 * i}_in', ncnn.Mat(states[3 * i]))
        ex.input(f'state_{3 * i + 1}_in', ncnn.Mat(states[3 * i + 1]))
        ex.input(f'state_{3 * i + 2}_in', ncnn.Mat(states[3 * i + 2]))
    ex.input('token', input)

    logits = np.array(ex.extract('logits')[1])
    for i in range(n_layer):
        states[3 * i] = np.array(ex.extract(f'state_{3 * i}_out')[1])
        states[3 * i + 1] = np.array(ex.extract(f'state_{3 * i + 1}_out')[1])
        states[3 * i + 2] = np.array(ex.extract(f'state_{3 * i + 2}_out')[1])

for i in range(50):
    id = sample_logits(logits)
    print(tokenizer.decode([id]), end='', flush=True)
    ex = net.create_extractor()
    input = ncnn.Mat(np.array([id], dtype=np.int32))
    for i in range(n_layer):
        ex.input(f'state_{3 * i}_in', ncnn.Mat(states[3 * i]))
        ex.input(f'state_{3 * i + 1}_in', ncnn.Mat(states[3 * i + 1]))
        ex.input(f'state_{3 * i + 2}_in', ncnn.Mat(states[3 * i + 2]))
    ex.input('token', input)

    logits = ex.extract('logits')[1].numpy()
    for i in range(n_layer):
        states[3 * i] = ex.extract(f'state_{3 * i}_out')[1].numpy()
        states[3 * i + 1] = ex.extract(f'state_{3 * i + 1}_out')[1].numpy()
        states[3 * i + 2] = ex.extract(f'state_{3 * i + 2}_out')[1].numpy()

print()