import coremltools as ct
from pathlib import Path
import argparse, types, os
import numpy as np
from transformers import AutoTokenizer
import time

parser = argparse.ArgumentParser(description='Test coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV mlpackage file')
parser_args = parser.parse_args()

model = ct.models.MLModel(str(parser_args.model), compute_units=ct.ComputeUnit.CPU_AND_NE)

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)

spec = model.get_spec()

num_layers = len(spec.description.input) // 3
hidden_size = spec.description.input[1].type.multiArrayType.shape[2]
num_heads = spec.description.input[2].type.multiArrayType.shape[1]
head_size = spec.description.input[2].type.multiArrayType.shape[2]
assert head_size == hidden_size // num_heads

print(f'num_layers: {num_layers}, hidden_size: {hidden_size}, num_heads: {num_heads}')

prompt = "The Eiffel Tower is in the city of"
print(prompt, end='', flush=True)

inputs = {'in0': np.array([[0.0]])}
for i in range(num_layers):
    inputs[f'state_{3*i}_in'] = np.zeros((1, 1, hidden_size))
    inputs[f'state_{3*i+1}_in'] = np.zeros((1, num_heads, head_size, head_size))
    inputs[f'state_{3*i+2}_in'] = np.zeros((1, 1, hidden_size))

for id in tokenizer.encode(prompt):
    inputs['in0'][0][0] = id
    outputs = model.predict(inputs)
    for i in range(num_layers):
        inputs[f'state_{3*i}_in'] = outputs[f'state_{3*i}_out']
        inputs[f'state_{3*i+1}_in'] = outputs[f'state_{3*i+1}_out']
        inputs[f'state_{3*i+2}_in'] = outputs[f'state_{3*i+2}_out']

# calculate the durations
durations = []
for i in range(128):
    token_id = np.argmax(outputs['logits'][0])
    inputs['in0'][0][0] = token_id
    print(tokenizer.decode([token_id]), end='', flush=True)
    for i in range(num_layers):
        inputs[f'state_{3*i}_in'] = outputs[f'state_{3*i}_out']
        inputs[f'state_{3*i+1}_in'] = outputs[f'state_{3*i+1}_out']
        inputs[f'state_{3*i+2}_in'] = outputs[f'state_{3*i+2}_out']
    start_time = time.time()
    outputs = model.predict(inputs)
    durations.append((time.time() - start_time) * 1000)  # convert to milliseconds

avg_duration = sum(durations) / len(durations)
print(f"\n\nAverage prediction time: {avg_duration:.2f} ms")
print(f"Tokens per second: {1000 / avg_duration:.2f}")

# outputs = model.predict(inputs)
# print(outputs)