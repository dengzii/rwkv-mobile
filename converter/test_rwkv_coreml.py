import coremltools as ct
from pathlib import Path
import argparse, types, os
import numpy as np
from transformers import AutoTokenizer
import time

parser = argparse.ArgumentParser(description='Test coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV mlpackage file')
parser.add_argument('--stateful', action='store_true', help='Use stateful model')
parser_args = parser.parse_args()

model = ct.models.MLModel(str(parser_args.model), compute_units=ct.ComputeUnit.CPU_ONLY)

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)

spec = model.get_spec()

inputs = {'in0': np.array([[0.0]])}
state = None

if not parser_args.stateful:
    # num_layers = spec.description.input[2].type.multiArrayType.shape[0]
    # num_heads = spec.description.input[2].type.multiArrayType.shape[1]
    # head_size = spec.description.input[2].type.multiArrayType.shape[2]
    # hidden_size = spec.description.input[1].type.multiArrayType.shape[2]

    num_layers = len(spec.description.input) // 3
    num_heads = spec.description.input[2].type.multiArrayType.shape[1]
    head_size = spec.description.input[2].type.multiArrayType.shape[2]
    hidden_size = spec.description.input[1].type.multiArrayType.shape[2]

    assert head_size == hidden_size // num_heads

    print(f'num_layers: {num_layers}, hidden_size: {hidden_size}, num_heads: {num_heads}')

    # inputs = {'in0': np.array([[0.0]]), 'state_tokenshift_in': np.zeros(spec.description.input[1].type.multiArrayType.shape), 'state_wkv_in': np.zeros(spec.description.input[2].type.multiArrayType.shape)}
    inputs = {'in0': np.array([[0.0]])}
    for i in range(num_layers):
        inputs[f'state_{3*i}_in'] = np.zeros(spec.description.input[3*i+1].type.multiArrayType.shape)
        inputs[f'state_{3*i+1}_in'] = np.zeros(spec.description.input[3*i+2].type.multiArrayType.shape)
        inputs[f'state_{3*i+2}_in'] = np.zeros(spec.description.input[3*i+3].type.multiArrayType.shape)
else:
    state = model.make_state()

prompt = "The Eiffel Tower is in the city of"
print(prompt, end='', flush=True)

for id in tokenizer.encode(prompt):
    inputs['in0'][0][0] = id
    if not parser_args.stateful:
        outputs = model.predict(inputs)
        # inputs['state_tokenshift_in'] = outputs['state_tokenshift_out']
        # inputs['state_wkv_in'] = outputs['state_wkv_out']
        for i in range(num_layers):
            inputs[f'state_{3*i}_in'] = outputs[f'state_{3*i}_out']
            inputs[f'state_{3*i+1}_in'] = outputs[f'state_{3*i+1}_out']
            inputs[f'state_{3*i+2}_in'] = outputs[f'state_{3*i+2}_out']
    else:
        outputs = model.predict(inputs, state=state)

# calculate the durations
durations = []
for i in range(128):
    token_id = np.argmax(outputs['logits'][0])
    inputs['in0'][0][0] = token_id
    print(tokenizer.decode([token_id]), end='', flush=True)
    if not parser_args.stateful:
        # inputs['state_tokenshift_in'] = outputs['state_tokenshift_out']
        # inputs['state_wkv_in'] = outputs['state_wkv_out']
        for i in range(num_layers):
            inputs[f'state_{3*i}_in'] = outputs[f'state_{3*i}_out']
            inputs[f'state_{3*i+1}_in'] = outputs[f'state_{3*i+1}_out']
            inputs[f'state_{3*i+2}_in'] = outputs[f'state_{3*i+2}_out']

    start_time = time.time()
    if not parser_args.stateful:
        outputs = model.predict(inputs)
    else:
        outputs = model.predict(inputs, state=state)
    durations.append((time.time() - start_time) * 1000)  # convert to milliseconds

avg_duration = sum(durations) / len(durations)
print(f"\n\nAverage prediction time: {avg_duration:.2f} ms")
print(f"Tokens per second: {1000 / avg_duration:.2f}")

# outputs = model.predict(inputs)
# print(outputs)