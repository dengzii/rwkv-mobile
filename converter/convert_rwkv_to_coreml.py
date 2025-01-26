from rwkv_src.rwkv_modeling import RWKV_RNN
from rwkv_src.model_utils import get_dummy_input_for_rwkv_causal_llm
import coremltools as ct
from pathlib import Path
import argparse, types, os
import torch

parser = argparse.ArgumentParser(description='Export coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser_args = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0

model_args.MODEL_NAME = str(parser_args.model).replace('.pth', '')
model = RWKV_RNN(model_args)

inputs = get_dummy_input_for_rwkv_causal_llm(1, 1, model.device, model.args)

model = torch.jit.trace(model, example_inputs=inputs)

ct_inputs = [ct.TensorType('in0', inputs[0].shape)] + [ct.TensorType(f'state_{i-1}_in', inputs[i].shape) for i in range(1, len(inputs))]
ct_outputs = [ct.TensorType(name='logits')] + [ct.TensorType(f'state_{i-1}_out') for i in range(1, len(inputs))]
mlmodel = ct.convert(
    model,
    inputs=ct_inputs,
    outputs=ct_outputs,
)
mlmodel.save(f'{str(os.path.basename(parser_args.model)).replace('.pth', '')}.mlpackage')
