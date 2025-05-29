from rwkv_src.rwkv_modeling import RWKV_RNN, RWKV_RNN_Stateful
from rwkv_src.model_utils import get_dummy_input_for_rwkv_causal_llm
import coremltools as ct
from coremltools.optimize.torch.quantization import PostTrainingQuantizer, PostTrainingQuantizerConfig
from pathlib import Path
import argparse, types, os
import torch
from transformers import AutoTokenizer
import numpy as np

parser = argparse.ArgumentParser(description='Export coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser.add_argument('--stateful', action='store_true', help='Use stateful model')
parser.add_argument('--int8', action='store_true', help='Use int8 quantization')
parser.add_argument('--int4', action='store_true', help='Use int4 quantization')
parser_args = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.USE_ONNX_L2NORM = False
model_args.USE_ONNX_REDUCE_L2 = False

model_args.MODEL_NAME = str(parser_args.model).replace('.pth', '')
model = RWKV_RNN_Stateful(model_args) if parser_args.stateful else RWKV_RNN(model_args)
args = model.args

merge_states = True

if parser_args.stateful:
    inputs = [torch.tensor([[0]*1 for _ in range(1)], dtype=torch.int32).to(model.device)]
else:
    inputs = get_dummy_input_for_rwkv_causal_llm(1, 1, model.device, model.args, merged_states=merge_states)

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)
prompt = "The Eiffel Tower is in the city of"

# print(prompt, end='', flush=True)
# for token in tokenizer.encode(prompt):
#     inputs[0][0] = token
#     # logits, state = model(*inputs)
#     logits = model(inputs[0])
#     # inputs[1], inputs[2] = state
#     # for i in range(args.n_layer):
#     #     inputs[3*i+1] = state[3*i]
#     #     inputs[3*i+2] = state[3*i+1]
#     #     inputs[3*i+3] = state[3*i+2]

# for i in range(128):
#     token = np.argmax(logits[0])
#     print(tokenizer.decode([token]), end='', flush=True)
#     inputs[0][0] = token
#     logits = model(inputs[0])
#     # logits, state = model(*inputs)
#     # # inputs[1], inputs[2] = state
#     # for i in range(args.n_layer):
#     #     inputs[3*i+1] = state[3*i]
#     #     inputs[3*i+2] = state[3*i+1]
#     #     inputs[3*i+3] = state[3*i+2]

# quit()

if parser_args.int4:
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int4",
                "granularity": "per_block",
                "block_size": 128,
            },
            "module_type_configs": {
            }
        }
    )
    quantizer = PostTrainingQuantizer(model, config)
    model = quantizer.compress()
elif parser_args.int8:
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int8",
                "granularity": "per_channel",
            },
            "module_type_configs": {
            }
        }
    )
    quantizer = PostTrainingQuantizer(model, config)
    model = quantizer.compress()

# model = torch.jit.trace(palettized_model, example_inputs=inputs)
model = torch.jit.trace(model, example_inputs=inputs)

ct_inputs = [ct.TensorType('in0', inputs[0].shape, dtype=np.int32)]
dtype = np.float16
if not parser_args.stateful:
    if not merge_states:
        ct_inputs += [ct.TensorType(f'state_{i}_in', inputs[i+1].shape, dtype=dtype) for i in range(len(inputs) - 1)]
    else:
        ct_inputs += [ct.TensorType(f'state_tokenshift_in', inputs[1].shape, dtype=dtype)]
        ct_inputs += [ct.TensorType(f'state_wkv_in', inputs[2].shape, dtype=dtype)]
ct_outputs = [ct.TensorType(name='logits', dtype=dtype)]
if not parser_args.stateful:
    if not merge_states:
        ct_outputs += [ct.TensorType(f'state_{i}_out', dtype=dtype) for i in range(len(inputs) - 1)]
    else:
        ct_outputs += [ct.TensorType(f'state_tokenshift_out', dtype=dtype)]
        ct_outputs += [ct.TensorType(f'state_wkv_out', dtype=dtype)]

mlmodel = None
if parser_args.stateful:
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2, args.n_layer, args.n_embd),
            ),
            name=f"state_tokenshift",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(args.n_layer, args.n_head, args.head_size, args.head_size),
            ),
            name=f"state_wkv",
        ),
    ]
    mlmodel = ct.convert(
        model,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
    )
    # test
    state = mlmodel.make_state()

    mlmodel.save(f'{str(os.path.basename(parser_args.model)).replace('.pth', '')}_stateful.mlpackage')
else:
    mlmodel = ct.convert(
        model,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS18,
        # compute_units=ct.ComputeUnit.CPU_AND_NE
    )

    mlmodel.save(f'{str(os.path.basename(parser_args.model)).replace('.pth', '')}.mlpackage')

