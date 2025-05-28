from rwkv_src.rwkv_modeling import RWKV_RNN
from rwkv_src.model_utils import get_dummy_input_for_rwkv_causal_llm
import coremltools as ct
from coremltools.optimize.torch.palettization import PostTrainingPalettizerConfig, PostTrainingPalettizer
from pathlib import Path
import argparse, types, os
import torch
from transformers import AutoTokenizer
import numpy as np

parser = argparse.ArgumentParser(description='Export coreml model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser_args = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = False
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 0
model_args.USE_ONNX_L2NORM = False
model_args.USE_ONNX_REDUCE_L2 = False
model_args.USE_STATEFUL_MODEL = False

model_args.MODEL_NAME = str(parser_args.model).replace('.pth', '')
model = RWKV_RNN(model_args)
args = model.args

merge_states = True

if model_args.USE_STATEFUL_MODEL:
    inputs = [torch.tensor([[0]*1 for _ in range(1)], dtype=torch.int32).to(model.device)]
else:
    inputs = get_dummy_input_for_rwkv_causal_llm(1, 1, model.device, model.args, merged_states=merge_states)

# tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-5-world-1b5", trust_remote_code=True)
# prompt = "The Eiffel Tower is in the city of"

# print(prompt, end='', flush=True)
# for token in tokenizer.encode(prompt):
#     inputs[0][0] = token
#     logits, state = model(*inputs)
#     # inputs[1], inputs[2] = state
#     for i in range(args.n_layer):
#         inputs[3*i+1] = state[3*i]
#         inputs[3*i+2] = state[3*i+1]
#         inputs[3*i+3] = state[3*i+2]

# for i in range(128):
#     token = np.argmax(logits[0])
#     print(tokenizer.decode([token]), end='', flush=True)
#     inputs[0][0] = token
#     logits, state = model(*inputs)
#     # inputs[1], inputs[2] = state
#     for i in range(args.n_layer):
#         inputs[3*i+1] = state[3*i]
#         inputs[3*i+2] = state[3*i+1]
#         inputs[3*i+3] = state[3*i+2]

# quit()

# config = PostTrainingPalettizerConfig.from_dict({"global_config": 
#                                                 {
#                                                 "n_bits": 4,
#                                                 "granularity": "per_grouped_channel",
#                                                 "group_size": 16
#                                                 }
#                                                 })
# palettizer = PostTr
# ainingPalettizer(model, config)
# palettized_model = palettizer.compress()

# model = torch.jit.trace(palettized_model, example_inputs=inputs)
model = torch.jit.trace(model, example_inputs=inputs)

ct_inputs = [ct.TensorType('in0', inputs[0].shape, dtype=np.int32)]
dtype = np.float16
if not model_args.USE_STATEFUL_MODEL:
    if not merge_states:
        ct_inputs += [ct.TensorType(f'state_{i}_in', inputs[i+1].shape, dtype=dtype) for i in range(len(inputs) - 1)]
    else:
        ct_inputs += [ct.TensorType(f'state_tokenshift_in', inputs[1].shape, dtype=dtype)]
        ct_inputs += [ct.TensorType(f'state_wkv_in', inputs[2].shape, dtype=dtype)]
ct_outputs = [ct.TensorType(name='logits', dtype=dtype)]
if not model_args.USE_STATEFUL_MODEL:
    if not merge_states:
        ct_outputs += [ct.TensorType(f'state_{i}_out', dtype=dtype) for i in range(len(inputs) - 1)]
    else:
        ct_outputs += [ct.TensorType(f'state_tokenshift_out', dtype=dtype)]
        ct_outputs += [ct.TensorType(f'state_wkv_out', dtype=dtype)]

mlmodel = None
if model_args.USE_STATEFUL_MODEL:
    states = []
    # for i in range(args.n_layer):
    #     states.append(
    #         ct.StateType(
    #             wrapped_type=ct.TensorType(
    #                 shape=(1, 1, args.n_embd),
    #             ),
    #             name=f"blocks.{i}.state_att_tokenshift",
    #         )
    #     )
    #     states.append(
    #         ct.StateType(
    #             wrapped_type=ct.TensorType(
    #                 shape=(1, args.n_head, args.head_size, args.head_size),
    #             ),
    #             name=f"blocks.{i}.state_wkv",
    #         )
    #     )
    #     states.append(
    #         ct.StateType(
    #             wrapped_type=ct.TensorType(
    #                 shape=(1, 1, args.n_embd),
    #             ),
    #             name=f"blocks.{i}.state_ffn_tokenshift",
    #         )
    #     )
    # mlmodel = ct.convert(
    #     model,
    #     inputs=ct_inputs,
    #     outputs=ct_outputs,
    #     states=states,
    #     minimum_deployment_target=ct.target.iOS18,
    # )
    # # test
    # state = mlmodel.make_state()
    pass
else:
    mlmodel = ct.convert(
        model,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS18,
        # compute_units=ct.ComputeUnit.CPU_AND_NE
    )

mlmodel.save(f'{str(os.path.basename(parser_args.model)).replace('.pth', '')}_iOS18.mlpackage')

