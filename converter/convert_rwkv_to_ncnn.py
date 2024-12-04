import argparse
import torch
import numpy as np

from ncnn_model_utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('input', help='input rwkv model file')
argparser.add_argument('output_name', help='output ncnn model name')
args = argparser.parse_args()

weights = torch.load(args.input, map_location='cpu')

version, n_layer, n_head, head_size, vocab_size = check_rwkv_info(weights)
print('version:', version)
print('n_layer:', n_layer)
print('n_head:', n_head)
print('head_size:', head_size)
print('vocab_size:', vocab_size)

ncnn_weights_file = open(args.output_name + '.bin', 'wb')

layer_count = 0
blob_count = 0
ncnn_param_lines = ['7767517\n', '[layer_count] [blob_count]\n']
layer_count, blob_count = build_inp_emb(ncnn_param_lines, ncnn_weights_file, weights, n_head, head_size, vocab_size, n_layer, layer_count, blob_count)

def build_time_mixing_v6(param_lines, fp, w, input, output, layer_id, layer_count, blob_count, n_head, head_size):
    prefix = f'att_{layer_id}_'
    layer_count, blob_count = build_split(param_lines, input, [prefix + 'x_last', prefix + 'x'], layer_count, blob_count)
    layer_count, blob_count = build_layernorm(param_lines, fp, prefix + 'x', prefix + 'xx', w[f'blocks.{layer_id}.ln1.weight'], w[f'blocks.{layer_id}.ln1.bias'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'xx', [prefix + 'xx_0', prefix + 'xx_1', prefix + 'xx_2', f'state_{3*layer_id}_out'], layer_count, blob_count)

    # sub_shifted
    layer_count, blob_count = build_sub(param_lines, f'state_{3*layer_id}_in', prefix + 'xx_0', prefix + 'sx', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'sx', [prefix + 'sx_0', prefix + 'sx_1'], layer_count, blob_count)

    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_maa_x'].flatten(), prefix + 'maa_x', torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_0', prefix + 'maa_x', prefix + 'maa_xx', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xx_1', prefix + 'maa_xx', prefix + 'xxx', layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_maa_w1'], prefix + 'maa_w1', torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'xxx', prefix + 'maa_w1', prefix + 'maa_x_lora', layer_count, blob_count)
    layer_count, blob_count = build_tanh(param_lines, prefix + 'maa_x_lora', prefix + 'maa_x_lora_tanh', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'maa_x_lora_tanh', prefix + 'maa_x_lora_tanh_reshape', [5, 1, -1], layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_maa_w2'], prefix + 'maa_w2', torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'maa_x_lora_tanh_reshape', prefix + 'maa_w2', prefix + 'maa_x_post_lora', layer_count, blob_count)

    w_maa = torch.cat([w[f'blocks.{layer_id}.att.time_maa_w'], w[f'blocks.{layer_id}.att.time_maa_k'], w[f'blocks.{layer_id}.att.time_maa_v'], w[f'blocks.{layer_id}.att.time_maa_r'], w[f'blocks.{layer_id}.att.time_maa_g']], dim=0)
    layer_count, blob_count = build_data(param_lines, fp, w_maa, prefix + 'maa', torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'maa_x_post_lora', prefix + 'maa', prefix + 'maa_wkvrg_pre', layer_count, blob_count)
    layer_count, blob_count = build_squeeze(param_lines, prefix + 'maa_wkvrg_pre', prefix + 'maa_wkvrg_pre_squeezed', [1], layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_1', prefix + 'maa_wkvrg_pre_squeezed', prefix + 'maa_wkvrg_sx', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xx_2', prefix + 'maa_wkvrg_sx', prefix + 'maa_wkvrg', layer_count, blob_count)
    layer_count, blob_count = build_slice_wkvrg(param_lines, prefix + 'maa_wkvrg', [prefix + 'mw', prefix + 'mk', prefix + 'mv', prefix + 'mr', prefix + 'mg'], layer_count, blob_count)

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mw', prefix + 'mw_lora', w[f'blocks.{layer_id}.att.time_decay_w1'].t(), torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_tanh(param_lines, prefix + 'mw_lora', prefix + 'mw_lora_tanh', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mw_lora_tanh', prefix + 'mw_lora_tanh_linear', w[f'blocks.{layer_id}.att.time_decay_w2'].t(), torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_decay'].flatten(), prefix + 'td', torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'mw_lora_tanh_linear', prefix + 'td', prefix + 'time_decay_pre', layer_count, blob_count)
    layer_count, blob_count = build_exp(param_lines, prefix + 'time_decay_pre', prefix + 'time_decay_exp0', layer_count, blob_count, neg_x=False)
    layer_count, blob_count = build_exp(param_lines, prefix + 'time_decay_exp0', prefix + 'time_decay', layer_count, blob_count, neg_x=True)

    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mk', prefix + 'key', (w[f'blocks.{layer_id}.att.key.weight'] / 2), torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mv', prefix + 'value', (w[f'blocks.{layer_id}.att.value.weight'] / 4), torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mr', prefix + 'receptance', w[f'blocks.{layer_id}.att.receptance.weight'], torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'mg', prefix + 'gate', w[f'blocks.{layer_id}.att.gate.weight'], torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'gate', [prefix + 'gate_0', prefix + 'gate_1'], layer_count, blob_count)
    layer_count, blob_count = build_sigmoid(param_lines, prefix + 'gate_0', prefix + 'gate_sigmoid', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'gate_1', prefix + 'gate_sigmoid', prefix + 'gate_silu', layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.att.time_faaaa'].unsqueeze(-1), prefix + 'time_first', torch.float16, layer_count, blob_count)

    # non-customlayer implementation
    layer_count, blob_count = build_reshape(param_lines, prefix + 'key', prefix + 'key_reshape', [n_head, head_size, 1], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'value', prefix + 'value_reshape', [n_head, 1, head_size], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'receptance', prefix + 'receptance_reshape', [n_head, 1, head_size], layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'time_decay', prefix + 'time_decay_reshape', [n_head, head_size, 1], layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'key_reshape', prefix + 'value_reshape', prefix + 'kv', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'kv', [prefix + 'kv_0', prefix + 'kv_1'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, f'state_{3*layer_id+1}_in', [prefix + 'wkv_state_0', prefix + 'wkv_state_1'], layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'kv_0', prefix + 'time_first', prefix + 'kv_time_first', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'kv_time_first', prefix + 'wkv_state_0', prefix + 'kv_tf_state', layer_count, blob_count)
    layer_count, blob_count = build_matmul(param_lines, prefix + 'receptance_reshape', prefix + 'kv_tf_state', prefix + 'wkv_out', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'wkv_state_1', prefix + 'time_decay_reshape', prefix + 'state_td', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'state_td', prefix + 'kv_1', f'state_{3*layer_id+1}_out', layer_count, blob_count)
    layer_count, blob_count = build_reshape(param_lines, prefix + 'wkv_out', prefix + 'wkv_out_flatten', [n_head * head_size], layer_count, blob_count)

    layer_count, blob_count = build_groupnorm(param_lines, fp, prefix + 'wkv_out_flatten', prefix + 'x_gn', w[f'blocks.{layer_id}.att.ln_x.weight'].flatten(), w[f'blocks.{layer_id}.att.ln_x.bias'].flatten(), n_head, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'x_gn', prefix + 'gate_silu', prefix + 'x_gate', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'x_gate', prefix + 'x_out', w[f'blocks.{layer_id}.att.output.weight'], torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'x_out', prefix + 'x_last', output, layer_count, blob_count)
    return layer_count, blob_count

def build_channel_mixing(param_lines, fp, w, input, output, layer_id, layer_count, blob_count):
    prefix = f'ffn_{layer_id}_'
    layer_count, blob_count = build_split(param_lines, input, [prefix + 'x_last', prefix + 'x'], layer_count, blob_count)
    layer_count, blob_count = build_layernorm(param_lines, fp, prefix + 'x', prefix + 'xx', w[f'blocks.{layer_id}.ln2.weight'], w[f'blocks.{layer_id}.ln2.bias'], layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'xx', [prefix + 'xx_0', prefix + 'xx_1', prefix + 'xx_2', f'state_{3*layer_id+2}_out'], layer_count, blob_count)

    # sub_shifted
    layer_count, blob_count = build_sub(param_lines, f'state_{3*layer_id+2}_in', prefix + 'xx_0', prefix + 'sx', layer_count, blob_count)
    layer_count, blob_count = build_split(param_lines, prefix + 'sx', [prefix + 'sx_0', prefix + 'sx_1'], layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.ffn.time_maa_k'].flatten(), prefix + 'maa_k', torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_data(param_lines, fp, w[f'blocks.{layer_id}.ffn.time_maa_r'].flatten(), prefix + 'maa_r', torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_0', prefix + 'maa_k', prefix + 'xk', layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'sx_1', prefix + 'maa_r', prefix + 'xr', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xk', prefix + 'xx_1', prefix + 'xxk', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'xr', prefix + 'xx_2', prefix + 'xxr', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'xxk', prefix + 'key', w[f'blocks.{layer_id}.ffn.key.weight'], torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'xxr', prefix + 'receptance', w[f'blocks.{layer_id}.ffn.receptance.weight'], torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_sigmoid(param_lines, prefix + 'receptance', prefix + 'receptance_sigmoid', layer_count, blob_count)
    layer_count, blob_count = build_relu(param_lines, prefix + 'key', prefix + 'key_relu', layer_count, blob_count)
    layer_count, blob_count = build_square(param_lines, prefix + 'key_relu', prefix + 'key_relu_square', layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, prefix + 'key_relu_square', prefix + 'value', w[f'blocks.{layer_id}.ffn.value.weight'], torch.float16, layer_count, blob_count)
    layer_count, blob_count = build_mul(param_lines, prefix + 'value', prefix + 'receptance_sigmoid', prefix + 'rv', layer_count, blob_count)
    layer_count, blob_count = build_add(param_lines, prefix + 'rv', prefix + 'x_last', output, layer_count, blob_count)
    return layer_count, blob_count

def build_output_head(param_lines, fp, w, input, output, layer_count, blob_count):
    layer_count, blob_count = build_layernorm(param_lines, fp, input, 'norm_head', w['ln_out.weight'].flatten(), w['ln_out.bias'].flatten(), layer_count, blob_count)
    layer_count, blob_count = build_linear(param_lines, fp, 'norm_head', output, w['head.weight'], torch.float16, layer_count, blob_count)
    return layer_count, blob_count

layer_input = 'emb'
for i in range(n_layer):
    if version == 6:
        layer_count, blob_count = build_time_mixing_v6(ncnn_param_lines, ncnn_weights_file, weights, layer_input, f'time_mixing_{i}_out', i, layer_count, blob_count, n_head, head_size)
        layer_count, blob_count = build_channel_mixing(ncnn_param_lines, ncnn_weights_file, weights, f'time_mixing_{i}_out', f'channel_mixing_{i}_out', i, layer_count, blob_count)
    else:
        assert 0, f'unsupported version {version}'
    layer_input = f'channel_mixing_{i}_out'
layer_count, blob_count = build_output_head(ncnn_param_lines, ncnn_weights_file, weights, layer_input, 'logits', layer_count, blob_count)

ncnn_param_lines[1] = f'{layer_count} {blob_count}\n'
with open(args.output_name + '.param', 'w') as ncnn_param_file:
    ncnn_param_file.writelines(ncnn_param_lines)

ncnn_weights_file.close()
