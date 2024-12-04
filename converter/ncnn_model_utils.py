import torch
import torch.nn.functional as F

def check_rwkv_info(state_dict):
    n_layer = 0
    version = 5
    n_head = 0
    head_size = 64
    for k in state_dict.keys():
        layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
        n_layer = max(n_layer, layer_id + 1)
        if 'ln_x' in k:
            version = max(5, version)
        if 'gate.weight' in k:
            version = max(5.1, version)
        if int(version) == 5 and 'att.time_decay' in k:
            n_head = state_dict[k].shape[0]
            if len(state_dict[k].shape) > 1:
                if state_dict[k].shape[1] > 1:
                    version = max(5.2, version)
        if 'time_maa' in k:
            version = max(6, version)
        if 'r_k' in k:
            version = max(7, version)
            n_head, _ = state_dict[k].shape
        if int(version) == 6 and 'time_faaaa' in k:
            n_head = state_dict[k].shape[0]
        if 'emb' in k:
            vocab_size, _ = state_dict[k].shape
    return version, n_layer, n_head, head_size, vocab_size

def write_weightdata(fp, dtype, tensor):
    if dtype == torch.float32:
        fp.write(tensor.to(torch.float32).numpy().tobytes())
    elif dtype == torch.float16:
        # write 0x01306B47
        fp.write(b'\x47\x6B\x30\x01')
        fp.write(tensor.to(torch.float16).numpy().tobytes())
        # align to 32bit
        if len(tensor.flatten()) % 2 != 0:
            fp.write(b'\x00\x00')
    else:
        assert 0, f'unsupported dtype {dtype}'

reshape_count = 0
def build_reshape(param_lines, input, output, shape, layer_count, blob_count):
    global reshape_count
    line = f'Reshape reshape_{reshape_count} 1 1 {input} {output}'
    if len(shape) == 1:
        line += f' 0={shape[0]}'
    elif len(shape) == 2:
        line += f' 0={shape[1]} 1={shape[0]}'
    elif len(shape) == 3:
        line += f' 0={shape[2]} 1={shape[1]} 2={shape[0]}'
    elif len(shape) == 4:
        line += f' 0={shape[3]} 1={shape[2]} 2={shape[1]} 11={shape[0]}'
    else:
        assert 0, f'unsupported weight shape {shape}'
    line += '\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 1
    reshape_count += 1
    return layer_count, blob_count

def build_inp_emb(param_lines, fp, w, n_head, head_size, vocab_size, n_layers, layer_count, blob_count):
    param_lines.append('Input input_0 0 1 token 0=1 1=1 2=1\n')
    for i in range(n_layers):
        param_lines.append(f'Input input_{3 * i + 1} 0 1 state_{3 * i}_in 0={n_head * head_size}\n')
        param_lines.append(f'Input input_{3 * i + 2} 0 1 state_{3 * i + 1}_in 0={head_size} 1={head_size} 2={n_head}\n')
        param_lines.append(f'Input input_{3 * i + 3} 0 1 state_{3 * i + 2}_in 0={n_head * head_size}\n')
        layer_count += 3
        blob_count += 3
    param_lines.append(f'Embed embedding 1 1 token emb 0={n_head * head_size} 1={vocab_size} 3={n_head * head_size * vocab_size}\n')
    write_weightdata(fp, torch.float16, F.layer_norm(w['emb.weight'], w['emb.weight'].size()[-1:], weight=w['blocks.0.ln0.weight'].flatten(), bias=w['blocks.0.ln0.bias'].flatten()).half())
    layer_count += 2
    blob_count += 2
    return layer_count, blob_count

layernorm_count = 0
def build_layernorm(param_lines, fp, input, output, weight_gamma, weight_beta, layer_count, blob_count):
    assert len(weight_gamma.shape) == 1
    assert len(weight_beta.shape) == 1
    assert weight_gamma.shape[0] == weight_beta.shape[0]
    global layernorm_count
    param_lines.append(f'LayerNorm layernorm_{layernorm_count} 1 1 {input} {output} 0={weight_gamma.shape[0]} 1=0.00001 2=1\n')
    write_weightdata(fp, torch.float32, weight_gamma)
    write_weightdata(fp, torch.float32, weight_beta)
    layer_count += 1
    blob_count += 1
    layernorm_count += 1
    return layer_count, blob_count

sub_count = 0
def build_sub(param_lines, input1, input2, output, layer_count, blob_count):
    global sub_count
    param_lines.append(f'BinaryOp sub_{sub_count} 2 1 {input1} {input2} {output} 0=1\n')
    layer_count += 1
    blob_count += 1
    sub_count += 1
    return layer_count, blob_count

add_count = 0
def build_add(param_lines, input1, input2, output, layer_count, blob_count):
    global add_count
    param_lines.append(f'BinaryOp add_{add_count} 2 1 {input1} {input2} {output} 0=0\n')
    layer_count += 1
    blob_count += 1
    add_count += 1
    return layer_count, blob_count

mul_count = 0
def build_mul(param_lines, input1, input2, output, layer_count, blob_count):
    global mul_count
    param_lines.append(f'BinaryOp mul_{mul_count} 2 1 {input1} {input2} {output} 0=2\n')
    layer_count += 1
    blob_count += 1
    mul_count += 1
    return layer_count, blob_count

def build_split(param_lines, input, output_list, layer_count, blob_count):
    line = f"Split split_{input} 1 {len(output_list)} {input}"
    for output in output_list:
        line += f' {output}'
    line += '\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += len(output_list)
    return layer_count, blob_count

def build_data(param_lines, fp, weight, name, dtype, layer_count, blob_count):
    line = f"MemoryData data_{name} 0 1 {name}"
    if len(weight.size()) == 1:
        line += f' 0={weight.size()[0]}'
    elif len(weight.size()) == 2:
        line += f' 0={weight.size()[1]} 1={weight.size()[0]}'
    elif len(weight.size()) == 3:
        line += f' 0={weight.size()[2]} 1={weight.size()[1]} 2={weight.size()[0]}'
    elif len(weight.size()) == 4:
        line += f' 0={weight.size()[3]} 1={weight.size()[2]} 2={weight.size()[1]} 11={weight.size()[0]}'
    else:
        assert 0, f'unsupported weight shape {weight.size()}'
    
    # auto dtype
    line += ' 21=0\n'
    param_lines.append(line)
    write_weightdata(fp, dtype, weight)
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

sigmoid_count = 0
def build_sigmoid(param_lines, input, output, layer_count, blob_count):
    global sigmoid_count
    param_lines.append(f'Sigmoid sigmoid_{sigmoid_count} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    sigmoid_count += 1
    return layer_count, blob_count

relu_count = 0
def build_relu(param_lines, input, output, layer_count, blob_count):
    global relu_count
    param_lines.append(f'ReLU relu_{relu_count} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    relu_count += 1
    return layer_count, blob_count

def build_square(param_lines, input, output, layer_count, blob_count):
    param_lines.append(f'UnaryOp square_{input} 1 1 {input} {output} 0=4\n')
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_tanh(param_lines, input, output, layer_count, blob_count):
    param_lines.append(f'TanH tanh_{input} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_linear(param_lines, fp, input, output, weight, dtype, layer_count, blob_count):
    param_lines.append(f'Gemm gemm_{output} 1 1 {input} {output} 4=0 5=1 6=0 7=0 8={weight.shape[0]} 9={weight.shape[1]} 10=-1\n')
    write_weightdata(fp, dtype, weight.t().flatten())
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_matmul(param_lines, input1, input2, output, layer_count, blob_count):
    param_lines.append(f'MatMul matmul_{output} 2 1 {input1} {input2} {output} 0=0\n')
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_squeeze(param_lines, input, output, dim, layer_count, blob_count):
    line = f'Squeeze squeeze_{output} 1 1 {input} {output}'
    for d in dim:
        if d == 3:
            line += ' 11=1'
        else:
            line += f' {d}=1'
    line += '\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 1
    return layer_count, blob_count

def build_slice_wkvrg(param_lines, input, outputs, layer_count, blob_count):
    line = f'Slice slice_{input} 1 5 {input}'
    for output in outputs:
        line += f' {output}'
    line += ' -23300=5,-233,-233,-233,-233,-233 1=0\n'
    param_lines.append(line)
    layer_count += 1
    blob_count += 5
    return layer_count, blob_count

groupnorm_count = 0
def build_groupnorm(param_lines, fp, input, output, weight, bias, num_groups, layer_count, blob_count):
    global groupnorm_count
    param_lines.append(f'GroupNorm groupnorm_{groupnorm_count} 1 1 {input} {output} 0={num_groups} 1={weight.flatten().shape[0]} 2=0.00005 3=1\n')
    layer_count += 1
    blob_count += 1
    groupnorm_count += 1
    write_weightdata(fp, torch.float32, weight)
    write_weightdata(fp, torch.float32, bias)
    return layer_count, blob_count

exp_count = 0
def build_exp(param_lines, input, output, layer_count, blob_count, neg_x=False):
    global exp_count
    if neg_x:
        param_lines.append(f'Exp exp_{exp_count} 1 1 {input} {output} 1=-1.0\n')
    else:
        param_lines.append(f'Exp exp_{exp_count} 1 1 {input} {output}\n')
    layer_count += 1
    blob_count += 1
    exp_count += 1
    return layer_count, blob_count