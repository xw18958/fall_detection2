"""Strictly fuse the five fixed-epsilon V4 layer-normalization subgraphs.

Only the standardized result is quantized. Learned gamma/beta remain in the
existing graph. This changes numerical lowering, not the trained formula.
The runtime uses int64 statistics and a float32 inverse square root.
"""
from collections import Counter
import numpy as np
import flatbuffers
from tensorflow.lite.python import schema_py_generated as s

def unpack(data):
    return s.ModelT.InitFromObj(s.Model.GetRootAsModel(data,0))

def code(model,op):
    return model.operatorCodes[op.opcodeIndex].builtinCode

def fuse(data):
    model=unpack(data)
    if len(model.subgraphs)!=1: raise ValueError('Expected one V4 subgraph')
    g=model.subgraphs[0]; producers={int(t):i for i,o in enumerate(g.operators) for t in o.outputs}
    consumers={i:set() for i in range(len(g.tensors))}
    for i,o in enumerate(g.operators):
        for t in o.inputs:
            if t>=0: consumers[int(t)].add(i)
    removed=set(); replacements={}
    custom=s.OperatorCodeT(); custom.builtinCode=s.BuiltinOperator.CUSTOM
    custom.deprecatedBuiltinCode=s.BuiltinOperator.CUSTOM; custom.customCode=b'LayerNormV4'; custom.version=1
    ci=len(model.operatorCodes); model.operatorCodes.append(custom)
    def producer(t,builtin):
        i=producers[int(t)]; o=g.operators[i]
        if code(model,o)!=builtin: raise ValueError('LayerNorm pattern changed')
        return i,o
    def axes(o):
        t=g.tensors[int(o.inputs[1])]
        values=np.frombuffer(model.buffers[t.buffer].data,dtype='<i4')
        if values.tolist()!=[-1]: raise ValueError('LayerNorm axis changed')
        if not o.builtinOptions.keepDims: raise ValueError('LayerNorm keepDims changed')
    for i,op in enumerate(g.operators):
        if code(model,op)!=s.BuiltinOperator.RSQRT: continue
        ai,add=producer(op.inputs[0],s.BuiltinOperator.ADD)
        vi,var=producer(add.inputs[0],s.BuiltinOperator.MEAN); axes(var)
        si,square=producer(var.inputs[0],s.BuiltinOperator.SQUARED_DIFFERENCE)
        mi,mean=producer(square.inputs[1],s.BuiltinOperator.MEAN); axes(mean)
        x=int(mean.inputs[0])
        if int(square.inputs[0])!=x: raise ValueError('Variance input mismatch')
        users=consumers[int(op.outputs[0])]
        if len(users)!=1: raise ValueError('Inverse has extra users')
        zi=next(iter(users)); z=g.operators[zi]
        if code(model,z)!=s.BuiltinOperator.MUL: raise ValueError('Expected standardized product')
        centered=next(int(t) for t in z.inputs if t!=op.outputs[0])
        di,sub=producer(centered,s.BuiltinOperator.SUB)
        if list(sub.inputs)!=[x,int(mean.outputs[0])]: raise ValueError('Centering mismatch')
        block={mi,si,vi,ai,i,di,zi}
        for oi in block-{zi}:
            for t in g.operators[oi].outputs:
                if consumers[int(t)]-block: raise ValueError('Normalization has external users')
        inp,out=g.tensors[x],g.tensors[int(z.outputs[0])]
        if list(inp.shape)!=list(out.shape) or inp.shape[-1] not in (24,48): raise ValueError('Normalization shape changed')
        new=s.OperatorT(); new.opcodeIndex=ci; new.inputs=np.array([x],np.int32); new.outputs=z.outputs
        replacements[zi]=new; removed|=block
    if len(replacements)!=5: raise ValueError(f'Expected five norms, got {len(replacements)}')
    g.operators=[replacements[i] if i in replacements else o for i,o in enumerate(g.operators) if i not in removed or i in replacements]
    # Remove dead intermediates to prevent unnecessary arena/tensor overhead.
    used=set(int(t) for o in g.operators for t in list(o.inputs)+list(o.outputs) if t>=0)|set(map(int,g.inputs))|set(map(int,g.outputs))
    remap={old:new for new,old in enumerate(sorted(used))};g.tensors=[g.tensors[i] for i in sorted(used)]
    for o in g.operators:
        o.inputs=np.array([remap[int(t)] if t>=0 else t for t in o.inputs],np.int32)
        o.outputs=np.array([remap[int(t)] for t in o.outputs],np.int32)
    g.inputs=np.array([remap[int(t)] for t in g.inputs],np.int32);g.outputs=np.array([remap[int(t)] for t in g.outputs],np.int32)
    model.signatureDefs=[]
    builder=flatbuffers.Builder(0);builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
    result=bytes(builder.Output());inventory(result);return result

def inventory(data):
    m=unpack(data);g=m.subgraphs[0]
    names={v:k for k,v in vars(s.BuiltinOperator).items() if isinstance(v,int)}
    for o in g.operators:
        if code(m,o)==s.BuiltinOperator.CUSTOM and m.operatorCodes[o.opcodeIndex].customCode != b'LayerNormV4':
            raise ValueError('Unexpected custom operator')
    ops=Counter('LayerNormV4' if code(m,o)==s.BuiltinOperator.CUSTOM else names[code(m,o)] for o in g.operators)
    types=Counter({s.TensorType.INT8:'int8',s.TensorType.INT32:'int32'}.get(t.type,'unsupported') for t in g.tensors)
    if 'unsupported' in types or ops['LayerNormV4']!=5: raise ValueError('Unsupported fused graph')
    def quant(t):return [float(t.quantization.scale[0]),int(t.quantization.zeroPoint[0])]
    return dict(operators=dict(ops),tensor_types=dict(types),float_tensors=[],
        per_channel_conv_scale_counts=[len(g.tensors[o.inputs[1]].quantization.scale) for o in g.operators if code(m,o)==s.BuiltinOperator.CONV_2D],
        input_scale_zero=quant(g.tensors[g.inputs[0]]),output_scale_zero=quant(g.tensors[g.outputs[0]]),
        custom_kernel_arithmetic='LayerNormV4: exact int64 statistics, float32 sqrt; INT8 tensor inputs/outputs')
