#!/usr/bin/env python3
"""
Zero-dependency protobuf generator for ONNX test fixtures.
Builds compliant ONNX ModelProto binaries for testing lib_onnx.
"""

import struct
import os
from pathlib import Path

def encode_varint(value: int) -> bytes:
    res = bytearray()
    while True:
        to_write = value & 0x7F
        value >>= 7
        if value:
            res.append(to_write | 0x80)
        else:
            res.append(to_write)
            break
    return bytes(res)

def field_varint(field_num: int, val: int) -> bytes:
    tag = (field_num << 3) | 0
    return encode_varint(tag) + encode_varint(val)

def field_bytes(field_num: int, data: bytes) -> bytes:
    tag = (field_num << 3) | 2
    return encode_varint(tag) + encode_varint(len(data)) + data

def field_str(field_num: int, s: str) -> bytes:
    return field_bytes(field_num, s.encode('utf-8'))

def make_dim(dim_val: int) -> bytes:
    # Dimension: field 1 = dim_value (int64)
    return field_varint(1, dim_val)

def make_shape(dims: list[int]) -> bytes:
    # TensorShapeProto: repeated Dimension dim = 1
    content = bytearray()
    for d in dims:
        content.extend(field_bytes(1, make_dim(d)))
    return bytes(content)

def make_tensor_type(elem_type: int, dims: list[int]) -> bytes:
    # Tensor: field 1 = elem_type (int32), field 2 = shape
    content = field_varint(1, elem_type) + field_bytes(2, make_shape(dims))
    # TypeProto: field 1 = tensor_type
    return field_bytes(1, content)

def make_value_info(name: str, elem_type: int, dims: list[int]) -> bytes:
    # ValueInfoProto: field 1 = name, field 2 = type
    return field_str(1, name) + field_bytes(2, make_tensor_type(elem_type, dims))

def make_node(op_type: str, inputs: list[str], outputs: list[str], name: str = "") -> bytes:
    # NodeProto: 1 = input, 2 = output, 3 = name, 4 = op_type
    content = bytearray()
    for inp in inputs:
        content.extend(field_str(1, inp))
    for out in outputs:
        content.extend(field_str(2, out))
    if name:
        content.extend(field_str(3, name))
    content.extend(field_str(4, op_type))
    return bytes(content)

def make_initializer_float(name: str, dims: list[int], values: list[float]) -> bytes:
    # TensorProto:
    # 1 = dims (repeated int64)
    # 2 = data_type (int32, FLOAT=1)
    # 4 = float_data (repeated float)
    # 7 = raw_data (bytes)
    # 8 = name (string)
    raw = struct.pack(f'<{len(values)}f', *values)
    content = bytearray()
    for d in dims:
        content.extend(field_varint(1, d))
    content.extend(field_varint(2, 1)) # FLOAT = 1
    content.extend(field_bytes(9, raw)) # raw_data is field 9
    content.extend(field_str(8, name))
    return bytes(content)

def make_model(nodes: list[bytes], inputs: list[bytes], outputs: list[bytes],
               initializers: list[bytes] = None, opset_version: int = 17) -> bytes:
    # GraphProto:
    # 1 = node, 2 = name, 5 = initializer, 11 = input, 12 = output
    graph = bytearray()
    for n in nodes:
        graph.extend(field_bytes(1, n))
    graph.extend(field_str(2, "lib_onnx_graph"))
    if initializers:
        for init in initializers:
            graph.extend(field_bytes(5, init))
    for inp in inputs:
        graph.extend(field_bytes(11, inp))
    for out in outputs:
        graph.extend(field_bytes(12, out))

    # OperatorSetIdProto: 1 = domain (""), 2 = version
    opset = field_str(1, "") + field_varint(2, opset_version)

    # ModelProto: 1 = ir_version (8), 2 = producer_name, 7 = graph, 8 = opset_import
    model = bytearray()
    model.extend(field_varint(1, 8)) # ir_version 8
    model.extend(field_str(2, "lib_onnx"))
    model.extend(field_bytes(7, bytes(graph)))
    model.extend(field_bytes(8, opset))
    return bytes(model)

def safety_models(fixtures_dir):
    for dtype, name in [(1, 'float32'), (10, 'float16'), (7, 'int64'),
                        (6, 'int32'), (3, 'int8'), (2, 'uint8'), (9, 'bool')]:
        for suffix, shape in [('', [1, 4]), ('_scalar', []), ('_empty', [0, 4]), ('_rank20', [1]*20)]:
            model = make_model([make_node('Identity', ['X'], ['Y'])],
                               [make_value_info('X', dtype, shape)],
                               [make_value_info('Y', dtype, shape)])
            Path(fixtures_dir, f'identity_{name}{suffix}.onnx').write_bytes(model)
    # A real long-running ORT Loop; cancellation uses RunOptions termination.
    def init(name, dtype, raw):
        return field_varint(2,dtype)+field_str(8,name)+field_bytes(9,raw)
    body = (field_bytes(1,make_node('Identity',['cond'],['cond_out'])) +
            field_bytes(1,make_node('Identity',['carry'],['carry_out'])) +
            field_str(2,'long_loop_body') +
            b''.join(field_bytes(11,make_value_info(n,t,d)) for n,t,d in
                     [('iteration',7,[]),('cond',9,[]),('carry',1,[1,4])]) +
            field_bytes(12,make_value_info('cond_out',9,[])) +
            field_bytes(12,make_value_info('carry_out',1,[1,4])))
    attr=field_str(1,'body')+field_bytes(6,body)+field_varint(20,5)
    loop=make_node('Loop',['trip','condition','X'],['Y'])+field_bytes(5,attr)
    Path(fixtures_dir,'long_loop.onnx').write_bytes(make_model(
        [loop],[make_value_info('X',1,[1,4])],[make_value_info('Y',1,[1,4])],
        [init('trip',7,struct.pack('<q',100000000)),init('condition',9,b'\1')]))
    # ORT returns a valid tensor followed by a sequence: exercise partial output cleanup.
    seq_type = field_bytes(4, field_bytes(1, make_tensor_type(1, [1, 4])))
    seq_info = field_str(1, 'sequence') + field_bytes(2, seq_type)
    model = make_model([make_node('Identity', ['X'], ['Y']),
                        make_node('SequenceConstruct', ['X'], ['sequence'])],
                       [make_value_info('X', 1, [1, 4])],
                       [make_value_info('Y', 1, [1, 4]), seq_info])
    Path(fixtures_dir, 'partial_sequence.onnx').write_bytes(model)
    # Cast float -> double, unsupported by this bridge's declared interface.
    attr = field_str(1, 'to') + field_varint(3, 11) + field_varint(20, 2)
    cast = make_node('Cast', ['X'], ['Y']) + field_bytes(5, attr)
    Path(fixtures_dir, 'unsupported_double.onnx').write_bytes(make_model(
        [cast], [make_value_info('X', 1, [1,4])], [make_value_info('Y', 11, [1,4])]))


def main():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    safety_models(fixtures_dir)

    # 1. Identity model: X [1, 4] -> Y [1, 4]
    n_id = make_node("Identity", ["X"], ["Y"], "id_0")
    v_in = make_value_info("X", 1, [1, 4])
    v_out = make_value_info("Y", 1, [1, 4])
    m_id = make_model([n_id], [v_in], [v_out])
    id_path = os.path.join(fixtures_dir, "identity_float32.onnx")
    with open(id_path, "wb") as f:
        f.write(m_id)
    print(f"Generated {id_path} ({len(m_id)} bytes)")

    # 2. MLP accelerator smoke model:
    # input [1, 4] -> MatMul(W1 [4, 4]) -> Add(B1 [1, 4]) -> Relu -> output [1, 4]
    w1_vals = [1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0] # Identity matrix
    b1_vals = [0.5, 0.5, 0.5, 0.5]

    w1_init = make_initializer_float("W1", [4, 4], w1_vals)
    b1_init = make_initializer_float("B1", [1, 4], b1_vals)

    n_mm = make_node("MatMul", ["input", "W1"], ["hidden_0"], "mm_0")
    n_add = make_node("Add", ["hidden_0", "B1"], ["hidden_1"], "add_0")
    n_relu = make_node("Relu", ["hidden_1"], ["output"], "relu_0")

    mlp_in = make_value_info("input", 1, [1, 4])
    mlp_out = make_value_info("output", 1, [1, 4])
    m_mlp = make_model([n_mm, n_add, n_relu], [mlp_in], [mlp_out], [w1_init, b1_init])
    mlp_path = os.path.join(fixtures_dir, "mlp_smoke.onnx")
    with open(mlp_path, "wb") as f:
        f.write(m_mlp)
    print(f"Generated {mlp_path} ({len(m_mlp)} bytes)")

if __name__ == "__main__":
    main()
