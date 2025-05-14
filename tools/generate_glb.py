import os
import base64
import numpy as np
import trimesh
from argparse import ArgumentParser
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Accessor, Buffer, BufferView, Material, Image, Texture, Asset
from pygltflib import Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing OBJ files')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for GLB file')
    parser.add_argument('--obj_suffix', type=str, default='.obj', help='Suffix for OBJ files')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output_path
    # input_dir = "/viscam/projects/animal_motion/briannlz/video_object_processing/data/data_3.0.0/test/wolf/zFB8Z2cy_xM_189_004"
    # output_path = "/viscam/projects/animal_motion/briannlz/video_object_processing/data/data_3.0.0/test/wolf/zFB8Z2cy_xM_189_004/mesh.glb"
    obj_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(args.obj_suffix)])

    if len(obj_files) < 2:
        raise ValueError("Need at least two OBJ files for morph target animation.")

    # Load base mesh
    base_mesh = trimesh.load_mesh(os.path.join(input_dir, obj_files[0]), process=False)
    base_positions = np.array(base_mesh.vertices, dtype=np.float32)
    base_faces = np.array(base_mesh.faces, dtype=np.uint32)

    # Load target meshes for morph targets
    target_positions_list = []
    for f in obj_files[1:]:
        tmesh = trimesh.load_mesh(os.path.join(input_dir, f), process=False)
        target_positions_list.append(np.array(tmesh.vertices, dtype=np.float32))

    vertex_count = len(base_positions)
    face_count = len(base_faces)

    # Ensure all targets have same vertex count
    for pos in target_positions_list:
        if pos.shape[0] != vertex_count:
            raise ValueError("All meshes must have the same number of vertices.")

    # Create morph targets: store POSITION deltas from base
    # GLTF morph targets are expressed as offsets from the base mesh
    morph_targets = []
    for pos in target_positions_list:
        morph_targets.append({
            "POSITION": pos - base_positions
        })

    # Build a GLTF structure
    gltf = GLTF2(asset=Asset(generator="CustomScript", version="2.0"))

    # Convert data to binary buffers
    def create_buffer_with_data(binary_data):
        # Base64 encode the binary data
        encoded = base64.b64encode(binary_data).decode('utf-8')
        return Buffer(uri="data:application/octet-stream;base64," + encoded, byteLength=len(binary_data))

    # Create a combined binary blob for base geometry
    # We need to store positions and indices, and morph targets
    #
    # Layout:
    # [positions][indices][morph_target_0_positions][morph_target_1_positions]...
    #
    positions_bytes = base_positions.tobytes()
    indices_bytes = base_faces.flatten().tobytes()

    morph_targets_bytes = []
    for mt in morph_targets:
        morph_pos_bytes = mt["POSITION"].tobytes()
        morph_targets_bytes.append(morph_pos_bytes)

    combined_bytes = positions_bytes + indices_bytes + b''.join(morph_targets_bytes)

    # Create single buffer
    gltf.buffers = [create_buffer_with_data(combined_bytes)]

    # Create Accessors and BufferViews
    byte_offset = 0

    def create_accessor(count, component_type, type_str, offset):
        acc = Accessor()
        acc.count = count
        acc.componentType = component_type
        acc.type = type_str
        acc.byteOffset = offset
        return acc

    def create_buffer_view(byte_length, offset):
        bv = BufferView()
        bv.buffer = 0
        bv.byteOffset = offset
        bv.byteLength = byte_length
        return bv

    # Positions Accessor & BufferView
    position_byte_length = len(positions_bytes)
    position_bv = create_buffer_view(position_byte_length, byte_offset)
    gltf.bufferViews.append(position_bv)
    pos_accessor_index = len(gltf.accessors)
    pos_accessor = create_accessor(vertex_count, 5126, "VEC3", 0) # 5126 = float
    pos_accessor.bufferView = len(gltf.bufferViews)-1
    # Compute min/max for position accessor (for viewer optimization)
    pos_min = base_positions.min(axis=0).tolist()
    pos_max = base_positions.max(axis=0).tolist()
    pos_accessor.min = pos_min
    pos_accessor.max = pos_max
    gltf.accessors.append(pos_accessor)
    byte_offset += position_byte_length

    # Indices Accessor & BufferView
    indices_byte_length = len(indices_bytes)
    indices_bv = create_buffer_view(indices_byte_length, byte_offset)
    gltf.bufferViews.append(indices_bv)
    indices_accessor_index = len(gltf.accessors)
    indices_accessor = create_accessor(face_count * 3, 5125, "SCALAR", 0) # 5125 = unsigned int
    indices_accessor.bufferView = len(gltf.bufferViews)-1
    # min/max for indices (not strictly required)
    indices_accessor.min = [int(base_faces.min())]
    indices_accessor.max = [int(base_faces.max())]
    gltf.accessors.append(indices_accessor)
    byte_offset += indices_byte_length

    # Morph target accessors
    morph_target_accessors = []
    for mt_i, mt_pos_bytes in enumerate(morph_targets_bytes):
        mt_pos_len = len(mt_pos_bytes)
        mt_bv = create_buffer_view(mt_pos_len, byte_offset)
        gltf.bufferViews.append(mt_bv)
        mt_pos_accessor = create_accessor(vertex_count, 5126, "VEC3", 0)
        mt_pos_accessor.bufferView = len(gltf.bufferViews)-1
        # Compute min/max of delta (optional, can help viewers)
        mt_positions = morph_targets[mt_i]["POSITION"]
        mt_pos_accessor.min = mt_positions.min(axis=0).tolist()
        mt_pos_accessor.max = mt_positions.max(axis=0).tolist()
        gltf.accessors.append(mt_pos_accessor)
        morph_target_accessors.append(len(gltf.accessors)-1)
        byte_offset += mt_pos_len

    # Create a Mesh with morph targets
    primitive = Primitive()
    primitive.attributes = {"POSITION": pos_accessor_index}
    primitive.indices = indices_accessor_index

    # Morph targets in GLTF are an array of dictionaries,
    # each referencing an accessor with the POSITION deltas.
    primitive.targets = []
    for mta_idx in morph_target_accessors:
        primitive.targets.append({"POSITION": mta_idx})

    my_mesh = Mesh(primitives=[primitive])
    # Initialize all morph target weights to 0.0
    my_mesh.weights = [0.0]*len(morph_target_accessors)
    gltf.meshes.append(my_mesh)

    # Create a node to hold this mesh
    my_node = Node(mesh=0)
    gltf.nodes.append(my_node)

    # Create a scene that includes the node
    my_scene = Scene(nodes=[0])
    gltf.scenes.append(my_scene)
    gltf.scene = 0

    # Create an animation that cycles through the morph targets
    # Let's say each morph target will be active at a different keyframe
    #
    # For simplicity, create an animation with a sampler that sets one
    # morph target weight to 1.0 at distinct times, and 0.0 otherwise.
    time_vals = np.linspace(0, len(morph_targets), len(morph_targets)+1, endpoint=True).astype(np.float32)
    # We'll cycle through each morph target one by one
    weight_keyframes = []
    for i in range(len(morph_targets)+1):
        # all zeros
        w = [0.0]*len(morph_targets)
        if i < len(morph_targets):
            w[i] = 1.0
        weight_keyframes.append(w)
    weight_keyframes = np.array(weight_keyframes, dtype=np.float32)

    # Flatten the weight array
    weight_flat = weight_keyframes.flatten()

    # Create a separate buffer for animation data (or put it in main buffer)
    # For simplicity, just inline them here:
    time_bytes = time_vals.tobytes()
    weights_bytes = weight_flat.tobytes()

    anim_combined = time_bytes + weights_bytes
    gltf.buffers[0].byteLength += len(anim_combined)
    gltf.buffers[0].uri = None  # we will re-encode
    full_data = combined_bytes + anim_combined
    gltf.buffers[0].uri = "data:application/octet-stream;base64," + base64.b64encode(full_data).decode('utf-8')

    # After re-encoding we need to fix offsets:
    # Time Accessor
    time_bv = create_buffer_view(len(time_bytes), len(combined_bytes))
    gltf.bufferViews.append(time_bv)
    time_accessor_index = len(gltf.accessors)
    time_accessor = Accessor(count=len(time_vals), componentType=5126, type="SCALAR", bufferView=len(gltf.bufferViews)-1)
    time_accessor.min = [float(time_vals.min())]
    time_accessor.max = [float(time_vals.max())]
    gltf.accessors.append(time_accessor)

    # Weights Accessor
    weights_bv = create_buffer_view(len(weights_bytes), len(combined_bytes) + len(time_bytes))
    gltf.bufferViews.append(weights_bv)
    weights_accessor_index = len(gltf.accessors)
    weights_accessor = Accessor(count=weight_keyframes.size, componentType=5126, type="SCALAR", bufferView=len(gltf.bufferViews)-1)
    gltf.accessors.append(weights_accessor)

    # Create a sampler for the animation
    sampler = AnimationSampler(input=time_accessor_index, output=weights_accessor_index, interpolation="LINEAR")

    # Channel: target is the node's morph weights
    channel = AnimationChannel(
        sampler=0,
        target=AnimationChannelTarget(node=0, path="weights")
    )

    anim = Animation(samplers=[sampler], channels=[channel])
    gltf.animations.append(anim)

    # Save as GLB (binary)
    gltf.save_binary(output_path)
    print("Saved animated model to", output_path)
