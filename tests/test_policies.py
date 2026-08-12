from subspace_gguf.policies import DEFAULT_POLICIES, assign_modules
from subspace_gguf.recipes import hf_to_gguf


def modules():
    return {
        "model.layers.0.mlp.gate_proj": {
            "energy_total": 80.0,
            "shape": [10, 10],
            "parameters": 100,
        },
        "model.layers.0.mlp.up_proj": {
            "energy_total": 10.0,
            "shape": [10, 20],
            "parameters": 200,
        },
        "model.layers.0.mlp.down_proj": {
            "energy_total": 8.0,
            "shape": [20, 10],
            "parameters": 200,
        },
        "model.layers.0.self_attn.q_proj": {
            "energy_total": 2.0,
            "shape": [10, 10],
            "parameters": 100,
        },
    }


def test_boundary_rule_keeps_whole_module_in_starting_tier():
    policy = DEFAULT_POLICIES[0]
    assignments = assign_modules(modules(), policy)
    assert [value["precision"] for value in assignments] == [
        "Q8_0",
        "Q8_0",
        "Q6_K",
        "Q4_K",
    ]


def test_qwen_llama_mapping():
    assert hf_to_gguf("model.layers.31.self_attn.o_proj") == "blk.31.attn_output.weight"
    assert hf_to_gguf("language_model.model.layers.4.mlp.down_proj") == "blk.4.ffn_down.weight"
