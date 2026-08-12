from subspace_gguf.report import build_report, markdown


def _module(layer, projection, energy):
    return {
        "layer": layer,
        "projection": projection,
        "shape": [2, 2],
        "parameters": 4,
        "atlas_rank": 2,
        "energy_total": energy,
        "direction_energy": [0.75 * energy, 0.25 * energy],
        "energy_top90_count": 2,
        "energy_top95_count": 2,
        "energy_top95_sample_frequency": [2, 2],
    }


def test_report_contains_starvation_and_six_policies():
    analysis = {
        "metadata": {
            "model": "example/model",
            "dataset": "example/data",
            "dataset_split": "train",
            "samples": 2,
            "sample_seed": 42,
            "metric": "energy",
            "global_energy_thresholds": [0.8, 0.9, 0.95, 0.98],
        },
        "coverage": {
            "full_matrix_rank_bound": 4,
            "top20_modules_by_raw_energy": [
                {
                    "module": "model.layers.0.mlp.gate_proj",
                    "energy": 80.0,
                    "energy_fraction": 0.8,
                }
            ]
        },
        "modules": {
            "model.layers.0.mlp.gate_proj": _module(0, "gate_proj", 80.0),
            "model.layers.1.mlp.gate_proj": _module(1, "gate_proj", 20.0),
        },
    }
    report = build_report(analysis)
    assert len(report["quantization_policies"]) == 6
    assert report["global_thresholds"][0]["zero_direction_modules"] == 1
    rendered = markdown(report)
    assert "Global-threshold starvation" in rendered
    assert "Q3Q2IQ1S" in rendered
