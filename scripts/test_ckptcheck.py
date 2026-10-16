#!/usr/bin/env python3
"""The contract checker accepts a config built from contract/bolt.json and rejects a changed group."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckptcheck


def config_from(contract):
    blocks = []
    moe = iter(contract["moe_schedule"])
    for mark in contract["pattern"]:
        if mark == "M":
            blocks.append({"block_type": "mamba"})
        elif mark == "A":
            blocks.append({"block_type": "attention"})
        else:
            row = next(moe)
            blocks.append({
                "block_type": "moe",
                "n_routed_experts": row["experts"],
                "moe_intermediate_size": row["width"],
                "num_experts_per_tok": row["top_k"],
            })
    layers = {}
    n = 0
    for algo, count in contract["quant_counts"].items():
        for i in range(count):
            layers["t%d" % n] = {"quant_algo": algo}
            n += 1
    vision = dict(contract["vision"])
    model = vision.pop("model")
    vision.pop("image_tag_type", None)
    vision.pop("force_image_size", None)
    vision["args"] = {"model": model}
    return {
        "model_type": contract["top_model_type"],
        "architectures": list(contract["architectures"]),
        "llm_config": {
            "model_type": contract["llm_model_type"],
            "hidden_size": contract["hidden"],
            "vocab_size": contract["vocab"],
            "max_position_embeddings": contract["max_position_embeddings"],
            "mamba_num_heads": contract["mamba_num_heads"],
            "mamba_head_dim": contract["mamba_head_dim"],
            "n_groups": contract["n_groups"],
            "ssm_state_size": contract["ssm_state_size"],
            "conv_kernel": contract["conv_kernel"],
            "num_attention_heads": contract["num_attention_heads"],
            "num_key_value_heads": contract["num_key_value_heads"],
            "head_dim": contract["head_dim"],
            "moe_latent_size": contract["moe_latent_size"],
            "moe_shared_expert_intermediate_size": contract["shared_expert_width"],
            "n_shared_experts": contract["n_shared_experts"],
            "routed_scaling_factor": contract["routed_scaling_factor"],
            "scoring_func": contract["scoring_func"],
            "hidden_act": contract["hidden_act"],
            "mamba_ssm_cache_dtype": contract["mamba_ssm_cache_dtype"],
            "block_configs": blocks,
        },
        "quantization_config": {
            "quant_method": contract["quant_method"],
            "quant_algo": contract["quant_algo"],
            "quantized_layers": layers,
        },
        "vision_config": vision,
    }


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = ckptcheck.load_contract()

    def test_contract_matches_itself(self):
        self.assertEqual(ckptcheck.check(config_from(self.contract), self.contract), [])

    def test_a_changed_expert_width_is_refused(self):
        cfg = config_from(self.contract)
        cfg["llm_config"]["block_configs"][1]["moe_intermediate_size"] = 999
        bad = ckptcheck.check(cfg, self.contract)
        self.assertTrue(any("layer 1 moe" in line for line in bad), bad)

    def test_summary_names_the_groups(self):
        text = ckptcheck.summary(self.contract)
        self.assertIn("IQ2_XXS 27728", text)
        self.assertIn("top-9", text)


if __name__ == "__main__":
    unittest.main()
