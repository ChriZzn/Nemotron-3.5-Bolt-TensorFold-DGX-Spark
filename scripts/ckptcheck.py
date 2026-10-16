#!/usr/bin/env python3
"""Check a Bolt 24GB snapshot against contract/bolt.json.

The contract is the checkpoint this recipe pins: nemotron_h_omni wrapping
nemotron_h_puzzle, the 88-layer schedule, and the three ModelOpt groups.
A directory or a config.json path both work. Shard files are checked when
model.safetensors.index.json sits next to config.json.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, "contract", "bolt.json")


def load_contract(path=CONTRACT):
    with open(path) as f:
        return json.load(f)


def check(cfg, contract):
    """Return a list of mismatches. Empty means the checkpoint matches the contract."""
    bad = []
    llm = cfg.get("llm_config") or {}
    if cfg.get("model_type") != contract["top_model_type"]:
        bad.append("model_type is %r" % cfg.get("model_type"))
    if llm.get("model_type") != contract["llm_model_type"]:
        bad.append("llm_config.model_type is %r" % llm.get("model_type"))
    if cfg.get("architectures") != contract["architectures"]:
        bad.append("architectures is %r" % cfg.get("architectures"))
    for key, src in (
        ("hidden", "hidden_size"),
        ("vocab", "vocab_size"),
        ("max_position_embeddings", "max_position_embeddings"),
        ("mamba_num_heads", "mamba_num_heads"),
        ("mamba_head_dim", "mamba_head_dim"),
        ("n_groups", "n_groups"),
        ("ssm_state_size", "ssm_state_size"),
        ("conv_kernel", "conv_kernel"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_key_value_heads"),
        ("head_dim", "head_dim"),
        ("moe_latent_size", "moe_latent_size"),
        ("shared_expert_width", "moe_shared_expert_intermediate_size"),
        ("n_shared_experts", "n_shared_experts"),
        ("routed_scaling_factor", "routed_scaling_factor"),
        ("scoring_func", "scoring_func"),
        ("hidden_act", "hidden_act"),
        ("mamba_ssm_cache_dtype", "mamba_ssm_cache_dtype"),
    ):
        if llm.get(src) != contract[key]:
            bad.append("llm_config.%s is %r" % (src, llm.get(src)))
    blocks = llm.get("block_configs") or []
    pattern = "".join({"mamba": "M", "moe": "E", "attention": "A"}.get(b.get("block_type"), "?") for b in blocks)
    if pattern != contract["pattern"]:
        bad.append("layer pattern is %d blocks, contract is %d" % (len(pattern), len(contract["pattern"])))
    moe = [b for b in blocks if b.get("block_type") == "moe"]
    if len(moe) != len(contract["moe_schedule"]):
        bad.append("moe layers %d" % len(moe))
    else:
        for got, exp in zip(moe, contract["moe_schedule"]):
            if got.get("n_routed_experts") != exp["experts"] or got.get("moe_intermediate_size") != exp["width"] or got.get("num_experts_per_tok") != exp["top_k"]:
                bad.append("layer %d moe is %s" % (exp["layer"], {k: got.get(k) for k in ("n_routed_experts", "moe_intermediate_size", "num_experts_per_tok")}))
                break
    q = cfg.get("quantization_config") or {}
    if str(q.get("quant_method", "")).lower() != contract["quant_method"] or q.get("quant_algo") != contract["quant_algo"]:
        bad.append("quantization is %r %r" % (q.get("quant_method"), q.get("quant_algo")))
    counts = {}
    for entry in (q.get("quantized_layers") or {}).values():
        if isinstance(entry, dict):
            counts[entry.get("quant_algo")] = counts.get(entry.get("quant_algo"), 0) + 1
    if counts != contract["quant_counts"]:
        bad.append("quant groups %s" % counts)
    vision = cfg.get("vision_config") or {}
    want = contract["vision"]
    if vision.get("model_type") != want["model_type"] or vision.get("num_hidden_layers") != want["num_hidden_layers"] or vision.get("hidden_size") != want["hidden_size"]:
        bad.append("vision tower is %r" % {k: vision.get(k) for k in ("model_type", "num_hidden_layers", "hidden_size")})
    args = vision.get("args") or {}
    if args.get("model") != want["model"]:
        bad.append("vision model is %r" % args.get("model"))
    return bad


def check_shards(directory, contract):
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return ["no model.safetensors.index.json"]
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != contract["shards"]:
        return ["%d shards, contract says %d" % (len(shards), contract["shards"])]
    missing = [s for s in shards if not os.path.isfile(os.path.join(directory, s)) or os.path.getsize(os.path.join(directory, s)) == 0]
    if missing:
        return ["missing or empty shards: " + ", ".join(missing[:5])]
    return []


def summary(contract):
    q = contract["quant_counts"]
    return (
        "nemotron_h_puzzle, %d layers (%d Mamba, %d MoE, %d attention), "
        "top-%d, latent %d, IQ2_XXS %d, NVFP4 %d, Q8_0 %d, %d shards"
        % (
            contract["layers"],
            contract["mamba_layers"],
            contract["moe_layers"],
            contract["attention_layers"],
            contract["top_k"],
            contract["moe_latent_size"],
            q["IQ2_XXS"],
            q["W4A16_NVFP4"],
            q["Q8_0"],
            contract["shards"],
        )
    )


def main(argv):
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print("usage: ckptcheck.py SNAPSHOT_DIR | config.json", file=sys.stderr)
        return 2
    path = argv[1]
    contract = load_contract()
    directory = None
    if os.path.isdir(path):
        directory = path
        path = os.path.join(path, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    bad = check(cfg, contract)
    if directory is not None:
        bad.extend(check_shards(directory, contract))
    if bad:
        print("checkpoint does not match %s @ %s" % (contract["model_id"], contract["revision"][:12]), file=sys.stderr)
        for line in bad:
            print("  " + line, file=sys.stderr)
        return 1
    print(summary(contract))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
