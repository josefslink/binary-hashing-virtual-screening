from __future__ import annotations

import argparse
import dataclasses

from src.core.config import load_yaml_config
from src.core.drugclip_wrapper import DrugClipBenchmarkConfig, DrugClipWrapper


def build_config(args: argparse.Namespace) -> DrugClipBenchmarkConfig:
    config_values = {}
    if args.config:
        config_values.update(load_yaml_config(args.config))

    overrides = {
        "checkpoint_path": args.checkpoint,
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "repo_dir": args.repo_dir,
        "task": args.task,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_pocket_atoms": args.max_pocket_atoms,
        "seed": args.seed,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            config_values[key] = value

    if args.no_fp16:
        config_values["fp16"] = False  # CLI flag wins outright
    elif "fp16" not in config_values:
        config_values["fp16"] = True  # otherwise default true unless the config said otherwise

    if args.allow_rdkit_mismatch:
        config_values["allow_rdkit_mismatch"] = True
    if args.cpu:
        config_values["cpu"] = True
        config_values["fp16"] = False  # fp16 needs a GPU; forcing cpu forces this off too

    required = ["checkpoint_path", "data_dir", "results_dir", "repo_dir"]
    missing = [key for key in required if not config_values.get(key)]
    if missing:
        raise SystemExit(f"Missing required DrugCLIP config values: {missing}")

    # A config may be shared with the downstream scoring pass (see
    # configs/drugclip/pcba_hamming.yaml), which adds keys like `metric` and
    # `embeddings_dir` that this dataclass does not model. Drop anything it
    # cannot accept rather than failing on the extra keys.
    known = {field.name for field in dataclasses.fields(DrugClipBenchmarkConfig)}
    return DrugClipBenchmarkConfig(**{k: v for k, v in config_values.items() if k in known})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DrugCLIP native benchmarking through a project wrapper.")
    parser.add_argument("--config", type=str, default="configs/drugclip/pcba.yaml")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--results-dir", type=str)
    parser.add_argument("--repo-dir", type=str)
    parser.add_argument("--task", choices=["PCBA", "DUDE"], default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-pocket-atoms", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--python-executable",
        type=str,
        help="Python executable from an isolated DrugCLIP environment",
    )
    parser.add_argument("--allow-rdkit-mismatch", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force DrugCLIP inference onto CPU")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    wrapper = DrugClipWrapper(config)
    issues = wrapper.validate_environment(python_executable=args.python_executable)
    command = wrapper.build_test_command(python_executable=args.python_executable)

    # always shown, even in dry-run, so the command can be inspected or copied
    print("DrugCLIP command:")
    print(" ".join(command))
    if issues:
        print("\nEnvironment issues:")
        for issue in issues:
            print(f"- {issue}")

    if args.print_command or not args.run:
        return  # dry-run by default; --run is required to actually launch DrugCLIP

    if issues:
        raise SystemExit("Resolve the listed environment issues before running DrugCLIP.")

    # Output streams live to stdout/stderr (see run_native_benchmark); nothing to print here.
    result = wrapper.run_native_benchmark(python_executable=args.python_executable)
    raise SystemExit(result.returncode)  # propagate DrugCLIP's own exit code


if __name__ == "__main__":
    main()