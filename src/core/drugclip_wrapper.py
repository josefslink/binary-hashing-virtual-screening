from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from src.core.vendor_compat import ensure_drugclip_workdir


@dataclass(slots=True)
class DrugClipBenchmarkConfig:
    checkpoint_path: str
    data_dir: str
    results_dir: str
    repo_dir: str
    task: str = "PCBA"
    batch_size: int = 8
    num_workers: int = 8
    max_pocket_atoms: int = 511  # matches upstream test.sh; its train/retrieval scripts use 256
    fp16: bool = True
    cpu: bool = False
    seed: int = 1
    device: int = 0
    allow_rdkit_mismatch: bool = False
    # Disposable cwd holding a `data` symlink; DrugCLIP resolves its LMDBs relative to
    # cwd (see docs/VENDOR_COMPAT.md).
    workdir: str = "outputs/drugclip_workdir"


class DrugClipWrapper:
    """Drives DrugCLIP's own unimol/test.py entry point as a subprocess."""

    def __init__(self, config: DrugClipBenchmarkConfig) -> None:
        self.config = config

    @staticmethod
    def _probe_python_environment(python_executable: str, allow_rdkit_mismatch: bool = False) -> list[str]:
        # runs in the *target* interpreter, not ours - the drugclip venv can have a
        # different rdkit/unicore than whatever environment is running this wrapper
        probe_script = textwrap.dedent(
            """
            import importlib.metadata
            import importlib.util
            import json

            issues = []
            if importlib.util.find_spec('unicore') is None:
                issues.append("Python package 'unicore' is not installed in the target DrugCLIP environment.")
            try:
                rdkit_version = importlib.metadata.version('rdkit')
            except importlib.metadata.PackageNotFoundError:
                issues.append("Python package 'rdkit' is not installed in the target DrugCLIP environment.")
            else:
                if rdkit_version != '2022.9.5' and not {allow_rdkit_mismatch}:
                    issues.append(
                        "DrugCLIP README expects rdkit==2022.9.5, but the target environment has rdkit==" + rdkit_version + "."
                    )
            print(json.dumps({{"issues": issues}}))
            """
        # double braces above survive this .format()
        ).format(allow_rdkit_mismatch=repr(allow_rdkit_mismatch)).strip()

        try:
            result = subprocess.run(
                [python_executable, "-c", probe_script],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return [f"Python executable not found: {python_executable}"]  # e.g. a venv that was never created

        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
            return [f"Failed to probe DrugCLIP environment with {python_executable}: {stderr}"]

        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError:
            return [
                f"Failed to parse DrugCLIP environment probe output from {python_executable}: {result.stdout.strip() or 'empty output'}"
            ]

        issues = payload.get("issues", [])
        if not isinstance(issues, list):  # defend against a malformed probe payload
            return [f"Unexpected DrugCLIP environment probe payload from {python_executable}: {payload!r}"]
        return [str(issue) for issue in issues]

    def validate_environment(self, python_executable: str | None = None) -> list[str]:
        issues: list[str] = []
        repo_dir = Path(self.config.repo_dir)
        checkpoint_path = Path(self.config.checkpoint_path)
        data_dir = Path(self.config.data_dir)

        if not repo_dir.exists():
            issues.append(f"DrugCLIP repo directory not found: {repo_dir}")
        if not (repo_dir / "unimol" / "test.py").exists():
            issues.append(f"DrugCLIP test entrypoint missing: {repo_dir / 'unimol' / 'test.py'}")
        if not checkpoint_path.exists():
            issues.append(f"DrugCLIP checkpoint not found: {checkpoint_path}")
        if not data_dir.exists():
            issues.append(f"DrugCLIP data directory not found: {data_dir}")
        # falls back to our own interpreter, not necessarily drugclip's venv
        probe_python = python_executable or sys.executable
        issues.extend(self._probe_python_environment(probe_python, allow_rdkit_mismatch=self.config.allow_rdkit_mismatch))
        return issues

    def build_test_command(self, python_executable: str | None = None) -> list[str]:
        # mirrors third_party/drugclip/test.sh's flags one-for-one
        python_bin = python_executable or sys.executable
        repo_dir = Path(self.config.repo_dir).resolve()
        test_script = repo_dir / "unimol" / "test.py"
        command = [
            python_bin,
            str(test_script),
            "--user-dir",
            str((repo_dir / "unimol").resolve()),
            str(Path(self.config.data_dir).resolve()),
            "--valid-subset",
            "test",
            "--results-path",
            str(Path(self.config.results_dir).resolve()),
            "--num-workers",
            str(self.config.num_workers),
            "--ddp-backend=c10d",
            "--batch-size",
            str(self.config.batch_size),
            "--task",
            "drugclip",
            "--loss",
            "in_batch_softmax",
            "--arch",
            "drugclip",
            "--seed",
            str(self.config.seed),
            "--path",
            str(Path(self.config.checkpoint_path).resolve()),
            "--log-interval",
            "100",
            "--log-format",
            "simple",
            "--max-pocket-atoms",
            str(self.config.max_pocket_atoms),
            "--test-task",
            self.config.task,
        ]
        if self.config.cpu:
            command.append("--cpu")
        if self.config.fp16 and not self.config.cpu:
            # test.sh's own fp16 settings
            command.extend(["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"])
        return command

    def run_native_benchmark(self, python_executable: str | None = None) -> subprocess.CompletedProcess[str]:
        issues = self.validate_environment(python_executable=python_executable)
        if issues:
            raise RuntimeError("DrugCLIP benchmark cannot run:\n- " + "\n- ".join(issues))
        command = self.build_test_command(python_executable=python_executable)
        env = os.environ.copy()
        if self.config.cpu:
            # hide every GPU from the subprocess so DrugCLIP's own cuda check fails and
            # it takes the --cpu path instead
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["NVIDIA_VISIBLE_DEVICES"] = ""
        # Deliberately do NOT capture_output here: DrugCLIP's PCBA eval loop can run
        # for many minutes with no output until it finishes, and capture_output=True
        # buffers everything in memory until the subprocess exits, making it look
        # hung. Inheriting the parent's stdout/stderr streams output live instead.
        # Run from the workdir, not the vendored checkout: DrugCLIP resolves
        # "./data/lit_pcba/..." against the cwd, and the vendored tree is pristine.
        workdir = ensure_drugclip_workdir(self.config.data_dir, self.config.workdir)
        return subprocess.run(
            command,
            cwd=workdir,
            env=env,
            check=False,  # surface DrugCLIP's own exit code instead of raising here
            text=True,
        )
