"""Audit the workflow's native unittest selection on the temporary check branch."""
import hashlib
import json
import os
from pathlib import Path
import importlib
import sys


def main():
    runner = importlib.import_module("step1_original_unittests")
    load = runner.load_selected

    def load_and_audit(root, filenames):
        import torch

        suite = load(root, filenames)
        raw = (root / "step1-manifest.json").read_bytes()
        files = {}
        for name in json.loads(raw)["files"]:
            module = "_gpu_tests." + name.removeprefix("test/").removesuffix(".py").replace("/", ".")
            files[name] = {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                           "selected_device": getattr(sys.modules.get(module), "DEVICE_NAME", None)}
        available = torch.cuda.is_available()
        audit = {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "run_flag": os.getenv("RUN_SUPPLEMENT_API_TESTS"),
                 "files": files, "environment": {"python": sys.version, "torch": str(torch.__version__),
                 "torch_git_version": torch.version.git_version, "cuda_build": torch.version.cuda,
                 "cuda_available": available, "gpu_name": torch.cuda.get_device_name(0) if available else None}}
        (Path(os.environ["GPU_TEST_OUTPUT_DIR"]) / "step1-audit.json").write_text(json.dumps(audit), encoding="utf-8")
        return suite

    runner.load_selected = load_and_audit
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
