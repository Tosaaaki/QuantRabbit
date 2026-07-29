from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inventory-legacy-strategies.py"
SPEC = importlib.util.spec_from_file_location("legacy_inventory", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aliases_merge_known_strategy_families() -> None:
    assert MODULE._canonical("macro_trendma") == "trend_ma"
    assert MODULE._canonical("ma_cross") == "trend_ma"
    assert MODULE._canonical("scalp_squeeze_pulse_break") == "pulse_break"
    assert MODULE._canonical("pulse_break") == "pulse_break"


def test_vm_evidence_never_emits_secret_values(tmp_path: Path) -> None:
    evidence = tmp_path / "deploy.sh"
    evidence.write_text(
        "\n".join(
            [
                "VM_NAME=quant-worker-01",
                "OANDA_API_TOKEN=do-not-emit-this-value",
                "gcloud compute instances describe quant-worker-01",
                "projects/demo/secrets/oanda-api-token/versions/latest",
                "quant-scalp-tick-imbalance.service",
                "scalp_tick_imbalance",
            ]
        )
    )
    payload = MODULE._safe_vm_evidence([evidence], ["scalp_tick_imbalance"])
    rendered = repr(payload)
    assert "quant-worker-01" in payload["instances"]
    assert "oanda-api-token" in payload["secret_names_only"]
    assert "OANDA_API_TOKEN" in payload["secret_names_only"]
    assert "do-not-emit-this-value" not in rendered
    assert "scalp_tick_imbalance" in payload["worker_links"]


def test_instance_name_requires_terminal_alphanumeric() -> None:
    assert MODULE.NAME_ASSIGN_RE.findall("VM_NAME=qr-\n") == []
    assert MODULE.NAME_ASSIGN_RE.findall("VM_NAME=qr-worker-01\n") == ["qr-worker-01"]
