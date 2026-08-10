import importlib.util
from pathlib import Path

PATH=Path(__file__).with_name("run_currency_rotation_v6.py")
SPEC=importlib.util.spec_from_file_location("rotation",PATH)
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

def test_pair_orientation():
    pairs={"EUR_USD","USD_JPY"}
    assert MOD.pair_for("EUR","USD",pairs)==("EUR_USD","LONG")
    assert MOD.pair_for("JPY","USD",pairs)==("USD_JPY","SHORT")
    assert MOD.pair_for("EUR","JPY",pairs)==(None,None)

def test_manifest_is_frozen():
    prereg=__import__("json").loads(MOD.PREREG.read_text())
    assert len(MOD.source_files())==28
    assert MOD.manifest_hash(MOD.source_files())==prereg["sorted_pair_sha_manifest_sha256"]
