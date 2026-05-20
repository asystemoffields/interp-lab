import inspect
import json
from pathlib import Path

import interp_lab
from interp_lab.contracts import PUBLIC_API_EXPORTS, PUBLIC_API_SIGNATURES, SCHEMA_CONTRACTS


def test_public_api_exports_match_contract():
    assert interp_lab.__all__ == PUBLIC_API_EXPORTS
    for name in PUBLIC_API_EXPORTS:
        assert hasattr(interp_lab, name), name


def test_public_api_contract_is_machine_readable():
    contract = interp_lab.public_api_contract()

    assert contract["schema_version"] == "interp-lab.public_api_contract.v1"
    assert contract["package"] == "interp-lab"
    assert contract["exports"] == interp_lab.__all__
    assert contract["schemas"] == SCHEMA_CONTRACTS
    assert contract["signatures"] == PUBLIC_API_SIGNATURES
    assert json.loads(json.dumps(contract)) == contract


def test_public_api_signature_contract_is_current():
    for function_name, expected_parameters in PUBLIC_API_SIGNATURES.items():
        function = getattr(interp_lab, function_name)
        actual_parameters = list(inspect.signature(function).parameters)
        for parameter in expected_parameters:
            assert parameter in actual_parameters, f"{function_name} missing {parameter}"
        assert expected_parameters == list(dict.fromkeys(expected_parameters)), function_name


def test_public_api_contract_covers_full_stable_workflow_signatures():
    for function_name in ("inspect", "intervene", "train_sae"):
        assert PUBLIC_API_SIGNATURES[function_name] == list(inspect.signature(getattr(interp_lab, function_name)).parameters)


def test_schema_contracts_stay_versioned():
    assert SCHEMA_CONTRACTS["scale_plan"] == "interp-lab.scale_plan.v2"
    for key, schema in SCHEMA_CONTRACTS.items():
        assert schema.startswith("interp-lab."), key
        assert schema.rsplit(".", maxsplit=1)[-1].startswith("v"), key


def test_real_model_demo_manifests_follow_public_schema():
    demo_dir = Path("examples/real_model_demos")
    manifests = sorted(demo_dir.glob("*.json"))

    assert len(manifests) >= 3
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_CONTRACTS["real_model_demo"]
        assert Path(payload["doc"]).exists()
        assert len(payload["commands"]) >= 3
        assert len(payload["expected_artifacts"]) >= 3
        for command in payload["commands"]:
            assert command["name"]
            assert command["argv"]
        for artifact in payload["expected_artifacts"]:
            assert artifact["path"]
            assert artifact["kind"]
            assert artifact["why_it_matters"]
            assert artifact["interpretation_notes"]
