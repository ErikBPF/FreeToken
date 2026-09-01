from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
import sys
from types import ModuleType

import pytest


_REGISTER_PATH = Path(__file__).parents[2] / "python/freetoken/models/register.py"
_REGISTER_SPEC = util.spec_from_file_location("freetoken_model_register_test", _REGISTER_PATH)
assert _REGISTER_SPEC is not None and _REGISTER_SPEC.loader is not None
_REGISTER_MODULE = util.module_from_spec(_REGISTER_SPEC)
sys.modules[_REGISTER_SPEC.name] = _REGISTER_MODULE
_REGISTER_SPEC.loader.exec_module(_REGISTER_MODULE)
ModelSpec = _REGISTER_MODULE.ModelSpec
get_model_spec = _REGISTER_MODULE.get_model_spec


@dataclass(frozen=True)
class _FakeDistribution:
    name: str


class _FakeEntryPoint:
    def __init__(self, architecture: str, distribution: str, provider: object):
        self.name = architecture
        self.dist = _FakeDistribution(distribution)
        self.value = f"{distribution}:provider"
        self._provider = provider
        self.loaded = False

    def load(self):
        self.loaded = True
        return self._provider


@pytest.fixture
def providers(monkeypatch):
    installed: list[_FakeEntryPoint] = []
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: installed if group == "freetoken.models" else [],
    )
    return installed


def test_allowed_distribution_supplies_model_spec(monkeypatch, providers):
    spec = ModelSpec("external_model", "ExternalForCausalLM")
    provider = _FakeEntryPoint("ExternalForCausalLM", "freetoken-external", spec)
    providers.append(provider)
    monkeypatch.setenv("FREETOKEN_MODEL_PROVIDERS", "freetoken-external")

    assert get_model_spec("ExternalForCausalLM") == spec
    assert provider.loaded


def test_provider_code_is_not_loaded_without_explicit_allowlist(monkeypatch, providers):
    provider = _FakeEntryPoint(
        "ExternalForCausalLM",
        "freetoken-external",
        ModelSpec("external_model", "ExternalForCausalLM"),
    )
    providers.append(provider)
    monkeypatch.delenv("FREETOKEN_MODEL_PROVIDERS", raising=False)

    with pytest.raises(ValueError, match="installed but not allowed.*freetoken-external"):
        get_model_spec("ExternalForCausalLM")

    assert not provider.loaded


def test_multiple_allowed_providers_are_rejected_before_import(monkeypatch, providers):
    first = _FakeEntryPoint(
        "ExternalForCausalLM",
        "provider-one",
        ModelSpec("first", "ExternalForCausalLM"),
    )
    second = _FakeEntryPoint(
        "ExternalForCausalLM",
        "provider-two",
        ModelSpec("second", "ExternalForCausalLM"),
    )
    providers.extend((first, second))
    monkeypatch.setenv("FREETOKEN_MODEL_PROVIDERS", "provider-one,provider-two")

    with pytest.raises(ValueError, match="multiple allowed providers.*provider-one.*provider-two"):
        get_model_spec("ExternalForCausalLM")

    assert not first.loaded
    assert not second.loaded


def test_external_provider_cannot_replace_builtin(monkeypatch, providers):
    provider = _FakeEntryPoint(
        "LlamaForCausalLM",
        "replacement-provider",
        ModelSpec("replacement", "LlamaForCausalLM"),
    )
    providers.append(provider)
    monkeypatch.setenv("FREETOKEN_MODEL_PROVIDERS", "replacement-provider")

    with pytest.raises(ValueError, match="cannot replace built-in.*LlamaForCausalLM"):
        get_model_spec("LlamaForCausalLM")

    assert not provider.loaded


def test_provider_entry_point_must_export_model_spec(monkeypatch, providers):
    providers.append(_FakeEntryPoint("ExternalForCausalLM", "broken-provider", object()))
    monkeypatch.setenv("FREETOKEN_MODEL_PROVIDERS", "broken-provider")

    with pytest.raises(TypeError, match="must export ModelSpec"):
        get_model_spec("ExternalForCausalLM")


def test_example_provider_is_discovered_from_distribution_metadata(monkeypatch, tmp_path):
    freetoken = ModuleType("freetoken")
    freetoken.__path__ = []
    models = ModuleType("freetoken.models")
    models.__path__ = []
    monkeypatch.setitem(sys.modules, "freetoken", freetoken)
    monkeypatch.setitem(sys.modules, "freetoken.models", models)
    monkeypatch.setitem(sys.modules, "freetoken.models.register", _REGISTER_MODULE)
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "examples/model_provider_plugin/src")
    )

    dist_info = tmp_path / "freetoken_example_model_provider-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: freetoken-example-model-provider\nVersion: 0.1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[freetoken.models]\n"
        "ExampleLlamaForCausalLM = freetoken_example_provider:MODEL_SPEC\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(
        "FREETOKEN_MODEL_PROVIDERS", "freetoken-example-model-provider"
    )

    spec = get_model_spec("ExampleLlamaForCausalLM")

    assert spec == ModelSpec("freetoken.models.llama", "LlamaForCausalLM")
