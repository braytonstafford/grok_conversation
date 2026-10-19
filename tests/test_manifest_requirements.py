"""Guard: manifest openai range must match what CI actually installs.

Without this, raising the floor above the min-HA pin (as in 1.10.1) still
passes every CI check because legs install openai from explicit pins, not
from the manifest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import openai
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "custom_components" / "grok_conversation" / "manifest.json"
REQ_MIN_PATH = ROOT / "requirements_test_min.txt"

_OPENAI_EQ_LINE = re.compile(r"^openai==([^\s#]+)", re.MULTILINE)


def _manifest_openai_requirement() -> Requirement:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    reqs = data.get("requirements") or []
    openai_reqs = [Requirement(r) for r in reqs if Requirement(r).name == "openai"]
    assert len(openai_reqs) == 1, f"expected one openai requirement, got {reqs!r}"
    return openai_reqs[0]


def _manifest_openai_specifier() -> SpecifierSet:
    return _manifest_openai_requirement().specifier


def _min_requirements_openai_pin() -> str:
    text = REQ_MIN_PATH.read_text(encoding="utf-8")
    match = _OPENAI_EQ_LINE.search(text)
    assert match, f"no openai== pin found in {REQ_MIN_PATH}"
    return match.group(1)


def _specifier_lower_bound(spec: SpecifierSet) -> str:
    ge = [s.version for s in spec if s.operator == ">="]
    assert len(ge) == 1, f"expected one >= lower bound in {spec}, found {ge!r}"
    return ge[0]


def test_installed_openai_satisfies_manifest_requirement() -> None:
    """Installed openai must be inside the manifest SpecifierSet (both CI legs)."""
    spec = _manifest_openai_specifier()
    version = openai.__version__
    assert version in spec, (
        f"installed openai {version} is outside manifest requirement {spec}; "
        f"CI pins must stay compatible with custom_components/.../manifest.json"
    )


def test_min_requirements_pin_matches_manifest_lower_bound() -> None:
    """requirements_test_min.txt openai==X must equal the manifest >=X floor."""
    pin = _min_requirements_openai_pin()
    floor = _specifier_lower_bound(_manifest_openai_specifier())
    assert pin == floor, (
        f"requirements_test_min.txt pins openai=={pin} but manifest lower "
        f"bound is >={floor}; they must match so the min-HA leg exercises "
        f"the declared floor"
    )
