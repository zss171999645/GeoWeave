import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_lists_all_easyvolcap_init_packages():
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r"(?ms)^packages\s*=\s*(\[[^\]]*\])", pyproject_text)
    assert match is not None
    configured_packages = set(ast.literal_eval(match.group(1)))

    discovered_packages = {
        init_path.parent.relative_to(REPO_ROOT).as_posix().replace("/", ".")
        for init_path in (REPO_ROOT / "easyvolcap").rglob("__init__.py")
        if "__pycache__" not in init_path.parts
    }

    missing = sorted(discovered_packages - configured_packages)
    assert missing == [], f"Missing packages in pyproject.toml: {missing}"
