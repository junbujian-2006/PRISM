import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_only_selected_modules_are_exported():
    import prism

    assert set(prism.__all__) == {"RDPE", "DSR", "HaarDWT2D", "HaarIDWT2D", "HWD", "HWU"}
    assert {path.name for path in (ROOT / "prism").glob("*.py")} == {
        "__init__.py", "dsr.py", "rdpe.py", "wavelet.py"
    }


def test_modules_have_only_required_imports():
    for path in (ROOT / "prism").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] == "torch" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    assert node.module.split(".")[0] in {"torch", "einops"}
