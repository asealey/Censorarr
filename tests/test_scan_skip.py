"""Dependency-free regression tests for automatic-scan skip rules in the worker.

The production module imports Whisper and media tooling, so extract only the
function under test via AST (same approach as test_worker_controls.py).
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "app" / "censorarr.py"


def extract(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        raise AssertionError(f"Missing definition: {name}")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


should_skip = extract("automatic_scan_should_skip")


class AutomaticScanSkipTests(unittest.TestCase):
    def row(self, status):
        return {"fingerprint": "fp1", "config_signature": "sig1", "status": status}

    def test_unchanged_file_awaiting_review_is_not_reanalyzed(self):
        self.assertTrue(should_skip(self.row("awaiting-review"), "fp1", "sig1"))

    def test_unchanged_dry_run_is_not_reanalyzed(self):
        self.assertTrue(should_skip(self.row("dry-run"), "fp1", "sig1"))

    def test_changed_file_awaiting_review_is_reanalyzed(self):
        self.assertFalse(should_skip(self.row("awaiting-review"), "fp2", "sig1"))

    def test_changed_settings_awaiting_review_is_reanalyzed(self):
        self.assertFalse(should_skip(self.row("awaiting-review"), "fp1", "sig2"))

    def test_other_statuses_keep_existing_handling(self):
        for status in ("error", "waiting-subtitle", "waiting-rating", "applied", None):
            self.assertFalse(should_skip(self.row(status), "fp1", "sig1"), status)

    def test_unknown_file_is_processed(self):
        self.assertFalse(should_skip({}, "fp1", "sig1"))


if __name__ == "__main__":
    unittest.main()
