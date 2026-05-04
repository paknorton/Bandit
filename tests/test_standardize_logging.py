"""Tests for standardized logging in bandit.py.

These tests use AST inspection to verify that prohibited output patterns
(bare print(), sys.stdout.write, sys.stdout.flush, import sys) are absent
from the bandit.py module.
"""

import ast
from pathlib import Path

import pytest


BANDIT_PY = Path(__file__).parent.parent / 'Bandit' / 'bandit.py'


@pytest.fixture(scope='module')
def bandit_tree():
    """Parse bandit.py into an AST tree."""
    source = BANDIT_PY.read_text()
    return ast.parse(source)


@pytest.fixture(scope='module')
def extract_func(bandit_tree):
    """Get the AST node for the extract() function."""
    for node in ast.walk(bandit_tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'extract':
            return node
    pytest.fail('Could not find extract() function in bandit.py')


class TestProhibitedPatterns:
    """AST-based tests verifying prohibited output patterns are absent."""

    def test_no_bare_print_in_extract(self, extract_func):
        """The extract function should not contain any bare print() calls.

        Validates: Requirements 1.3, 2.2
        """
        for node in ast.walk(extract_func):
            if isinstance(node, ast.Call):
                func = node.func
                # Check for bare print() call
                if isinstance(func, ast.Name) and func.id == 'print':
                    pytest.fail(
                        f'Found bare print() call at line {node.lineno} in extract()'
                    )

    def test_no_sys_stdout_write(self, bandit_tree):
        """bandit.py should not contain any sys.stdout.write() calls.

        Validates: Requirement 3.1
        """
        for node in ast.walk(bandit_tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == 'write'
                        and isinstance(func.value, ast.Attribute)
                        and func.value.attr == 'stdout'
                        and isinstance(func.value.value, ast.Name)
                        and func.value.value.id == 'sys'):
                    pytest.fail(
                        f'Found sys.stdout.write() call at line {node.lineno}'
                    )

    def test_no_sys_stdout_flush(self, bandit_tree):
        """bandit.py should not contain any sys.stdout.flush() calls.

        Validates: Requirement 3.2
        """
        for node in ast.walk(bandit_tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == 'flush'
                        and isinstance(func.value, ast.Attribute)
                        and func.value.attr == 'stdout'
                        and isinstance(func.value.value, ast.Name)
                        and func.value.value.id == 'sys'):
                    pytest.fail(
                        f'Found sys.stdout.flush() call at line {node.lineno}'
                    )

    def test_no_import_sys(self, bandit_tree):
        """bandit.py should not import the sys module.

        Validates: Requirement 6.1
        """
        for node in ast.walk(bandit_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'sys':
                        pytest.fail(
                            f'Found "import sys" at line {node.lineno}'
                        )
