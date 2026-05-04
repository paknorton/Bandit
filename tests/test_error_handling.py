"""Tests for error handling: BanditError exception and CLI integration."""

import ast
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from Bandit.exceptions import BanditError


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


# Feature: error-handling-cleanup, Property 1: BanditError construction round-trip
class TestPropertyBanditErrorConstructionRoundTrip:
    """Property 1: For any integer exit_code and any non-empty string message,
    constructing BanditError(message, exit_code) produces an object where
    error.exit_code == exit_code and str(error) == message.

    **Validates: Requirements 1.2, 1.3, 1.4**
    """

    @given(
        exit_code=st.integers(),
        message=st.text(min_size=1),
    )
    @settings(max_examples=100)
    def test_construction_round_trip(self, exit_code, message):
        """BanditError preserves exit_code and message through construction."""
        error = BanditError(message, exit_code)
        assert error.exit_code == exit_code, (
            f"Expected exit_code={exit_code}, got {error.exit_code}"
        )
        assert str(error) == message, (
            f"Expected str(error)={message!r}, got {str(error)!r}"
        )


# ---------------------------------------------------------------------------
# Unit tests — BanditError basics
# ---------------------------------------------------------------------------


class TestBanditErrorUnit:
    """Unit tests for BanditError construction and behavior.

    Validates: Requirements 1.1, 1.4, 1.5
    """

    def test_is_subclass_of_exception(self):
        """BanditError should be a subclass of Exception."""
        assert issubclass(BanditError, Exception)

    def test_default_exit_code_is_one(self):
        """When no exit_code is provided, it should default to 1."""
        error = BanditError("something went wrong")
        assert error.exit_code == 1

    def test_str_returns_message(self):
        """str(error) should return the message passed at construction."""
        error = BanditError("file not found", exit_code=2)
        assert str(error) == "file not found"


# ---------------------------------------------------------------------------
# Static inspection — no bare exit() calls
# ---------------------------------------------------------------------------


class TestNoBareExitCalls:
    """Static inspection test verifying no bare exit() calls remain in bandit.py.

    Validates: Requirement 2.7
    """

    @staticmethod
    def _find_bare_exit_calls(source: str) -> list[int]:
        """Parse *source* with AST and return line numbers of bare ``exit()`` calls.

        A "bare exit()" is a call whose function node is a plain ``Name``
        with ``id == 'exit'``.  Calls to ``sys.exit`` (represented as an
        ``Attribute`` node with ``attr == 'exit'``) are intentionally
        excluded.
        """
        tree = ast.parse(source)
        bare_exit_lines: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Bare exit(): the function is just a Name node with id "exit"
                if isinstance(func, ast.Name) and func.id == "exit":
                    bare_exit_lines.append(node.lineno)
        return bare_exit_lines

    def test_no_bare_exit_calls_in_bandit_py(self):
        """Bandit/bandit.py must contain zero bare exit() calls."""
        bandit_py = Path(__file__).resolve().parent.parent / "Bandit" / "bandit.py"
        source = bandit_py.read_text(encoding="utf-8")

        bare_exit_lines = self._find_bare_exit_calls(source)

        assert bare_exit_lines == [], (
            f"Found bare exit() call(s) on line(s) {bare_exit_lines} in Bandit/bandit.py. "
            f"Replace them with 'raise BanditError(...)' per the design."
        )


# ---------------------------------------------------------------------------
# Property-based tests — CLI entry point
# ---------------------------------------------------------------------------


# Feature: error-handling-cleanup, Property 2: CLI entry point translates BanditError to sys.exit
class TestPropertyCLIEntryPointBanditErrorTranslation:
    """Property 2: For any BanditError with an arbitrary integer exit_code
    and string message, when app() raises that error, main() calls sys.exit()
    with exactly that exit_code.

    **Validates: Requirements 3.1, 3.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**
    """

    @given(
        exit_code=st.integers(),
        message=st.text(min_size=1),
    )
    @settings(max_examples=100, deadline=None)
    def test_main_translates_bandit_error_to_sys_exit(self, exit_code, message):
        """main() catches BanditError from app() and calls sys.exit(exit_code)."""
        import pytest
        from unittest.mock import patch

        from Bandit.bandit import main

        error = BanditError(message, exit_code)

        with patch("Bandit.bandit.app", side_effect=error):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == exit_code, (
                f"Expected sys.exit({exit_code}), got sys.exit({exc_info.value.code})"
            )


# ---------------------------------------------------------------------------
# Unit tests — main() behavior
# ---------------------------------------------------------------------------


class TestMainBehavior:
    """Unit tests for main() success and error-logging paths.

    Validates: Requirements 3.1, 3.2, 3.3
    """

    def test_main_no_sys_exit_on_success(self):
        """When app() succeeds, main() should return normally without SystemExit.

        **Validates: Requirement 3.2**
        """
        from unittest.mock import patch

        from Bandit.bandit import main

        with patch("Bandit.bandit.app"):
            # If main() calls sys.exit(), pytest will see a SystemExit exception.
            # A successful run should complete without raising anything.
            main()

    def test_main_logs_error_before_exit(self):
        """When app() raises BanditError, main() logs the message at ERROR level.

        **Validates: Requirements 3.1, 3.3**
        """
        import pytest
        from unittest.mock import patch

        from Bandit.bandit import main

        error = BanditError("test msg", exit_code=42)

        with patch("Bandit.bandit.app", side_effect=error), \
             patch("Bandit.bandit.bandit_log") as mock_log:
            with pytest.raises(SystemExit):
                main()

            mock_log.error.assert_called_once_with("test msg")
