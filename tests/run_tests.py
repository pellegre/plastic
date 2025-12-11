from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        import pytest  # type: ignore
    except ImportError as exc:
        print(
            "[-] pytest is required to run the test suite. install with `pip install pytest`."
        )
        raise SystemExit(1) from exc

    test_dir = Path(__file__).resolve().parent
    sys.exit(pytest.main([str(test_dir)]))


if __name__ == "__main__":
    main()
