import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("Skipping tests for Python 3.14 and above", allow_module_level=True)
