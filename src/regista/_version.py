"""Single source of truth for the package version.

Lives in its own module (not ``__init__``) so submodules like session.py can
record it in traces without importing the package root — which would be
circular, since the root imports them. Keep in sync with pyproject.toml.
"""

__version__ = "0.2.0"
