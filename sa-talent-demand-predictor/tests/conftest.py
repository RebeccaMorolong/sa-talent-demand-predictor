"""
Shared pytest fixtures.
"""
import sys
from pathlib import Path

# Make sure src/ is importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
