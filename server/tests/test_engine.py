import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app')))
from server.api.engine import get_engine

def test_engine_loads():
    engine = get_engine()
    assert engine is not None
    assert hasattr(engine, 'forward')
