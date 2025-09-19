from appetition_aversion.regulator import appetition_aversion_score
from common.utils import clamp

def test_appetition_aversion_score_positive():
    assert appetition_aversion_score(2.0, 1.0) == 1.0

def test_appetition_aversion_score_negative():
    assert appetition_aversion_score(1.0, 3.0) == -2.0

def test_clamp_function():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
