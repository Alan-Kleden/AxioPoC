from memotion_decay.decay import memotion_decay, DecayParams

def test_positive_decay_decreases():
    v0 = 1.0
    assert memotion_decay(v0, 0) == v0
    assert memotion_decay(v0, 10) < v0

def test_negative_decay_uses_lambda_neg():
    p = DecayParams(lambda_pos=0.05, lambda_neg=0.1)
    v_neg = -1.0
    # plus “rapide” avec lambda_neg>lambda_pos → amplitude |v| diminue plus vite
    at_5 = memotion_decay(v_neg, 5, p)
    at_10 = memotion_decay(v_neg, 10, p)
    assert abs(at_10) < abs(at_5)

def test_asymmetry_effect():
    # même |v0| mais lambdas différents → valeurs différentes à t>0
    v_pos = memotion_decay( 1.0, 10, DecayParams(lambda_pos=0.02, lambda_neg=0.05))
    v_neg = memotion_decay(-1.0, 10, DecayParams(lambda_pos=0.02, lambda_neg=0.05))
    # |v_neg| décroît plus vite (lambda_neg > lambda_pos)
    assert abs(v_neg) < abs(v_pos)
