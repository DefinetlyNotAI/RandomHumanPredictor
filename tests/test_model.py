import math
import pytest
from human_randomness.model import OnlinePredictor
from human_randomness.evaluation import evaluate
from human_randomness.storage import load, save, profile_path
from human_randomness.cli import main


def test_prediction_is_normalized_and_has_no_zeroes():
    model = OnlinePredictor()
    for value in [37, 55, 37, 73, 12, 37, 19]:
        pred = model.predict()
        assert math.isclose(sum(pred.probabilities), 1, abs_tol=1e-10)
        assert all(p > 0 for p in pred.probabilities)
        assert len(set(pred.top)) == 5
        assert math.isclose(sum(pred.weights.values()), 1, abs_tol=1e-10)
        model.learn(value)


def test_learning_changes_distribution():
    model = OnlinePredictor()
    initial = model.predict().probabilities
    for _ in range(50):
        model.learn(37)
    assert model.predict().probabilities[37] > initial[37]


def test_model_roundtrip_replay():
    choices = [33, 41, 33, 99, 0, 33, 91]
    a = OnlinePredictor(choices)
    b = OnlinePredictor()
    for choice in choices:
        b.learn(choice)
    assert a.predict().probabilities == b.predict().probabilities


def test_chronological_evaluation_and_baseline():
    score = evaluate([37] * 30)
    assert score["rounds"] == 30
    assert 0 <= score["top1"] <= 1
    assert score["baseline_log_loss"] == pytest.approx(math.log(101))
    assert score["log_loss"] < score["baseline_log_loss"]


def test_invalid_numbers():
    model = OnlinePredictor()
    for bad in (-1, 101, 1.5, True, "37"):
        with pytest.raises(ValueError):
            model.learn(bad)


def test_storage_and_profile_path(tmp_path):
    data = {"version": 1, "rounds": [{"actual": 24}]}
    save("alice", data, tmp_path)
    assert load("alice", tmp_path) == data
    for name in ("../oops", "", "a/b"):
        with pytest.raises(ValueError):
            profile_path(name, tmp_path)


def test_cli_import_stats(tmp_path, capsys):
    path = tmp_path / "numbers.txt"
    path.write_text("37, 42\n15\n")
    assert main(["--home", str(tmp_path), "import", str(path), "--profile", "t"]) == 0
    assert main(["--home", str(tmp_path), "stats", "--profile", "t", "--json"]) == 0
    assert '"rounds": 3' in capsys.readouterr().out


def test_cli_rejects_bad_import_without_partial_write(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("37 42 invalid 100")
    assert main(["--home", str(tmp_path), "import", str(path)]) == 1
    assert load("default", tmp_path)["rounds"] == []
