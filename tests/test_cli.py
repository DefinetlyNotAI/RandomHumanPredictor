"""Smoke tests for the Rich CLI and machine-readable output."""
import json
from human_randomness.cli import main


def test_stats_json_unformatted(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "stats", "--profile", "fresh", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rounds"] == 0


def test_play_saves_hidden_prediction(tmp_path, monkeypatch):
    inputs = iter(["50", "q"])
    monkeypatch.setattr("human_randomness.cli.Prompt.ask", lambda *a, **kw: next(inputs))
    assert main(["--home", str(tmp_path), "play", "--profile", "rich"]) == 0
    from human_randomness.storage import load
    saved = load("rich", tmp_path)
    record = saved["rounds"][0]
    assert record["actual"] == 50
    assert len(record["commitment"]) == 64
    assert "nonce" in record
