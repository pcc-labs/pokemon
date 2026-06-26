"""Tests for run_10_agents.py — 100% coverage."""

import json
import os
import runpy
import subprocess as sp
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import run_10_agents as mod
from run_10_agents import (
    PARAM_VARIANTS,
    MAX_TURNS,
    score,  # re-exported from evolve
    run_one_agent,
    main,
)


# ── PARAM_VARIANTS validation ────────────────────────────────────────


class TestParamVariants:
    def test_has_12_variants(self):
        assert len(PARAM_VARIANTS) == 12

    def test_all_variants_have_required_keys(self):
        required = {"stuck_threshold", "door_cooldown", "waypoint_skip_distance",
                     "axis_preference_map_0", "label",
                     "bt_max_snapshots", "bt_restore_threshold",
                     "bt_max_attempts", "bt_snapshot_interval"}
        for i, variant in enumerate(PARAM_VARIANTS):
            missing = required - set(variant.keys())
            assert not missing, f"Variant {i} ({variant.get('label', '?')}) missing: {missing}"

    def test_labels_unique(self):
        labels = [v["label"] for v in PARAM_VARIANTS]
        assert len(labels) == len(set(labels)), "Duplicate labels found"


# ── score() ───────────────────────────────────────────────────────────


class TestScore:
    def test_zero_fitness(self):
        assert score({}) == 0.0

    def test_positive_score(self):
        f = {
            "final_map_id": 1,
            "badges": 1,
            "party_size": 1,
            "battles_won": 5,
            "stuck_count": 2,
            "turns": 100,
        }
        expected = 1000 + 5000 + 500 + 50 - 10 - 10.0
        assert score(f) == expected

    def test_stuck_penalizes(self):
        base = {"final_map_id": 1}
        stuck = {"final_map_id": 1, "stuck_count": 100}
        assert score(stuck) < score(base)


# ── run_one_agent() ───────────────────────────────────────────────────


class TestRunOneAgent:
    def _make_fitness(self, **overrides):
        f = {"final_map_id": 1, "badges": 0, "party_size": 1,
             "battles_won": 3, "stuck_count": 2, "turns": 50}
        f.update(overrides)
        return f

    def test_success(self):
        fitness = self._make_fitness()

        def mock_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout=json.dumps(fitness))

        params = {"stuck_threshold": 8, "door_cooldown": 4,
                  "waypoint_skip_distance": 3, "axis_preference_map_0": "y",
                  "label": "test_label"}

        with patch("run_10_agents.subprocess.run", side_effect=mock_run):
            result = run_one_agent("/fake/rom.gb", params, 0, use_paper=False)

        assert result["agent_id"] == 0
        assert result["label"] == "test_label"
        assert result["fitness"] == fitness
        assert result["score"] == score(fitness)
        assert result["returncode"] == 0
        assert "error" not in result
        # label should be stripped from params passed to agent
        assert "label" not in result["params"]

    def test_label_defaults_to_agent_id(self):
        fitness = self._make_fitness()

        def mock_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout=json.dumps(fitness))

        params = {"stuck_threshold": 8, "door_cooldown": 4,
                  "waypoint_skip_distance": 3, "axis_preference_map_0": "y"}

        with patch("run_10_agents.subprocess.run", side_effect=mock_run):
            result = run_one_agent("/fake/rom.gb", params, 7, use_paper=False)

        assert result["label"] == "agent_7"

    def test_timeout_returns_error(self):
        params = {"stuck_threshold": 8, "label": "timeout_test"}

        with patch("run_10_agents.subprocess.run",
                   side_effect=sp.TimeoutExpired("cmd", 300)):
            result = run_one_agent("/fake/rom.gb", params, 1, use_paper=False)

        assert result["score"] == -999
        assert result["fitness"] == {}
        assert "error" in result

    def test_file_not_found_returns_error(self):
        params = {"stuck_threshold": 8, "label": "fnf_test"}

        with patch("run_10_agents.subprocess.run",
                   side_effect=FileNotFoundError("no python")):
            result = run_one_agent("/fake/rom.gb", params, 2, use_paper=False)

        assert result["score"] == -999
        assert "error" in result

    def test_invalid_json_returns_error(self):
        params = {"stuck_threshold": 8, "label": "bad_json"}

        def mock_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="not json at all")

        with patch("run_10_agents.subprocess.run", side_effect=mock_run):
            result = run_one_agent("/fake/rom.gb", params, 3, use_paper=False)

        # _extract_fitness returns {} when no party_size JSON found → score -999
        assert result["score"] == -999

    def test_params_embedded_in_prompt(self):
        fitness = self._make_fitness()
        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock(returncode=0, stdout=json.dumps(fitness))

        params = {"stuck_threshold": 10, "door_cooldown": 6, "label": "env_test"}

        with patch("run_10_agents.subprocess.run", side_effect=mock_run):
            run_one_agent("/fake/rom.gb", params, 0, use_paper=False)

        # EVOLVE_PARAMS is embedded in the prompt string (last cmd arg)
        prompt = captured_cmd[-1]
        assert "EVOLVE_PARAMS" in prompt
        assert '"stuck_threshold": 10' in prompt
        assert "label" not in json.loads(
            prompt.split("EVOLVE_PARAMS='")[1].split("'")[0]
        )

    def test_non_paper_keeps_api_key_in_env(self):
        fitness = self._make_fitness()
        captured_env = {}

        def mock_run(cmd, env=None, **kwargs):
            if env:
                captured_env.update(env)
            return MagicMock(returncode=0, stdout=json.dumps(fitness))

        params = {"stuck_threshold": 8, "label": "env_key_test"}

        with patch("run_10_agents.subprocess.run", side_effect=mock_run), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            run_one_agent("/fake/rom.gb", params, 0, use_paper=False)

        assert captured_env.get("ANTHROPIC_API_KEY") == "test-key"


# ── main() ────────────────────────────────────────────────────────────


class TestMain:
    def test_no_args_exits(self, capsys):
        with patch("sys.argv", ["run_10_agents.py"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        assert "Usage" in capsys.readouterr().out

    def test_rom_not_found_exits(self, capsys):
        with patch("sys.argv", ["run_10_agents.py", "/nonexistent/rom.gb"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        assert "ROM not found" in capsys.readouterr().out

    def test_full_run(self, tmp_path, capsys):
        rom = tmp_path / "test.gb"
        rom.write_bytes(b"\x00" * 100)

        fake_result = {
            "agent_id": 0,
            "label": "test",
            "params": {},
            "fitness": {"final_map_id": 1, "badges": 0, "party_size": 1,
                        "battles_won": 3, "stuck_count": 2, "turns": 50},
            "score": 1530.0,
            "elapsed": 1.0,
            "returncode": 0,
        }

        def mock_run_one_agent(rom_path, params, agent_id, use_paper):
            return dict(fake_result, agent_id=agent_id,
                        label=params.get("label", f"agent_{agent_id}"))

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()

        with patch("sys.argv", ["run_10_agents.py", str(rom)]), \
             patch("run_10_agents.run_one_agent", side_effect=mock_run_one_agent), \
             patch("run_10_agents.ThreadPoolExecutor", ThreadPoolExecutor), \
             patch.object(mod, "SCRIPT_DIR", scripts_dir), \
             patch.object(mod, "WORKSPACE", tmp_path):
            main()

        output = capsys.readouterr().out
        assert "Winner:" in output
        assert "score=" in output

        saved = tmp_path / "pokedex" / "evolve_results.json"
        assert saved.exists()
        data = json.loads(saved.read_text())
        assert len(data) == len(PARAM_VARIANTS)

    def test_error_result_shows_fail(self, tmp_path, capsys):
        rom = tmp_path / "test.gb"
        rom.write_bytes(b"\x00" * 100)

        def mock_run_one_agent(rom_path, params, agent_id, use_paper):
            return {
                "agent_id": agent_id,
                "label": params.get("label", f"agent_{agent_id}"),
                "params": {},
                "fitness": {},
                "score": -999,
                "elapsed": 0.5,
                "error": "timeout",
            }

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()

        with patch("sys.argv", ["run_10_agents.py", str(rom)]), \
             patch("run_10_agents.run_one_agent", side_effect=mock_run_one_agent), \
             patch("run_10_agents.ThreadPoolExecutor", ThreadPoolExecutor), \
             patch.object(mod, "SCRIPT_DIR", scripts_dir), \
             patch.object(mod, "WORKSPACE", tmp_path):
            main()

        output = capsys.readouterr().out
        assert "[FAIL(" in output


# ── __main__ guard ────────────────────────────────────────────────────


class TestMainGuard:
    def test_dunder_main_calls_main(self, tmp_path):
        rom = tmp_path / "test.gb"
        rom.write_bytes(b"\x00" * 100)

        fake_result = {
            "agent_id": 0, "label": "test", "params": {},
            "fitness": {"final_map_id": 0}, "score": 0.0,
            "elapsed": 0.1, "returncode": 0,
        }

        def mock_run_one_agent(rom_path, params, agent_id, use_paper):
            return dict(fake_result, agent_id=agent_id,
                        label=params.get("label", f"agent_{agent_id}"))

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()

        with patch("sys.argv", ["run_10_agents.py", str(rom)]), \
             patch("run_10_agents.run_one_agent", side_effect=mock_run_one_agent), \
             patch("run_10_agents.ThreadPoolExecutor", ThreadPoolExecutor), \
             patch.object(mod, "SCRIPT_DIR", scripts_dir), \
             patch.object(mod, "WORKSPACE", tmp_path):
            runpy.run_path(
                str(Path(mod.__file__).resolve()),
                run_name="__main__",
            )
