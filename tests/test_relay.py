"""Tests for relay profiles (`cswap relay`)."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli, relay
from claude_swap.relay import Balance, Profile, RelayError

BASE = "https://relay.example.com"


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    with patch("os.geteuid", return_value=1000, create=True), \
         patch.object(sys, "argv", ["claude-swap", "relay", *argv]):
        code = 0
        try:
            cli.main()
        except SystemExit as e:
            code = e.code or 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _usage_response(available: int, granted: int = 1_000_000_000, name: str = "t") -> io.BytesIO:
    body = {
        "code": True,
        "data": {
            "name": name,
            "total_available": available,
            "total_granted": granted,
            "total_used": granted - available,
            "unlimited_quota": False,
            "expires_at": 0,
        },
        "message": "ok",
    }
    return io.BytesIO(json.dumps(body).encode())


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(balances: dict[str, int]):
    """urlopen stub keyed by bearer token -> available quota units."""

    def urlopen(req, timeout=None):
        assert req.full_url == BASE + relay.USAGE_PATH
        token = req.headers["Authorization"].removeprefix("Bearer ")
        if token not in balances:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        return _Resp(_usage_response(balances[token]).getvalue())

    return urlopen


class TestNormalizeBaseUrl:
    @pytest.mark.parametrize("raw", [BASE, BASE + "/", BASE + "/v1", BASE + "/v1/"])
    def test_strips_slash_and_v1(self, raw):
        assert relay.normalize_base_url(raw) == BASE

    def test_rejects_missing_scheme(self):
        with pytest.raises(RelayError):
            relay.normalize_base_url("relay.example.com")


class TestProfiles:
    def test_add_inherits_base_url_and_model(self, temp_home):
        relay.add_profile(None, "a", "sk-a", BASE + "/v1", "claude-opus-5-5")
        p = relay.add_profile(None, "b", "sk-b", None, None)
        assert p == Profile("b", BASE, "sk-b", "claude-opus-5-5")
        assert [x.name for x in relay.load_profiles()] == ["a", "b"]

    def test_first_profile_needs_base_url(self, temp_home):
        with pytest.raises(RelayError, match="--base-url"):
            relay.add_profile(None, "a", "sk-a", None, None)

    def test_duplicate_needs_force(self, temp_home):
        relay.add_profile(None, "a", "sk-a", BASE, None)
        with pytest.raises(RelayError, match="already exists"):
            relay.add_profile(None, "a", "sk-x", BASE, None)
        relay.add_profile(None, "a", "sk-x", BASE, None, force=True)
        assert relay.load_profiles()[0].key == "sk-x"

    def test_store_is_private(self, temp_home):
        relay.add_profile(None, "a", "sk-a", BASE, None)
        if sys.platform != "win32":
            assert relay.profiles_path().stat().st_mode & 0o777 == 0o600

    def test_remove(self, temp_home):
        relay.add_profile(None, "a", "sk-a", BASE, None)
        relay.remove_profile(None, "a")
        assert relay.load_profiles() == []
        with pytest.raises(RelayError, match="No relay profile"):
            relay.remove_profile(None, "a")


class TestBalance:
    def test_converts_quota_units_to_usd(self):
        p = Profile("a", BASE, "sk-a")
        with patch("urllib.request.urlopen", _fake_urlopen({"sk-a": 450_000_000})):
            b = relay.fetch_balance(p)
        assert b.available_usd == pytest.approx(900.0)
        assert b.granted_usd == pytest.approx(2000.0)

    def test_401_is_reported(self):
        with patch("urllib.request.urlopen", _fake_urlopen({})):
            with pytest.raises(RelayError, match="401"):
                relay.fetch_balance(Profile("a", BASE, "sk-bad"))

    def test_pick_best_prefers_most_available(self):
        ps = [Profile(n, BASE, f"sk-{n}") for n in "abc"]
        bs = [
            Balance("a", 10, 5, 5, False, 0),
            RelayError("down"),
            Balance("c", 10, 1, 9, False, 0),
        ]
        assert relay.pick_best(ps, bs)[0].name == "c"

    def test_pick_best_skips_expired_and_empty(self):
        ps = [Profile(n, BASE, f"sk-{n}") for n in "ab"]
        bs = [Balance("a", 10, 0, 10, False, 1), Balance("b", 10, 10, 0, False, 0)]
        with pytest.raises(RelayError, match="quota"):
            relay.pick_best(ps, bs)


class TestClaudeSettings:
    def test_use_merges_and_off_restores(self, temp_home):
        path = temp_home / ".claude" / "settings.json"
        path.write_text(json.dumps({"theme": "dark", "env": {"FOO": "1"}}))
        p = Profile("a", BASE, "sk-a", "claude-opus-5-5")

        relay.apply_to_claude_settings(p)
        data = json.loads(path.read_text())
        assert data["theme"] == "dark"
        assert data["env"] == {
            "FOO": "1",
            "ANTHROPIC_BASE_URL": BASE,
            "ANTHROPIC_AUTH_TOKEN": "sk-a",
            "ANTHROPIC_MODEL": "claude-opus-5-5",
        }
        assert relay.active_profile_name([p]) == "a"

        relay.apply_to_claude_settings(None)
        assert json.loads(path.read_text()) == {"theme": "dark", "env": {"FOO": "1"}}

    def test_switching_drops_stale_model(self, temp_home):
        relay.apply_to_claude_settings(Profile("a", BASE, "sk-a", "m"))
        relay.apply_to_claude_settings(Profile("b", BASE, "sk-b"))
        env = json.loads((temp_home / ".claude" / "settings.json").read_text())["env"]
        assert "ANTHROPIC_MODEL" not in env
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-b"

    def test_invalid_json_is_not_overwritten(self, temp_home):
        path = temp_home / ".claude" / "settings.json"
        path.write_text("{oops")
        with pytest.raises(RelayError, match="not valid JSON"):
            relay.apply_to_claude_settings(Profile("a", BASE, "sk-a"))
        assert path.read_text() == "{oops"

    def test_writes_through_symlink(self, temp_home, tmp_path):
        if sys.platform == "win32":
            pytest.skip("symlinks")
        real = tmp_path / "dotfiles-settings.json"
        real.write_text("{}")
        link = temp_home / ".claude" / "settings.json"
        link.symlink_to(real)
        relay.apply_to_claude_settings(Profile("a", BASE, "sk-a"))
        assert link.is_symlink()
        assert json.loads(real.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-a"


class TestCli:
    def test_add_list_use_auto(self, temp_home, capsys):
        assert _run(["add", "blue-az-0", "sk-zero", "--base-url", BASE + "/v1",
                     "--model", "claude-opus-5-5"], capsys)[0] == 0
        assert _run(["add", "blue-az-1", "sk-one"], capsys)[0] == 0

        fake = _fake_urlopen({"sk-zero": 100_000_000, "sk-one": 300_000_000})
        with patch("urllib.request.urlopen", fake):
            code, out, _ = _run(["list", "--json"], capsys)
            assert code == 0
            data = json.loads(out)
            assert [p["balance"]["availableUsd"] for p in data["profiles"]] == [200.0, 600.0]
            assert "sk-one" not in out  # keys are masked

            code, out, _ = _run(["use", "auto"], capsys)
            assert code == 0 and "blue-az-1" in out

        env = json.loads((temp_home / ".claude" / "settings.json").read_text())["env"]
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-one"
        assert env["ANTHROPIC_MODEL"] == "claude-opus-5-5"

    def test_run_execs_claude_with_settings(self, temp_home, capsys):
        _run(["add", "a", "sk-a", "--base-url", BASE], capsys)
        with patch("shutil.which", return_value="/bin/claude"), \
             patch("os.execvp", side_effect=SystemExit(0)) as execvp, \
             patch("subprocess.run") as run:
            run.return_value.returncode = 0
            code, _, _ = _run(["run", "a", "--", "--resume"], capsys)
        assert code == 0
        argv = (execvp.call_args or run.call_args).args[-1]
        assert argv[:2] == ["/bin/claude", "--settings"]
        assert argv[-1] == "--resume"
        assert json.loads(Path(argv[2]).read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-a"

    def test_unknown_profile_errors(self, temp_home, capsys):
        _run(["add", "a", "sk-a", "--base-url", BASE], capsys)
        code, _, err = _run(["use", "nope"], capsys)
        assert code == 1 and "No relay profile" in err
