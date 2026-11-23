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

    def test_context_overrides_restore_previous_values_when_switching_and_turning_off(self, temp_home):
        path = temp_home / ".claude" / "settings.json"
        original_env = {relay.CONTEXT_WINDOW_ENV: "200000", "OTHER": "value"}
        path.write_text(json.dumps({"autoCompactWindow": 180000, "env": original_env}))
        tuned = Profile("a", BASE, "shared", "glm", 1000000, 850000, 128000)
        other = Profile("b", BASE, "shared", "kimi", auto_compact="off")

        relay.apply_to_claude_settings(tuned)
        env = json.loads(path.read_text())["env"]
        assert env[relay.CONTEXT_WINDOW_ENV] == "1000000"
        assert env[relay.AUTO_COMPACT_WINDOW_ENV] == "850000"
        assert env[relay.MAX_OUTPUT_TOKENS_ENV] == "128000"
        relay.apply_to_claude_settings(tuned)

        relay.apply_to_claude_settings(other)
        env = json.loads(path.read_text())["env"]
        assert env[relay.CONTEXT_WINDOW_ENV] == "200000"
        assert relay.AUTO_COMPACT_WINDOW_ENV not in env
        assert relay.MAX_OUTPUT_TOKENS_ENV not in env
        assert env[relay.DISABLE_AUTO_COMPACT_ENV] == "1"
        assert relay.active_profile_name([tuned, other]) == "b"

        relay.apply_to_claude_settings(None)
        assert json.loads(path.read_text()) == {"autoCompactWindow": 180000, "env": original_env}

    def test_context_switch_preserves_edits_made_while_relay_is_active(self, temp_home):
        path = temp_home / ".claude" / "settings.json"
        relay.apply_to_claude_settings(Profile("a", BASE, "sk-a", context_window=200000))
        data = json.loads(path.read_text())
        data["env"][relay.CONTEXT_WINDOW_ENV] = "123000"
        path.write_text(json.dumps(data))

        relay.apply_to_claude_settings(None)
        assert json.loads(path.read_text())["env"] == {relay.CONTEXT_WINDOW_ENV: "123000"}

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

    def test_list_distinguishes_missing_balance_api_from_unlimited(self, temp_home, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "140")
        metered = Profile("blue-az", BASE, "shared-key", "claude-opus-5-5")
        another_model = Profile("blue-az-other", BASE, "shared-key", "claude-sonnet-5")
        unreported = Profile("blue-csi", "https://other.example.com", "sk-demo-private-4321", "GLM-5.3")
        unlimited = Profile("blue-unlimited", BASE, "unlimited-key", "kimi-k3")
        relay.save_profiles([metered, another_model, unreported, unlimited])
        relay.apply_to_claude_settings(unreported)

        def urlopen(req, timeout=None):
            token = req.headers["Authorization"].removeprefix("Bearer ")
            if token == unreported.key:
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
            payload = json.load(_usage_response(100_000_000))
            payload["data"]["unlimited_quota"] = token == unlimited.key
            return _Resp(json.dumps(payload).encode())

        with patch("urllib.request.urlopen", urlopen):
            code, out, _ = _run(["list"], capsys)
            assert code == 0
            assert all(label in out for label in ("PROFILE", "KEY", "MODEL", "GATEWAY", "BALANCE"))
            csi_row = next(line for line in out.splitlines() if unreported.name in line)
            assert csi_row.split() == [
                "*", "blue-csi", "sk-demo…4321", "GLM-5.3", "other.example.com", "No", "balance", "API",
            ]
            metered_row = next(line for line in out.splitlines() if line.split()[:1] == [metered.name])
            assert "$200.00 left of  $2,000.00  (90.0% used)  expires never" in metered_row
            assert metered_row.index("$200.00") == csi_row.index("No balance API")
            assert metered_row.index("$200.00") == out.splitlines()[0].index("BALANCE")
            assert "Known available: $200.00" in out
            unlimited_row = next(line for line in out.splitlines() if unlimited.name in line)
            assert unlimited_row.rstrip().endswith("unlimited  expires never")
            assert all(p.key not in out for p in (metered, unreported, unlimited))

            code, out, _ = _run(["list", "--json"], capsys)
            assert code == 0
            assert json.loads(out)["profiles"][2]["balance"] == {"error": "HTTP 404"}

            assert _run(["use", "auto"], capsys)[0] == 0
            assert relay.active_profile_name(relay.load_profiles()) == unlimited.name

    def test_list_without_reported_balances_does_not_print_zero_total(self, temp_home, capsys):
        relay.save_profiles([Profile("blue-csi", BASE, "sk-private", "kimi-k3")])
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(BASE, 404, "Not Found", {}, None)):
            code, out, _ = _run(["list"], capsys)
        assert code == 0
        assert "No balance API" in out
        assert "Known available" not in out

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

    def test_add_context_options_are_opt_in_and_force_can_clear_them(self, temp_home, capsys):
        assert _run([
            "add", "a", "sk-a", "--base-url", BASE, "--context-window", "1M",
            "--auto-compact", "850k", "--max-output-tokens", "128000",
        ], capsys)[0] == 0
        assert _run(["add", "b", "sk-b"], capsys)[0] == 0
        assert [p.context_options() for p in relay.load_profiles()] == [
            {"contextWindow": 1000000, "autoCompact": 850000, "maxOutputTokens": 128000}, {},
        ]

        assert _run([
            "add", "a", "sk-new", "--force", "--context-window", "auto", "--auto-compact", "off",
        ], capsys)[0] == 0
        tuned, plain = relay.load_profiles()
        assert tuned.context_options() == {"autoCompact": "off", "maxOutputTokens": 128000}
        assert plain.context_options() == {}

        assert _run([
            "add", "a", "sk-new", "--force", "--auto-compact", "auto", "--max-output-tokens", "auto",
        ], capsys)[0] == 0
        assert relay.load_profiles()[0].context_options() == {}

    @pytest.mark.parametrize("flag,value", [
        ("--context-window", "0"), ("--auto-compact", "99k"),
        ("--auto-compact", "2M"), ("--max-output-tokens", "-1"),
    ])
    def test_add_rejects_invalid_context_options(self, temp_home, capsys, flag, value):
        code, _, err = _run(["add", "a", "sk-a", "--base-url", BASE, flag, value], capsys)
        assert code == 2 and flag in err
        assert relay.load_profiles() == []

    def test_run_uses_requested_context_without_inheriting_active_relay(self, temp_home, capsys, monkeypatch):
        path = temp_home / ".claude" / "settings.json"
        path.write_text(json.dumps({"env": {relay.CONTEXT_WINDOW_ENV: "200000"}}))
        monkeypatch.delenv(relay.DISABLE_AUTO_COMPACT_ENV, raising=False)
        assert _run([
            "add", "a", "sk-a", "--base-url", BASE, "--context-window", "1M", "--auto-compact", "off",
        ], capsys)[0] == 0
        assert _run(["add", "b", "sk-b", "--max-output-tokens", "32k"], capsys)[0] == 0
        assert _run(["use", "a"], capsys)[0] == 0
        active_settings = path.read_text()

        with patch.object(relay, "exec_claude") as launch:
            assert _run(["run", "b"], capsys)[0] == 0
        env = json.loads(launch.call_args.args[0].read_text())["env"]
        assert env[relay.CONTEXT_WINDOW_ENV] == "200000"
        assert env[relay.DISABLE_AUTO_COMPACT_ENV] == ""
        assert env[relay.MAX_OUTPUT_TOKENS_ENV] == "32000"
        assert path.read_text() == active_settings

    def test_unknown_profile_errors(self, temp_home, capsys):
        _run(["add", "a", "sk-a", "--base-url", BASE], capsys)
        code, _, err = _run(["use", "nope"], capsys)
        assert code == 1 and "No relay profile" in err
