"""Relay profiles: third-party Anthropic-compatible endpoints driven by settings.json.

A relay profile is a (base URL, key, model) triple for a new-api/one-api style
relay. Instead of logging in, cswap hands the triple to Claude Code through a
settings file's ``env`` block (``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN``
/ ``ANTHROPIC_MODEL``):

- ``cswap relay use NAME|auto`` merges that block into ``~/.claude/settings.json``
  so every new Claude Code instance (terminal and IDE) uses the relay;
  ``--off`` removes it again, leaving every other key untouched.
- ``cswap relay run NAME|auto`` writes a per-profile settings file under the
  backup root and execs ``claude --settings <file>``: this terminal only.

Balances come from the relay's token-usage endpoint (``GET
<base>/api/usage/token/`` with the key as a Bearer token), the same call the
relay's ``/balance`` page makes. Quota is reported in raw units of
``QUOTA_PER_USD`` per US dollar.

Profiles live in ``<backup_root>/relay/profiles.json`` (0600). This module is
self-contained on purpose — it only hooks into ``cli.main`` — so the fork
rebases cleanly onto upstream.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from claude_swap import paths
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.fsutil import replace_with_retry
from claude_swap.printer import accent, bolded, dimmed, error, muted, warning
from claude_swap.settings import atomic_write_json

QUOTA_PER_USD = 500_000
USAGE_PATH = "/api/usage/token/"
ENV_KEYS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL")
SCHEMA_VERSION = 1


class RelayError(ClaudeSwitchError):
    """A relay profile operation failed."""


@dataclass(frozen=True)
class Profile:
    name: str
    base_url: str
    key: str
    model: str | None = None

    def to_json(self) -> dict:
        data = {"name": self.name, "baseUrl": self.base_url, "key": self.key}
        if self.model:
            data["model"] = self.model
        return data

    @classmethod
    def from_json(cls, data: dict) -> Profile:
        return cls(
            name=data["name"],
            base_url=data["baseUrl"],
            key=data["key"],
            model=data.get("model") or None,
        )


@dataclass(frozen=True)
class Balance:
    token_name: str
    granted_usd: float
    used_usd: float
    available_usd: float
    unlimited: bool
    expires_at: int  # unix seconds; 0 means never

    @property
    def expired(self) -> bool:
        return self.expires_at > 0 and self.expires_at < datetime.now(timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def relay_dir(root: Path | None = None) -> Path:
    return (root or paths.get_backup_root()) / "relay"


def profiles_path(root: Path | None = None) -> Path:
    return relay_dir(root) / "profiles.json"


def normalize_base_url(url: str) -> str:
    """Strip trailing slashes and a trailing ``/v1``.

    Claude Code appends ``/v1/messages`` itself, and relays usually advertise
    their OpenAI-style ``.../v1`` URL, so both spellings are accepted.
    """
    url = url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    if not url.startswith(("http://", "https://")):
        raise RelayError(f"Base URL must start with http:// or https://: {url!r}")
    return url


def load_profiles(root: Path | None = None) -> list[Profile]:
    path = profiles_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        raise RelayError(f"Cannot read {path}: {e}") from e
    return [Profile.from_json(p) for p in data.get("profiles", [])]


def save_profiles(profiles: list[Profile], root: Path | None = None) -> None:
    atomic_write_json(
        profiles_path(root),
        {"schemaVersion": SCHEMA_VERSION, "profiles": [p.to_json() for p in profiles]},
    )


def find_profile(profiles: list[Profile], name: str) -> Profile:
    for p in profiles:
        if p.name == name:
            return p
    known = ", ".join(p.name for p in profiles) or "none"
    raise RelayError(f"No relay profile named {name!r} (known: {known})")


def add_profile(
    root: Path | None,
    name: str,
    key: str,
    base_url: str | None,
    model: str | None,
    *,
    force: bool = False,
) -> Profile:
    """Add (or with ``force`` replace) a profile.

    ``base_url`` and ``model`` default to the most recently added profile's,
    so a batch of keys for one relay only needs them spelled out once.
    """
    profiles = load_profiles(root)
    key = key.strip()
    if not key:
        raise RelayError("Key must not be empty")
    last = profiles[-1] if profiles else None
    if base_url is None:
        if last is None:
            raise RelayError("--base-url is required for the first relay profile")
        base_url = last.base_url
    if model is None and last is not None:
        model = last.model
    profile = Profile(name=name, base_url=normalize_base_url(base_url), key=key, model=model)

    existing = [i for i, p in enumerate(profiles) if p.name == name]
    if existing and not force:
        raise RelayError(f"Relay profile {name!r} already exists (use --force to replace)")
    if existing:
        profiles[existing[0]] = profile
    else:
        profiles.append(profile)
    save_profiles(profiles, root)
    return profile


def remove_profile(root: Path | None, name: str) -> None:
    profiles = load_profiles(root)
    find_profile(profiles, name)
    save_profiles([p for p in profiles if p.name != name], root)


def mask_key(key: str) -> str:
    return f"{key[:7]}…{key[-4:]}" if len(key) > 14 else "…"


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------


def fetch_balance(profile: Profile, timeout: float = 15) -> Balance:
    req = urllib.request.Request(
        profile.base_url + USAGE_PATH,
        headers={
            "Authorization": f"Bearer {profile.key}",
            "Accept": "application/json",
            "User-Agent": "claude-swap-relay",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RelayError("key invalid, expired, disabled or out of quota (HTTP 401)") from e
        raise RelayError(f"HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RelayError(f"network error: {getattr(e, 'reason', e)}") from e
    except ValueError as e:
        raise RelayError("response is not JSON") from e

    if isinstance(payload, dict) and payload.get("error"):
        err = payload["error"]
        msg = err.get("message", err) if isinstance(err, dict) else err
        raise RelayError(f"relay error: {msg}")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise RelayError("response has no 'data' field")
    return Balance(
        token_name=str(data.get("name", "")),
        granted_usd=data.get("total_granted", 0) / QUOTA_PER_USD,
        used_usd=data.get("total_used", 0) / QUOTA_PER_USD,
        available_usd=data.get("total_available", 0) / QUOTA_PER_USD,
        unlimited=bool(data.get("unlimited_quota")),
        expires_at=int(data.get("expires_at") or 0),
    )


def fetch_all(profiles: list[Profile]) -> list[Balance | RelayError]:
    """Fetch every profile's balance concurrently, preserving order."""

    def one(p: Profile) -> Balance | RelayError:
        try:
            return fetch_balance(p)
        except RelayError as e:
            return e

    if not profiles:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(profiles))) as pool:
        return list(pool.map(one, profiles))


def pick_best(
    profiles: list[Profile], balances: list[Balance | RelayError]
) -> tuple[Profile, Balance]:
    """The usable profile with the most available quota (unlimited wins)."""
    usable = [
        (p, b)
        for p, b in zip(profiles, balances)
        if isinstance(b, Balance) and not b.expired and (b.unlimited or b.available_usd > 0)
    ]
    if not usable:
        raise RelayError("No relay profile has quota left")
    return max(usable, key=lambda pb: (pb[1].unlimited, pb[1].available_usd))


def resolve_target(root: Path | None, target: str) -> tuple[Profile, Balance | None]:
    """``auto`` picks by balance; anything else is a profile name."""
    profiles = load_profiles(root)
    if not profiles:
        raise RelayError("No relay profiles yet; add one with `cswap relay add`")
    if target == "auto":
        return pick_best(profiles, fetch_all(profiles))
    return find_profile(profiles, target), None


# ---------------------------------------------------------------------------
# Claude Code settings
# ---------------------------------------------------------------------------


def settings_env(profile: Profile) -> dict[str, str]:
    env = {
        "ANTHROPIC_BASE_URL": profile.base_url,
        "ANTHROPIC_AUTH_TOKEN": profile.key,
    }
    if profile.model:
        env["ANTHROPIC_MODEL"] = profile.model
    return env


def claude_settings_path() -> Path:
    return paths.get_claude_config_home() / "settings.json"


def _read_json_object(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError as e:
        raise RelayError(f"{path} is not valid JSON; fix it before switching: {e}") from e
    if not isinstance(data, dict):
        raise RelayError(f"{path} does not hold a JSON object")
    return data


def _write_json_preserving_mode(path: Path, data: dict) -> None:
    """Atomic write that keeps the file's mode and writes through symlinks.

    Not ``atomic_write_json``: that one hardens the parent to 0700, which is
    right for the backup root but not ours to impose on ``~/.claude``.
    """
    target = Path(os.path.realpath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o600
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        if sys.platform != "win32":
            os.chmod(tmp, mode)
        replace_with_retry(tmp, str(target))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply_to_claude_settings(profile: Profile | None, path: Path | None = None) -> Path:
    """Merge ``profile``'s env into Claude's settings.json; ``None`` removes it."""
    path = path or claude_settings_path()
    data = _read_json_object(path)
    env = data.get("env")
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise RelayError(f"'env' in {path} is not an object")
    for k in ENV_KEYS:
        env.pop(k, None)
    if profile is not None:
        env.update(settings_env(profile))
    if env:
        data["env"] = env
    else:
        data.pop("env", None)
    _write_json_preserving_mode(path, data)
    return path


def active_profile_name(profiles: list[Profile], path: Path | None = None) -> str | None:
    """The profile whose key is in Claude's settings.json, if any."""
    try:
        env = _read_json_object(path or claude_settings_path()).get("env") or {}
    except RelayError:
        return None
    token = env.get("ANTHROPIC_AUTH_TOKEN") if isinstance(env, dict) else None
    for p in profiles:
        if p.key == token:
            return p.name
    return None


def write_profile_settings(profile: Profile, root: Path | None = None) -> Path:
    path = relay_dir(root) / "settings" / f"{profile.name}.json"
    atomic_write_json(path, {"env": settings_env(profile)})
    return path


def exec_claude(settings_file: Path, claude_args: list[str]) -> NoReturn:
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RelayError("'claude' was not found on PATH. Install Claude Code first.")
    argv = [claude_bin, "--settings", str(settings_file), *claude_args]
    if sys.platform == "win32":
        try:
            rc = subprocess.run(argv).returncode
        except KeyboardInterrupt:
            rc = 130
        sys.exit(rc)
    os.execvp(claude_bin, argv)
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}"


def _fmt_expiry(ts: int) -> str:
    if ts == 0:
        return "never"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _balance_json(b: Balance | RelayError) -> dict:
    if isinstance(b, RelayError):
        return {"error": str(b)}
    return {
        "tokenName": b.token_name,
        "grantedUsd": round(b.granted_usd, 6),
        "usedUsd": round(b.used_usd, 6),
        "availableUsd": round(b.available_usd, 6),
        "unlimited": b.unlimited,
        "expiresAt": b.expires_at,
        "expired": b.expired,
    }


def _cmd_list(args) -> None:
    profiles = load_profiles()
    if not profiles:
        print(muted("No relay profiles yet; add one with `cswap relay add NAME KEY --base-url URL`"))
        return
    balances = fetch_all(profiles)
    active = active_profile_name(profiles)
    if args.json:
        print(json.dumps({
            "schemaVersion": SCHEMA_VERSION,
            "active": active,
            "profiles": [
                {
                    "name": p.name,
                    "baseUrl": p.base_url,
                    "key": mask_key(p.key),
                    "model": p.model,
                    "balance": _balance_json(b),
                }
                for p, b in zip(profiles, balances)
            ],
        }, indent=2))
        return

    width = max(len(p.name) for p in profiles)
    total = 0.0
    for p, b in zip(profiles, balances):
        marker = accent("*") if p.name == active else " "
        head = f"{marker} {bolded(p.name.ljust(width))}  {dimmed(mask_key(p.key))}"
        if isinstance(b, RelayError):
            print(f"{head}  {muted('error: ' + str(b))}")
            continue
        if b.unlimited:
            quota = "unlimited"
        else:
            total += b.available_usd
            pct = (b.used_usd / b.granted_usd * 100) if b.granted_usd else 0.0
            quota = (
                f"{_fmt_usd(b.available_usd):>10} left of {_fmt_usd(b.granted_usd):>10}"
                f"  ({pct:4.1f}% used)"
            )
        expiry = "EXPIRED" if b.expired else f"expires {_fmt_expiry(b.expires_at)}"
        print(f"{head}  {quota}  {dimmed(expiry)}")
    print(dimmed(f"\n  total available: {_fmt_usd(total)}"
                 + ("   (* = active in ~/.claude/settings.json)" if active else "")))


def _cmd_add(args) -> None:
    key = sys.stdin.read().strip() if args.key == "-" else args.key
    p = add_profile(None, args.name, key, args.base_url, args.model, force=args.force)
    print(f"Added relay profile {bolded(p.name)} ({mask_key(p.key)} @ {p.base_url}"
          + (f", model {p.model}" if p.model else "") + ")")


def _cmd_remove(args) -> None:
    profiles = load_profiles()
    if active_profile_name(profiles) == args.name:
        warning(f"{args.name} is still active in ~/.claude/settings.json; "
                "run `cswap relay use --off` or switch to another profile", file=sys.stderr)
    remove_profile(None, args.name)
    print(f"Removed relay profile {args.name}")


def _describe_pick(profile: Profile, balance: Balance | None) -> str:
    if balance is None:
        return profile.name
    left = "unlimited" if balance.unlimited else f"{_fmt_usd(balance.available_usd)} left"
    return f"{profile.name} ({left})"


def _cmd_use(args) -> None:
    if args.off:
        path = apply_to_claude_settings(None)
        print(f"Removed relay env from {path}")
        return
    if not args.target:
        raise RelayError("Give a profile name, `auto`, or --off")
    profile, balance = resolve_target(None, args.target)
    path = apply_to_claude_settings(profile)
    print(f"Using relay {bolded(_describe_pick(profile, balance))} via {path}")
    print(dimmed("New Claude Code sessions pick it up; restart running ones."))


def _cmd_run(args, tail: list[str]) -> None:
    profile, balance = resolve_target(None, args.target)
    settings_file = write_profile_settings(profile)
    print(dimmed(f"Launching claude on relay {_describe_pick(profile, balance)}"), file=sys.stderr)
    exec_claude(settings_file, tail)


def _cmd_path(args) -> None:
    print(profiles_path())


def main(argv: list[str]) -> None:
    """Handle ``cswap relay <subcommand>`` (pre-dispatched from ``cli.main``)."""
    if "--" in argv:
        split = argv.index("--")
        head, tail = argv[:split], argv[split + 1:]
    else:
        head, tail = argv, []

    parser = argparse.ArgumentParser(
        prog="cswap relay",
        description="Manage third-party relay keys applied through Claude Code's settings.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cswap relay add blue-az-0 sk-... --base-url https://relay.example.com/v1 --model claude-opus-5-5
  cswap relay add blue-az-1 sk-...          # reuses the previous base URL and model
  cswap relay list                          # balances of every key
  cswap relay use auto                      # write the richest key into ~/.claude/settings.json
  cswap relay use blue-az-1
  cswap relay use --off                     # remove the relay env again
  cswap relay run auto -- --resume          # this terminal only
        """,
    )
    sub = parser.add_subparsers(dest="action")

    p_list = sub.add_parser("list", help="Show every profile with its live balance (default)")
    p_list.add_argument("--json", action="store_true", help="Emit JSON")

    p_add = sub.add_parser("add", help="Add a relay key")
    p_add.add_argument("name")
    p_add.add_argument("key", help="The relay key, or - to read it from stdin")
    p_add.add_argument("--base-url", help="Relay URL (with or without /v1); defaults to the previous profile's")
    p_add.add_argument("--model", help="ANTHROPIC_MODEL to set; defaults to the previous profile's")
    p_add.add_argument("--force", action="store_true", help="Replace an existing profile")

    p_rm = sub.add_parser("remove", help="Remove a relay profile")
    p_rm.add_argument("name")

    p_use = sub.add_parser("use", help="Apply a profile to ~/.claude/settings.json")
    p_use.add_argument("target", nargs="?", metavar="NAME|auto")
    p_use.add_argument("--off", action="store_true", help="Remove the relay env instead")

    p_run = sub.add_parser("run", help="Launch claude on a profile in this terminal only")
    p_run.add_argument("target", metavar="NAME|auto")

    sub.add_parser("path", help="Print where relay profiles are stored")

    args = parser.parse_args(head)
    if tail and args.action != "run":
        parser.error("arguments after -- are only accepted by `run`")

    try:
        if args.action == "add":
            _cmd_add(args)
        elif args.action == "remove":
            _cmd_remove(args)
        elif args.action == "use":
            _cmd_use(args)
        elif args.action == "run":
            _cmd_run(args, tail)
        elif args.action == "path":
            _cmd_path(args)
        else:
            if args.action is None:
                args.json = False
            _cmd_list(args)
    except ClaudeSwitchError as e:
        error(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{dimmed('Operation cancelled')}", file=sys.stderr)
        sys.exit(130)
