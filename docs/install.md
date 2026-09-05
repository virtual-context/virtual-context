# Install

This page provides copy-paste install commands for macOS/Linux and Windows, plus daemon setup instructions.

## Requirements

- Python **3.11 or newer**. On older Pythons, pip will refuse to resolve the package.
- The base install includes the engine, SQLite storage, filesystem utilities, HTTP providers, and exact token counting. Server, UI, MCP, and local embedding features are optional.

## pip

```bash
pip install virtual-context
```

Optional extras for specific storage backends and features:

```bash
pip install "virtual-context[proxy]"             # HTTP proxy and dashboard
pip install "virtual-context[embeddings]"        # Local semantic search (PyTorch)
pip install "virtual-context[tui]"               # Terminal chat UI
pip install "virtual-context[mcp]"               # MCP server
pip install "virtual-context[postgres]"          # PostgreSQL backend
pip install "virtual-context[redis]"             # Redis session cache
pip install "virtual-context[providers]"         # Optional provider SDK integrations
pip install "virtual-context[neo4j]"             # Direct Neo4j utilities
pip install "virtual-context[falkordb]"          # Direct FalkorDB utilities
pip install "virtual-context[proxy,embeddings]"  # Typical local daemon
pip install "virtual-context[all]"               # Every optional feature
```

The `storage.backend: postgres` configuration requires the `postgres` extra. Local embedding models load only on first use; installing the base engine does not install PyTorch or download model weights. HTTP provider integrations work without provider SDKs. `bridge` remains an alias for `proxy`; `tiktoken` remains an accepted compatibility extra because exact token counting is included in the base package.

### Reproducible source installs

The checked-in `uv.lock` pins the complete dependency graph, including optional features and artifact hashes. Select only the features needed by that environment:

```bash
python -m pip install uv==0.9.30
uv sync --locked                                      # Base engine
uv sync --locked --extra proxy --extra embeddings     # Local daemon
uv sync --locked --extra all --extra dev               # Complete development environment
```

Use `uv lock --check` to verify that dependency metadata and the lock agree. Update the lock alongside any dependency changes. Deployment builds should install from this lock with their selected extras; unconstrained `pip install` commands above are for released packages.

### Config discovery

Commands look for a config file named `virtual-context.yaml` (also accepted: `.yml`, `.json`, and the `virtualcontext.` prefix) starting in the current directory and walking up to your home directory, then fall back to `~/.virtualcontext/config.yaml`. Pass `-c <path>` to use an explicit file.

## macOS / Linux (install script)

Install the CLI, proxy, and local embedding support:

```bash
curl -fsSL https://raw.githubusercontent.com/virtual-context/virtual-context/main/scripts/install.sh | bash
```

Or run locally from a clone:

```bash
bash scripts/install.sh
```

Run guided setup (creates config, picks your tagger provider, and optionally installs daemon):

```bash
virtual-context onboard --upstream https://api.anthropic.com
```

Or use a preset without the interactive wizard:

```bash
virtual-context init coding
virtual-context config validate
```

## Windows (PowerShell)

Install the CLI:

```powershell
iwr https://raw.githubusercontent.com/virtual-context/virtual-context/main/scripts/install.ps1 -useb | iex
```

Or run locally from a clone:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

Run guided setup:

```powershell
virtual-context onboard --upstream https://api.anthropic.com
```

Or use a preset without the interactive wizard:

```powershell
virtual-context init coding
virtual-context config validate
```

## One-Command Daemon Setup

Install and start a background proxy service. Creates `~/.virtualcontext/` with config and data automatically:

```bash
virtual-context daemon install --upstream https://api.anthropic.com
```

The wizard runs automatically in a terminal. For multi-instance setups:

```bash
virtual-context onboard --install-daemon
```

Options:
- Add `--no-start` to install service files without starting immediately.
- Omit `--upstream` when using multi-instance proxy mode in `virtual-context.yaml`.

The daemon runs the proxy on the default port **5757** (there is no `--port` flag on `daemon install`; multi-instance configs set ports per instance).

After install, use daemon lifecycle commands:

```bash
virtual-context daemon status
virtual-context daemon start
virtual-context daemon stop
virtual-context daemon restart
virtual-context daemon uninstall
```

## Install Daemon (macOS)

Create a LaunchAgent so the proxy runs in the background.

1. Run `virtual-context daemon install --upstream ...` (recommended; it writes a plist with fully resolved paths), or manually create the plist below.
2. Save this as `~/Library/LaunchAgents/io.virtualcontext.proxy.plist`, **replacing `/Users/YOURNAME` with your actual home directory in the two log paths**. launchd does not expand `$HOME` in `StandardOutPath`/`StandardErrorPath` (those keys are used verbatim), so shell variables there break logging and can keep the job from spawning. `$HOME` inside `ProgramArguments` is fine because that line runs through `bash -lc`.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>io.virtualcontext.proxy</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>virtual-context -c $HOME/.virtualcontext/config.yaml proxy --upstream https://api.anthropic.com</string>
  </array>

  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/Users/YOURNAME/Library/Logs/virtual-context.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOURNAME/Library/Logs/virtual-context.err.log</string>
</dict>
</plist>
```

Load and start:

```bash
launchctl unload ~/Library/LaunchAgents/io.virtualcontext.proxy.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/io.virtualcontext.proxy.plist
launchctl start io.virtualcontext.proxy
```

Check status/logs:

```bash
launchctl list | grep virtualcontext
tail -n 100 ~/Library/Logs/virtual-context.log
```

## Install Daemon (Linux systemd --user)

Create `~/.config/systemd/user/virtual-context.service`:

```ini
[Unit]
Description=virtual-context proxy
After=network-online.target

[Service]
Type=simple
ExecStart=%h/.local/bin/virtual-context -c %h/.virtualcontext/config.yaml proxy --upstream https://api.anthropic.com
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now virtual-context
```

Check status/logs:

```bash
systemctl --user status virtual-context
journalctl --user -u virtual-context -f
```

## Install Daemon (Windows)

Use Task Scheduler to run at logon:

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -Command \"virtual-context -c $HOME\\virtual-context.yaml proxy --upstream https://api.anthropic.com\""
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "virtual-context-proxy" -Action $action -Trigger $trigger -Description "Run virtual-context proxy"
Start-ScheduledTask -TaskName "virtual-context-proxy"
```

Check task state:

```powershell
Get-ScheduledTask -TaskName "virtual-context-proxy"
```

## Interactive Wizard

The `onboard` command runs an interactive wizard by default in a terminal:

1. **Upstream provider** — where your LLM requests go (Anthropic, OpenAI, Gemini, custom)
2. **Tagging/summarization provider + model** — auto-inferred from upstream (cheapest option), with override
3. **Inbound tagging mode** (embedding, LLM, or keyword)
4. **Proxy instances** — one or multiple, each with its own upstream provider, port, and label
5. **Per-instance config** — each instance gets a standalone YAML config with isolated storage
6. **Daemon install** — optionally install as a background service

```bash
virtual-context onboard
```

Skip the wizard with `--no-wizard` to use preset defaults.

## Per-Instance Config

When using multi-instance proxy, each instance can have its own config file for isolated storage, tag generator, and summarization provider:

```yaml
# Master config: virtual-context.yaml
proxy:
  instances:
    - port: 5757
      upstream: https://api.anthropic.com
      label: anthropic
      config: ./virtual-context-proxy-anthropic.yaml

    - port: 5758
      upstream: https://api.openai.com/v1
      label: openai
      config: ./virtual-context-proxy-openai.yaml
```

Each instance config is a full standalone config:

```yaml
# virtual-context-proxy-anthropic.yaml
version: '0.2'
storage_root: .virtualcontext/anthropic
tag_generator:
  type: llm
  provider: anthropic
  model: claude-haiku-4-5-20251001
summarization:
  provider: anthropic
  model: claude-haiku-4-5-20251001
storage:
  backend: sqlite
  sqlite:
    path: .virtualcontext/anthropic/store.db
```

Instances without a `config` field share the master engine.

## Presets

List available presets:

```bash
virtual-context presets list
```

Show a preset's config as YAML:

```bash
virtual-context presets show coding
virtual-context presets show agentic
```

## Daemon Restart

Restart the proxy daemon (stop + start):

```bash
virtual-context daemon restart
```

All daemon commands:

```bash
virtual-context daemon status
virtual-context daemon start
virtual-context daemon stop
virtual-context daemon restart
virtual-context daemon uninstall
```

## Notes

- If you use multi-instance proxy mode in YAML (`proxy.instances`), run `virtual-context -c <config> proxy` without `--upstream`.
- If installed with `pipx`, command path is managed automatically.
- If installed with `pip --user`, ensure your user scripts directory is on PATH.
