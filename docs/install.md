# Install

This page provides copy-paste install commands for macOS/Linux and Windows, plus daemon setup instructions.

## macOS / Linux

Install the CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/virtual-context/virtual-context/main/scripts/install.sh | bash
```

Or run locally from a clone:

```bash
bash scripts/install.sh
```

Bootstrap config and validate:

```bash
virtual-context init coding
virtual-context config validate
```

Or run guided setup (creates config if missing):

```bash
virtual-context onboard
virtual-context onboard --wizard
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

Bootstrap config and validate:

```powershell
virtual-context init coding
virtual-context config validate
```

Or run guided setup (creates config if missing):

```powershell
virtual-context onboard
virtual-context onboard --wizard
```

## One-Command Daemon Setup

Use the onboarding wizard to install and start a background proxy service:

```bash
virtual-context onboard --wizard --install-daemon
virtual-context onboard --install-daemon --upstream https://api.anthropic.com
```

Options:
- Add `--no-start` to install service files without starting immediately.
- Omit `--upstream` when using multi-instance proxy mode in `virtual-context.yaml`.

After install, use daemon lifecycle commands:

```bash
virtual-context daemon status
virtual-context daemon start
virtual-context daemon stop
virtual-context daemon uninstall
```

## Install Daemon (macOS)

Create a LaunchAgent so the proxy runs in the background.

1. Create config first (`virtual-context.yaml`) and make sure it includes your upstream strategy.
2. Save this as `~/Library/LaunchAgents/io.virtualcontext.proxy.plist`:

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
    <string>virtual-context -c $HOME/virtual-context.yaml proxy --upstream https://api.anthropic.com</string>
  </array>

  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>$HOME/Library/Logs/virtual-context.log</string>
  <key>StandardErrorPath</key>
  <string>$HOME/Library/Logs/virtual-context.err.log</string>
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
launchctl list | rg virtualcontext
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
ExecStart=%h/.local/bin/virtual-context -c %h/virtual-context.yaml proxy --upstream https://api.anthropic.com
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

## Enhanced Wizard

The `onboard --wizard` flow guides you through:

1. **Tagging/summarization provider + model** (Anthropic, OpenAI, Ollama, or custom)
2. **Inbound tagging mode** (embedding, LLM, or keyword)
3. **Proxy instances** — one or multiple, each with its own upstream provider, port, and label
4. **Per-instance config** — each instance gets a standalone YAML config with isolated storage
5. **Daemon install** — optionally install as a background service

```bash
virtual-context onboard --wizard
```

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
virtual-context presets show general
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
