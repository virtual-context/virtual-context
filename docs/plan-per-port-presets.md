# Implementation Plan: Per-Port Config + Installer/Onboarding

## 1. Goals

1. Each proxy instance gets its own complete config file (own engine, own storage, fully independent).
2. Wizard generates N config files (one per instance) + one master config that references them.
3. Install scripts for macOS/Linux (bash) and Windows (PowerShell).
4. Daemon management across launchd, systemd, and Task Scheduler (already partially implemented).
5. Backward compatible — existing single-instance configs work unchanged.

## 2. What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| Multi-instance runner | `proxy/multi.py` | Works, but shares one engine across all instances |
| Daemon commands | `cli/main.py` (lines 720-773) | `status`, `start`, `stop`, `uninstall` on all 3 platforms |
| Daemon installers | `cli/main.py` (lines 527-634) | launchd, systemd, Task Scheduler |
| Onboard wizard | `cli/main.py` (lines 636-706) | Exists, generates config + installs daemon |
| Instance wizard | `cli/main.py` (lines 470-509) | `_run_instance_wizard()` — loops per instance |
| Preset templates | `presets/coding.py`, `presets/general.py` | Starter YAML templates for `init` command |
| Proxy config | `virtual-context-proxy.yaml` | Complete production config (Haiku cloud, SQLite, embeddings) |

## 3. Architecture: Per-Instance Config Files

### 3.1 Config model

Each instance references its own complete YAML config file:

```yaml
# virtual-context-proxy.yaml (master)
proxy:
  instances:
    - port: 5757
      label: anthropic
      upstream: https://api.anthropic.com
      config: virtual-context-proxy-anthropic.yaml
    - port: 5758
      label: openai
      upstream: https://api.openai.com/v1
      config: virtual-context-proxy-openai.yaml
```

Each referenced config file is a full standalone config (tag_generator, summarization, storage, retrieval, etc.) — the same format as today's `virtual-context-proxy.yaml`.

### 3.2 Storage isolation

Each instance config file points to its own SQLite path. The wizard auto-generates these:

- Instance `anthropic` → `.virtualcontext/anthropic/store.db`
- Instance `openai` → `.virtualcontext/openai/store.db`

No shared storage by default. If an advanced user manually points two configs at the same DB, that's their choice.

### 3.3 Type changes

```python
# virtual_context/types.py
@dataclass
class ProxyInstanceConfig:
    port: int = 5757
    upstream: str = ""
    label: str = ""
    host: str = "127.0.0.1"
    config: str = ""  # NEW: path to instance-specific config file
```

### 3.4 Config parsing changes

```python
# virtual_context/config.py — in instance parsing loop
ProxyInstanceConfig(
    port=inst.get("port", 5757),
    upstream=inst.get("upstream", ""),
    label=inst.get("label", ""),
    host=inst.get("host", "127.0.0.1"),
    config=inst.get("config", ""),  # NEW
)
```

Validation: if `config` is set, verify the file exists and is parseable.

## 4. Runtime Changes

### 4.1 `proxy/multi.py` — per-instance engine

Current: one shared engine, passed to all instances.

New:
```python
async def run_multi_instance(instances, config_path):
    servers = []
    for inst in instances:
        # Each instance gets its own engine from its own config
        inst_config_path = inst.config if inst.config else config_path
        app = create_app(
            upstream=inst.upstream,
            config_path=inst_config_path,
            # No shared engine — each app creates its own
            instance_label=inst.label,
        )
        config = uvicorn.Config(app, host=inst.host, port=inst.port)
        servers.append(uvicorn.Server(config))
    await asyncio.gather(*(s.serve() for s in servers))
```

Each `create_app()` call instantiates its own `VirtualContextEngine` from its own config file. Fully independent.

### 4.2 Metrics

Per-instance `ProxyMetrics`. Each engine/app tracks its own stats. The dashboard aggregates at read time if needed.

### 4.3 Backward compatibility

- If `config` field is empty/missing → use the master config path (current behavior).
- If `proxy.instances` is empty → single-instance mode with `--upstream` (current behavior).
- No breaking changes.

## 5. Wizard Flow

### 5.1 Entry

```
virtual-context onboard --wizard
```

### 5.2 Steps

1. **Provider for tagging + summarization**
   - Provider menu: [Anthropic, OpenAI-compatible, Gemini, Ollama, Custom]
   - Model menu (provider-specific): e.g. [haiku (recommended), sonnet, opus, custom]

2. **Multiple instances?**
   - "Do you want to support multiple AI providers through VC?" → yes/no
   - If no → single instance: ask upstream URL, port, done.
   - If yes → "How many instances?" → loop:

3. **Per-instance loop** (for each instance):
   - Upstream provider: [Anthropic, OpenAI, Gemini, Custom URL]
   - Upstream URL (pre-filled from provider choice)
   - Port (auto-increment from 5757)
   - Label (auto-generated from provider, editable)

4. **Inbound tagging mode**
   - [embedding (recommended), llm, keyword]

5. **Review screen**
   - Show: N instances, each with label/port/upstream/config path
   - Show: tagger provider+model, summarizer provider+model
   - Show: storage paths (one per instance)

6. **Confirm write**
   - Generates N instance config files + 1 master config
   - Each instance config: full standalone YAML with isolated storage path

7. **Daemon install**
   - "Install as system daemon?" → [install + start, install only, skip]

### 5.3 Non-interactive mode

```bash
# Single instance
virtual-context onboard --no-wizard \
  --upstream https://api.anthropic.com \
  --port 5757 \
  --install-daemon --no-start

# Multi-instance (from existing config)
virtual-context onboard --no-wizard \
  --config virtual-context-proxy.yaml \
  --install-daemon
```

## 6. Generated Config Files

### 6.1 Master config (virtual-context-proxy.yaml)

```yaml
version: "0.2"

proxy:
  instances:
    - port: 5757
      label: anthropic
      upstream: https://api.anthropic.com
      config: virtual-context-proxy-anthropic.yaml
    - port: 5758
      label: openai
      upstream: https://api.openai.com/v1
      config: virtual-context-proxy-openai.yaml
```

### 6.2 Instance config (virtual-context-proxy-anthropic.yaml)

Full standalone config — same structure as today's `virtual-context-proxy.yaml`, with:
- `storage.sqlite.path: ".virtualcontext/anthropic/store.db"`
- Provider/model settings as chosen in wizard
- All other sections (compaction, retrieval, assembly, etc.) from the preset template used

### 6.3 Template source

The wizard uses the existing preset templates (`general`, `coding`) as the base for generating instance configs. Presets remain starter templates — not a runtime concept.

## 7. Install Scripts

### 7.1 `scripts/install.sh` (macOS/Linux)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Prefer pipx, fallback to pip --user
if command -v pipx &>/dev/null; then
    pipx install virtual-context
elif command -v pip3 &>/dev/null; then
    pip3 install --user virtual-context
else
    echo "Error: pip3 or pipx required"
    exit 1
fi

echo "Installed. Run: virtual-context onboard --wizard"
```

Usage:
```bash
curl -fsSL https://raw.githubusercontent.com/virtual-context/virtual-context/main/scripts/install.sh | bash
```

### 7.2 `scripts/install.ps1` (Windows)

```powershell
# Prefer pipx, fallback to pip --user
if (Get-Command pipx -ErrorAction SilentlyContinue) {
    pipx install virtual-context
} elseif (Get-Command pip -ErrorAction SilentlyContinue) {
    pip install --user virtual-context
} else {
    Write-Error "pip or pipx required"
    exit 1
}

Write-Host "Installed. Run: virtual-context onboard --wizard"
```

Usage:
```powershell
iwr https://raw.githubusercontent.com/virtual-context/virtual-context/main/scripts/install.ps1 -useb | iex
```

### 7.3 Primary install path in docs

`pipx install virtual-context` is the recommended method. Scripts are convenience wrappers.

## 8. Daemon Behavior

### 8.1 Already implemented

- `virtual-context daemon status|start|stop|uninstall`
- Platform backends: launchd (macOS), systemd --user (Linux), Task Scheduler (Windows)

### 8.2 Needed changes

- **Daemon command string**: currently hardcoded. Needs to use the config path from onboarding:
  - Single: `virtual-context -c <config> proxy --upstream <url>`
  - Multi: `virtual-context -c <config> proxy`
- **Idempotency**: `install` twice should update existing unit cleanly. `uninstall` should not fail if already absent.
- **Restart**: add `daemon restart` (stop + start).

## 9. CLI Additions

### 9.1 Preset listing

```
virtual-context presets list
virtual-context presets show <name>
```

- `list`: table of name + description for each registered preset
- `show`: dump the full config dict (YAML format) for a preset

### 9.2 Daemon restart

```
virtual-context daemon restart
```

## 10. Validation & Test Plan

### 10.1 Unit tests

- Parse `instances[].config` field
- Missing config file → clear error
- Invalid config file → clear error
- Empty config field → falls back to master config path

### 10.2 Integration tests

- Multi-instance boot creates separate engines per instance
- Each instance uses its own SQLite path
- No cross-instance state (different tag indexes, different summaries)
- Single-instance backward compatibility unchanged

### 10.3 CLI tests

- `onboard --wizard` with mocked input transcript
- Non-interactive onboarding flags
- `daemon restart` (mock subprocess)
- `presets list` and `presets show`

### 10.4 Regression

- Existing configs without `config` field still work
- Existing single-instance proxy command unchanged
- All ~960 existing tests pass

## 11. Implementation Phases

1. **Schema**: Add `config` field to `ProxyInstanceConfig`, update parser + validation
2. **Runtime**: Update `proxy/multi.py` to create per-instance engines from per-instance configs
3. **Wizard**: Update onboard wizard to generate N config files + master config
4. **CLI**: Add `presets list/show`, `daemon restart`
5. **Install scripts**: `scripts/install.sh` + `scripts/install.ps1`
6. **Docs**: `docs/install.md` with all install paths, wizard usage, daemon lifecycle, multi-instance example
7. **Daemon hardening**: idempotent install/uninstall, correct command strings
8. **Full regression run**

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Config file sprawl (N+1 files) | Wizard auto-names them; master config is the single entry point |
| User edits wrong file | Master config has comments pointing to instance configs |
| Daemon references stale config path | Daemon install stores absolute path; `daemon status` shows it |
| Platform daemon edge cases | Existing implementations cover the basics; harden incrementally |
