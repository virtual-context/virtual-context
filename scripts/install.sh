#!/usr/bin/env bash
set -euo pipefail

PACKAGE_NAME="virtual-context"

log() {
  printf "[virtual-context] %s\n" "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_with_pipx() {
  if need_cmd pipx; then
    log "Installing with pipx..."
    pipx install --force "$PACKAGE_NAME"
    return 0
  fi

  if need_cmd python3; then
    log "pipx not found. Installing pipx via python3 --user..."
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath >/dev/null 2>&1 || true

    # shellcheck disable=SC2016
    log 'If this is your first pipx install, restart your shell and rerun this script.'
    if python3 -m pipx --version >/dev/null 2>&1; then
      python3 -m pipx install --force "$PACKAGE_NAME"
      return 0
    fi
  fi

  return 1
}

install_with_pip_user() {
  if ! need_cmd python3; then
    return 1
  fi
  log "Falling back to python3 -m pip --user install..."
  python3 -m pip install --user --upgrade "$PACKAGE_NAME"
}

main() {
  log "Installing $PACKAGE_NAME"

  if install_with_pipx; then
    :
  elif install_with_pip_user; then
    :
  else
    log "Installation failed: need python3 (and pip) or pipx"
    exit 1
  fi

  if ! need_cmd virtual-context; then
    log "Installed, but 'virtual-context' is not on PATH yet."
    log "Open a new shell, then run: virtual-context --help"
    exit 0
  fi

  log "Install complete"
  log "Next:"
  log "  1) virtual-context onboard --wizard    # guided setup (config + proxy instances + daemon)"
  log "  Or manual setup:"
  log "  1) virtual-context init coding"
  log "  2) virtual-context config validate"
  log "  3) virtual-context proxy --upstream https://api.anthropic.com"
}

main "$@"
