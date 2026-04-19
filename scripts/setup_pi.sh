#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APT_REQUIREMENTS_FILE="$ROOT_DIR/requirements-raspberry-pi-apt.txt"
PIP_REQUIREMENTS_FILE="$ROOT_DIR/requirements-raspberry-pi.txt"
VENV_DIR="$ROOT_DIR/venv"

if [[ ! -f "$APT_REQUIREMENTS_FILE" ]]; then
  echo "Missing apt requirements file: $APT_REQUIREMENTS_FILE" >&2
  exit 1
fi

if [[ ! -f "$PIP_REQUIREMENTS_FILE" ]]; then
  echo "Missing pip requirements file: $PIP_REQUIREMENTS_FILE" >&2
  exit 1
fi

mapfile -t APT_PACKAGES < <(grep -Ev '^\s*(#|$)' "$APT_REQUIREMENTS_FILE")

if [[ ${#APT_PACKAGES[@]} -eq 0 ]]; then
  echo "No apt packages found in $APT_REQUIREMENTS_FILE" >&2
  exit 1
fi

UDEV_RULES_SRC="$ROOT_DIR/config/99-djr3x.rules"
UDEV_RULES_DEST="/etc/udev/rules.d/99-djr3x.rules"

echo "==> Updating apt package lists"
sudo apt update

echo "==> Installing Raspberry Pi system packages"
sudo apt install -y "${APT_PACKAGES[@]}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating Python virtual environment"
  python3 -m venv "$VENV_DIR"
fi

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing Python requirements"
pip install -r "$PIP_REQUIREMENTS_FILE"

if [[ -f "$UDEV_RULES_SRC" ]]; then
  echo "==> Installing udev rules ($UDEV_RULES_DEST)"
  sudo cp "$UDEV_RULES_SRC" "$UDEV_RULES_DEST"
  sudo udevadm control --reload-rules
  sudo udevadm trigger --subsystem-match=tty
  sudo udevadm trigger --subsystem-match=video4linux
  echo "    udev rules installed — /dev/maestro, /dev/arduino_uno, /dev/arduino_nano, /dev/camera_main"
else
  echo "WARNING: udev rules source not found at $UDEV_RULES_SRC — skipping"
fi

echo
echo "Pi setup complete."
echo "If this is a new system or Bluetooth audio still does not appear, reboot once:"
echo "  sudo reboot"
