#!/bin/bash

set -e

PERMANENT=false

for arg in "$@"; do
    if [ "$arg" == "--permanent" ]; then
        PERMANENT=true
    fi
done

detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            fedora) echo "fedora" ;;
            arch) echo "arch" ;;
            debian|ubuntu) echo "debian" ;;
            *) echo "unknown" ;;
        esac
    else
        echo "unknown"
    fi
}

DISTRO=$(detect_distro)

if [ "$DISTRO" == "unknown" ]; then
    echo "[!] Cannot distinguish the distro. Aborting."
    exit 1
fi

echo "[+] Distribution: $DISTRO"
echo "[+] Installing mosquitto, pip and dependencies..."

case "$DISTRO" in
    fedora)
        sudo dnf install -y mosquitto python3-pip
        ;;
    arch)
        sudo pacman -Syu --noconfirm mosquitto mosquitto-clients python-pip
        ;;
    debian)
        sudo apt update
        sudo apt install -y mosquitto mosquitto-clients python3-pip
        ;;
esac

echo "[+] Installing Python-dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install paho-mqtt Pillow numpy

echo "[+] Staring Mosquitto..."
sudo systemctl start mosquitto

if [ "$PERMANENT" == true ]; then
    echo "[✓] Enabling Mosquitto at boot..."
    sudo systemctl enable mosquitto
else
    echo "[*] Mosquitto will work only before reboot."
fi

echo "[✓] Installation completed!"
