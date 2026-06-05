#!/usr/bin/env bash
# setup.sh — Instalación completa de OptionsDesk en Ubuntu 22.04/24.04
# Ejecutar como root (o con sudo) desde la raíz del repo clonado:
#
#   sudo bash deploy/setup.sh
#
# Qué hace:
#   1. Instala dependencias del sistema (python3.11+, nginx, git)
#   2. Crea el usuario 'optionsdesk' y /opt/optionsdesk
#   3. Copia el código y crea un virtualenv
#   4. Instala las units de systemd y el config de nginx
#   5. Habilita y arranca los servicios
#
# PREREQUISITO MANUAL (no automatizable):
#   cp .env /opt/optionsdesk/.env   ← hacerlo ANTES de correr este script,
#                                     o editar /opt/optionsdesk/.env después.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="/opt/optionsdesk"
APP_USER="optionsdesk"
VENV="$INSTALL_DIR/.venv"
PYTHON_BIN="$(command -v python3.11 || command -v python3)"

echo "==> [1/6] Actualizando paquetes e instalando dependencias del sistema..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    nginx git curl

echo "==> [2/6] Creando usuario y directorio de instalación..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$APP_USER"
fi
mkdir -p "$INSTALL_DIR"
chown "$APP_USER:$APP_USER" "$INSTALL_DIR"

echo "==> [3/6] Copiando código fuente a $INSTALL_DIR..."
rsync -a --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '/data/' \
    --exclude '/logs/' \
    --exclude '.env' \
    "$REPO_SRC/" "$INSTALL_DIR/"
chown -R "$APP_USER:$APP_USER" "$INSTALL_DIR"

# Crear directorios de datos con permisos correctos
mkdir -p "$INSTALL_DIR/data/snapshots" "$INSTALL_DIR/data/history"
chown -R "$APP_USER:$APP_USER" "$INSTALL_DIR/data"

echo "==> [4/6] Creando virtualenv e instalando dependencias Python..."
if [ ! -d "$VENV" ]; then
    "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e "$INSTALL_DIR"

echo "==> [5/6] Instalando .env (si existe en el repo local)..."
if [ -f "$REPO_SRC/.env" ]; then
    cp "$REPO_SRC/.env" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
    chown "$APP_USER:$APP_USER" "$INSTALL_DIR/.env"
    echo "    .env copiado desde $REPO_SRC/.env"
else
    echo "    AVISO: .env no encontrado. Copialo manualmente:"
    echo "      cp /tu/.env $INSTALL_DIR/.env"
    echo "      chmod 600 $INSTALL_DIR/.env"
    echo "      chown $APP_USER:$APP_USER $INSTALL_DIR/.env"
    # Crear .env vacío para que las units no fallen con EnvironmentFile
    [ -f "$INSTALL_DIR/.env" ] || touch "$INSTALL_DIR/.env"
fi

echo "==> [6/6] Instalando y arrancando servicios..."

# systemd units
for unit in optionsdesk-dashboard.service optionsdesk-recorder.service \
            optionsdesk-demo-runner.service \
            optionsdesk-monitor.service optionsdesk-monitor.timer; do
    cp "$REPO_SRC/deploy/$unit" "/etc/systemd/system/$unit"
done

# Actualizar las units para apuntar a $VENV si cambió el path
sed -i "s|/opt/optionsdesk/.venv|$VENV|g" /etc/systemd/system/optionsdesk-*.service 2>/dev/null || true

systemctl daemon-reload
systemctl enable --now optionsdesk-dashboard.service
systemctl enable --now optionsdesk-recorder.service
systemctl enable --now optionsdesk-demo-runner.service
systemctl enable --now optionsdesk-monitor.timer

# nginx
cp "$REPO_SRC/deploy/nginx.conf" /etc/nginx/sites-available/optionsdesk
ln -sf /etc/nginx/sites-available/optionsdesk /etc/nginx/sites-enabled/optionsdesk
# Desactivar el default de nginx para evitar conflicto en el puerto 80
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo ""
echo "============================================================"
echo " OptionsDesk instalado correctamente."
echo ""
echo " Dashboard: http://$(curl -s ifconfig.me 2>/dev/null || echo '<IP_DEL_VPS>')"
echo ""
echo " Estado de servicios:"
systemctl is-active optionsdesk-dashboard.service  && echo "  dashboard : activo" || echo "  dashboard : FALLO"
systemctl is-active optionsdesk-recorder.service   && echo "  recorder  : activo" || echo "  recorder  : FALLO"
systemctl is-active optionsdesk-demo-runner.service && echo "  runner    : activo" || echo "  runner    : FALLO"
systemctl is-active optionsdesk-monitor.timer      && echo "  monitor   : activo" || echo "  monitor   : FALLO"
echo ""
echo " Logs en tiempo real:"
echo "   journalctl -fu optionsdesk-dashboard"
echo "   journalctl -fu optionsdesk-recorder"
echo "   journalctl -fu optionsdesk-demo-runner"
echo "============================================================"
