# Deploy en VPS — OptionsDesk

Ubuntu 22.04 / 24.04. Acceso vía SSH. Sin dominio (IP pública directa).

---

## Instalación rápida

```bash
# 1. Clonar (o subir el repo al VPS)
git clone <url-del-repo> /tmp/optionsdesk-src
cd /tmp/optionsdesk-src

# 2. Copiar credenciales ANTES de correr el script
#    El .env nunca se commitea — copia desde tu máquina local
scp .env usuario@<IP_VPS>:/tmp/optionsdesk-src/.env

# 3. Instalar (como root o con sudo)
sudo bash deploy/setup.sh
```

El script hace todo: instala Python/nginx, crea el usuario `optionsdesk`,
copia el código a `/opt/optionsdesk`, crea el virtualenv, configura systemd y nginx.

Al terminar te muestra la URL: `http://<IP_DEL_VPS>`.

---

## Estructura instalada

```
/opt/optionsdesk/          ← código
/opt/optionsdesk/.env      ← credenciales (chmod 600, solo root + optionsdesk)
/opt/optionsdesk/.venv/    ← virtualenv
/opt/optionsdesk/data/     ← snapshots + historial (persistente entre deploys)
```

## Servicios systemd

| Servicio | Qué hace |
|----------|----------|
| `optionsdesk-dashboard` | Streamlit en :8501 (nginx hace proxy → :80) |
| `optionsdesk-recorder` | Graba snapshots cada 15s vía WebSocket Bull Market |
| `optionsdesk-monitor.timer` | Dispara el monitor de posiciones cada 5 min (L-V 10:30-17:00) |

```bash
# Ver estado
systemctl status optionsdesk-dashboard
systemctl status optionsdesk-recorder
systemctl list-timers optionsdesk-monitor.timer

# Logs en vivo
journalctl -fu optionsdesk-dashboard
journalctl -fu optionsdesk-recorder

# Reiniciar manualmente
systemctl restart optionsdesk-dashboard
systemctl restart optionsdesk-recorder
```

---

## Actualizar el código

```bash
cd /tmp/optionsdesk-src
git pull
sudo bash deploy/setup.sh   # rsync + reinstall + reload servicios
```

O más quirúrgico:

```bash
sudo rsync -a --exclude '.git' --exclude '.venv' --exclude 'data' --exclude '.env' \
    ./ /opt/optionsdesk/
sudo -u optionsdesk /opt/optionsdesk/.venv/bin/pip install -q -e /opt/optionsdesk
sudo systemctl restart optionsdesk-dashboard optionsdesk-recorder
```

---

## Firewall (UFW)

```bash
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # Dashboard
ufw enable
```

No abrir el puerto 8501 directamente — nginx hace el proxy.

---

## Troubleshooting

### Dashboard no carga / 502 Bad Gateway

```bash
systemctl status optionsdesk-dashboard
journalctl -n 50 -u optionsdesk-dashboard
# Causa más común: Streamlit no inició aún. Esperar 10s y recargar.
```

### Recorder no conecta a Bull Market

```bash
journalctl -n 50 -u optionsdesk-recorder
# Causas: .env mal configurado, IP del VPS bloqueada por Bull Market (poco frecuente),
# pyhomebroker actualización breaking. Verificar en local primero.
cat /opt/optionsdesk/.env   # confirmar que tiene HB_DNI, HB_USER, etc.
```

### nginx: "port 80 already in use"

```bash
systemctl stop apache2 2>/dev/null; systemctl disable apache2 2>/dev/null
systemctl reload nginx
```

### Snapshots no se acumulan

```bash
ls /opt/optionsdesk/data/snapshots/
# Si vacío: el recorder puede estar corriendo pero sin datos (no es horario de mercado,
# o el WS aún no recibió callbacks). Esperar 60s y revisar logs.
journalctl -n 30 -u optionsdesk-recorder
```

### Timer del monitor no dispara

```bash
systemctl list-timers optionsdesk-monitor.timer
# Si no aparece: el timer no está habilitado
systemctl enable --now optionsdesk-monitor.timer
# Nota: el timer solo dispara L-V 10:30-17:00 hora Argentina (13:30-20:00 UTC).
# Para probar manualmente:
sudo systemctl start optionsdesk-monitor.service
journalctl -n 20 -u optionsdesk-monitor
```

---

## Variables de entorno (.env)

```ini
# Bull Market (requerido para datos reales)
HB_DNI=12345678
HB_USER=tu_usuario
HB_PASSWORD=tu_password
HB_BROKER_ID=6          # Bull Market = 6 (ver pyhomebroker README)

# Telegram (opcional — alertas)
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=

# Ajustes opcionales
RECORDER_INTERVAL_S=15  # default 15s
MIN_TNA_SPREAD_PCT=5.0
HORIZON_MONITOR_ENABLED=true
OPEN_POSITIONS_FILE=data/open_positions.jsonl
```

---

## Acceso desde tu PC local (SSH tunnel — alternativa a abrir el puerto 80)

Si preferís no exponer el puerto 80 públicamente:

```bash
ssh -L 8080:127.0.0.1:8501 usuario@<IP_VPS>
# Luego abrir: http://localhost:8080
```
