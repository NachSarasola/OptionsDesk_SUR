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
| `optionsdesk-recorder` | Graba snapshots. Prefiere Primary WebSocket; con IOL REST aplica un piso conservador de 900s |
| `optionsdesk-demo-runner` | Corre el laboratorio demo `LAB_INFINITE` 24/7 durante rueda; selecciona top 20 por liquidez y aprende por forwardtesting |
| `optionsdesk-monitor.timer` | Dispara el monitor de posiciones cada 5 min (L-V 10:30-17:00) |

```bash
# Ver estado
systemctl status optionsdesk-dashboard
systemctl status optionsdesk-recorder
systemctl status optionsdesk-demo-runner
systemctl list-timers optionsdesk-monitor.timer

# Logs en vivo
journalctl -fu optionsdesk-dashboard
journalctl -fu optionsdesk-recorder
journalctl -fu optionsdesk-demo-runner

# Reiniciar manualmente
systemctl restart optionsdesk-dashboard
systemctl restart optionsdesk-recorder
systemctl restart optionsdesk-demo-runner
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
sudo systemctl restart optionsdesk-dashboard optionsdesk-recorder optionsdesk-demo-runner
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

### Recorder no conecta al feed de mercado

```bash
journalctl -n 50 -u optionsdesk-recorder
# Causas: .env mal configurado, host Primary incorrecto para el ALyC,
# credenciales vencidas o API IOL temporalmente no disponible.
cat /opt/optionsdesk/.env   # confirmar las variables del proveedor elegido
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
# Primary Trading API / Matriz OMS (recomendado para datos realtime)
# Sandbox gratuito: https://api.remarkets.primary.com.ar
# Produccion: usar el host xOMS informado por tu ALyC.
# Hasta homologar Primary, dejar IOL. Luego cambiar a PRIMARY o AUTO.
MARKET_DATA_PROVIDER=IOL
PRIMARY_BASE_URL=https://api.remarkets.primary.com.ar
PRIMARY_USER=tu_usuario
PRIMARY_PASSWORD=tu_password
# Opcionales: el provider descubre los instrumentos automaticamente.
PRIMARY_WS_URL=
PRIMARY_SPOT_SYMBOL=
PRIMARY_CAUCION_SYMBOL=

# IOL (fallback REST)
IOL_USER=
IOL_PASSWORD=
IOL_DASHBOARD_REFRESH_S=60
IOL_RECORDER_INTERVAL_S=900
IOL_OPTION_QUOTES_PER_CYCLE=8
IOL_OPTIONS_LIST_TTL_S=900
SPOT_TAPE_INTERVAL_S=60

# Laboratorio demo
STOCK_DEMO_MODE=LAB_INFINITE
STOCK_UNIVERSE_TOP_N=20
STOCK_UNIVERSE_VOLUME_LOOKBACK=20
LAB_INFINITE_CAPITAL_ARS=1000000000000
STRATEGY_TREE_MIN_DEPLOY_FITNESS=0
STRATEGY_TREE_MIN_DEPLOY_TRADES=12

# Telegram (opcional — alertas)
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=

# Ajustes opcionales
RECORDER_INTERVAL_S=15  # default 15s
MIN_TNA_SPREAD_PCT=5.0
HORIZON_MONITOR_ENABLED=true
OPEN_POSITIONS_FILE=data/open_positions.jsonl

# Obligatorio antes de habilitar tickets reales
COSTS_PROFILE=iol-public-ceiling-conservative-2026-06-02
COSTS_VERIFIED=false
STOCK_COMMISSION_PCT=1.00
OPTION_COMMISSION_PCT=1.00
EXERCISE_COMMISSION_PCT=1.00
STOCK_MARKET_FEE_PCT=0.050
OPTION_MARKET_FEE_PCT=0.200
EXERCISE_MARKET_FEE_PCT=0.050
IVA_RATE=0.21
```

Para validar Primary sin enviar ordenes:

```bash
python -m scripts.check_primary_readonly
```

Antes de operar manualmente en Matriz:

```bash
python -m scripts.check_operational_readonly
```

---

## Acceso desde tu PC local (SSH tunnel — alternativa a abrir el puerto 80)

Si preferís no exponer el puerto 80 públicamente:

```bash
ssh -L 8080:127.0.0.1:8501 usuario@<IP_VPS>
# Luego abrir: http://localhost:8080
```
