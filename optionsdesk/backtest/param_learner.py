"""Optimizador walk-forward de parametros con recocido (annealing).

Cierra el loop que faltaba: el sistema ya MIDE el edge realizado por setup/grade
(attribution.py) y APAGA lo que pierde, pero nunca ajustaba los parametros que
generan las señales. Este modulo los ajusta a medida que la demo opera.

Mecanismo (champion / challenger walk-forward):
  - El "campeon" es el mejor set de parametros conocido (se despliega y mide).
  - Periodicamente se propone un "retador": una perturbacion del campeon. Se
    despliega EN VIVO para que genere trades reales bajo esos parametros.
  - Tras juntar muestra suficiente, se compara el edge realizado del retador vs
    el del campeon:
       · si el retador es mejor  → se PROMUEVE (nuevo campeon) y baja la exploracion;
       · si es peor o igual      → se REVIERTE y SUBE la exploracion (annealing).
  - "A medida que falla, varia mas": cada reversion sube la temperatura, que
    agranda el paso de perturbacion y la cantidad de parametros que se mueven.
    Cada promocion la baja (converge cuando encuentra terreno bueno).

Asignacion de credito: cuando un setup prueba edge negativo, el learner prioriza
perturbar los parametros que alimentan ESE setup (param_store.TUNABLES[*].setups),
en vez de mover todo a ciegas.

Todo es determinista dado un `random.Random` sembrado → testeable.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from optionsdesk.backtest.param_store import TUNABLES, default_params
from optionsdesk.performance.attribution import StrategyEdge, Verdict, attribute

logger = logging.getLogger(__name__)

# Margen de mejora (en "ARS-fitness") para promover un retador. Evita promover por
# ruido: el retador tiene que ganarle al campeon por algo, no empatar.
_PROMOTE_MARGIN = 1.0
# Muestra minima de trades cerrados bajo un set antes de juzgarlo.
_MIN_SAMPLE = 12
# Temperatura inicial y limites.
_TEMP_INIT = 0.35
_TEMP_MIN = 0.10
_TEMP_MAX = 1.00
# Probabilidad base de perturbar cada parametro; sube si su setup esta fallando.
_PERTURB_BASE = 0.5
_PERTURB_FAILING = 0.92
# Penalizaciones de la fitness.
_CALIB_PENALTY_W = 800.0   # pesos de fitness por punto de brecha de calibracion
_JUNK_PENALTY_W = 0.5      # fraccion del PnL de setups con edge negativo confirmado


# ── Estado persistente ────────────────────────────────────────────────────────

@dataclass
class LearnerState:
    champion: dict = field(default_factory=default_params)
    champion_fitness: Optional[float] = None
    challenger: Optional[dict] = None
    active: str = "champion"          # "champion" | "challenger" — cual esta desplegado
    deployed_at: Optional[str] = None  # ISO; baseline para scopear trades del set activo
    temperature: float = _TEMP_INIT
    iterations: int = 0
    consecutive_reverts: int = 0
    last_action: str = "init"
    note: str = ""
    history: list = field(default_factory=list)   # [{iter, action, fitness, temperature, ts}]

    def deployed_params(self) -> dict:
        """Los parametros que estan operando en vivo ahora."""
        if self.active == "challenger" and self.challenger is not None:
            return self.challenger
        return self.champion

    def to_dict(self) -> dict:
        return {
            "champion": self.champion,
            "champion_fitness": self.champion_fitness,
            "challenger": self.challenger,
            "active": self.active,
            "deployed_at": self.deployed_at,
            "temperature": self.temperature,
            "iterations": self.iterations,
            "consecutive_reverts": self.consecutive_reverts,
            "last_action": self.last_action,
            "note": self.note,
            "history": self.history[-50:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearnerState":
        base = default_params()
        champ = {**base, **(d.get("champion") or {})}
        chal = d.get("challenger")
        chal = {**base, **chal} if chal else None
        return cls(
            champion=champ,
            champion_fitness=d.get("champion_fitness"),
            challenger=chal,
            active=d.get("active", "champion"),
            deployed_at=d.get("deployed_at"),
            temperature=float(d.get("temperature", _TEMP_INIT)),
            iterations=int(d.get("iterations", 0)),
            consecutive_reverts=int(d.get("consecutive_reverts", 0)),
            last_action=d.get("last_action", "loaded"),
            note=d.get("note", ""),
            history=list(d.get("history", [])),
        )


# ── Fitness ────────────────────────────────────────────────────────────────────

def fitness(edges: list[StrategyEdge]) -> float:
    """Calidad de un set de parametros segun el edge realizado de sus trades.

    En unidades de "ARS de fitness": parte del PnL total realizado y penaliza
    (a) los setups con edge negativo confirmado y (b) la mala calibracion (cuando
    la PoP esperada no matchea el win rate real). Mayor = mejor.

    Sin trades → muy negativo (un set que no genera trades no sirve).
    """
    if not edges:
        return -1e9

    total = sum(e.total_pnl_ars for e in edges)
    junk = sum(
        abs(e.total_pnl_ars) for e in edges if e.verdict == Verdict.NO_EDGE
    ) * _JUNK_PENALTY_W
    calib = sum(
        abs(e.calibration_gap) for e in edges if e.calibration_gap is not None
    ) * _CALIB_PENALTY_W
    return round(total - junk - calib, 2)


def failing_setups(edges: list[StrategyEdge]) -> set[str]:
    """Setups con edge negativo confirmado (para asignacion de credito)."""
    return {e.strategy.upper() for e in edges if e.verdict == Verdict.NO_EDGE}


# ── Propuesta de retador ───────────────────────────────────────────────────────

def _param_is_failing(param_name: str, failing: set[str]) -> bool:
    """True si algun setup que depende de este parametro esta fallando."""
    spec = TUNABLES[param_name]
    for setup in spec.setups:
        su = setup.upper()
        if any(su in f or f in su for f in failing):
            return True
    return False


def propose_challenger(
    champion: dict,
    temperature: float,
    failing: set[str],
    rng: random.Random,
) -> dict:
    """Genera un retador perturbando el campeon.

    Cada parametro se mueve con probabilidad p (mayor si su setup esta fallando) y
    con un paso ~N(0, temperature*scale). Garantiza que al menos un parametro
    cambie (un retador identico al campeon no aporta informacion).
    """
    challenger = dict(champion)
    changed = False
    for name, spec in TUNABLES.items():
        p = _PERTURB_FAILING if _param_is_failing(name, failing) else _PERTURB_BASE
        if rng.random() >= p:
            continue
        step = rng.gauss(0.0, max(temperature, 0.05)) * spec.scale
        base = champion.get(name, spec.default)
        new_val = spec.clamp(base + step)
        if new_val != champion.get(name, spec.default):
            challenger[name] = new_val
            changed = True

    if not changed:
        # Forzar el movimiento de un parametro (prioriza los que fallan).
        candidates = [n for n in TUNABLES if _param_is_failing(n, failing)] or list(TUNABLES)
        name = rng.choice(candidates)
        spec = TUNABLES[name]
        step = (rng.choice([-1.0, 1.0])) * max(temperature, 0.2) * spec.scale
        challenger[name] = spec.clamp(champion.get(name, spec.default) + step)

    return challenger


# ── Avance del learner ─────────────────────────────────────────────────────────

def _anneal_up(temp: float) -> float:
    return round(min(_TEMP_MAX, temp * 1.5 + 0.1), 4)


def _anneal_down(temp: float) -> float:
    return round(max(_TEMP_MIN, temp * 0.6), 4)


def _scope_edges(trades: list, deployed_at: Optional[str], min_sample: int):
    """Atribuye solo los trades cerrados desde que el set activo se desplego.

    Devuelve (edges, n_settled).
    """
    cutoff: Optional[datetime] = None
    if deployed_at:
        try:
            cutoff = datetime.fromisoformat(deployed_at)
        except ValueError:
            cutoff = None

    scoped = []
    for t in trades:
        ca = getattr(t, "closed_at", None)
        if cutoff is not None and ca is not None:
            c, s = ca, cutoff
            if (c.tzinfo is None) != (s.tzinfo is None):
                c, s = c.replace(tzinfo=None), s.replace(tzinfo=None)
            if c < s:
                continue
        scoped.append(t)

    settled = [t for t in scoped if getattr(t, "realized_pnl_ars", None) is not None]
    edges = attribute(scoped, min_sample=min_sample) if scoped else []
    return edges, len(settled)


def advance_learner(
    state: LearnerState,
    trades: list,
    *,
    now: Optional[datetime] = None,
    min_sample: int = _MIN_SAMPLE,
    rng: Optional[random.Random] = None,
) -> LearnerState:
    """Avanza un paso del learner segun los trades cerrados acumulados.

    `trades`: list[ClosedTrade] (de attribution.load_all_closed_trades).
    Decide si junta mas muestra, promueve o revierte; deja el retador desplegado.
    """
    rng = rng or random.Random()
    current = now or datetime.now().astimezone()
    now_iso = current.isoformat(timespec="seconds")

    # Primera vez: arranca midiendo al campeon (defaults) desde ahora.
    if state.deployed_at is None:
        state.deployed_at = now_iso
        state.active = "champion"
        state.last_action = "init_baseline"
        state.note = "midiendo baseline del campeon"
        return state

    edges, n_settled = _scope_edges(trades, state.deployed_at, min_sample)

    if n_settled < min_sample:
        state.last_action = "gathering"
        state.note = (
            f"juntando muestra del set {state.active}: "
            f"{n_settled}/{min_sample} trades cerrados"
        )
        return state

    fit = fitness(edges)
    failing = failing_setups(edges)

    if state.active == "champion":
        # Termino de medir el baseline del campeon → lanza el primer retador.
        state.champion_fitness = fit
        state.challenger = propose_challenger(state.champion, state.temperature, failing, rng)
        state.active = "challenger"
        state.deployed_at = now_iso
        state.iterations += 1
        state.last_action = "deploy_challenger"
        state.note = f"campeon fitness={fit:.0f}; probando retador #{state.iterations}"
        _log_history(state, fit, current)
        return state

    # active == "challenger": comparar contra el campeon.
    champ_fit = state.champion_fitness if state.champion_fitness is not None else -1e9
    if fit > champ_fit + _PROMOTE_MARGIN:
        # PROMOVER: el retador gano → nuevo campeon, baja exploracion.
        state.champion = dict(state.challenger or state.champion)
        state.champion_fitness = fit
        state.temperature = _anneal_down(state.temperature)
        state.consecutive_reverts = 0
        state.iterations += 1
        state.last_action = "promote"
        state.note = (
            f"retador promovido (fitness {fit:.0f} > {champ_fit:.0f}); "
            f"temp baja a {state.temperature}"
        )
        # Lanzar de inmediato un nuevo retador desde el campeon mejorado.
        state.challenger = propose_challenger(state.champion, state.temperature, failing, rng)
        state.active = "challenger"
        state.deployed_at = now_iso
        _log_history(state, fit, current)
        return state

    # REVERTIR: el retador no mejoro → descartar, subir exploracion (annealing).
    state.temperature = _anneal_up(state.temperature)
    state.consecutive_reverts += 1
    state.iterations += 1
    state.last_action = "revert"
    state.note = (
        f"retador descartado (fitness {fit:.0f} ≤ {champ_fit:.0f}); "
        f"temp sube a {state.temperature} (reverts seguidos: {state.consecutive_reverts})"
    )
    # Nuevo retador desde el campeon, con mas exploracion.
    state.challenger = propose_challenger(state.champion, state.temperature, failing, rng)
    state.active = "challenger"
    state.deployed_at = now_iso
    _log_history(state, fit, current)
    return state


def _log_history(state: LearnerState, fit: float, now: datetime) -> None:
    state.history.append({
        "iter": state.iterations,
        "action": state.last_action,
        "fitness": round(fit, 2),
        "temperature": state.temperature,
        "ts": now.isoformat(timespec="seconds"),
    })
    state.history = state.history[-50:]


# ── Persistencia + driver de alto nivel ────────────────────────────────────────

import json as _json
from pathlib import Path as _Path

_STATE_FILE = _Path("data") / "param_learner_state.json"


def load_state(path: Optional[_Path] = None) -> LearnerState:
    """Carga el estado del learner desde disco (o uno fresco con los defaults)."""
    target = _Path(path) if path else _STATE_FILE
    if not target.exists():
        return LearnerState()
    try:
        return LearnerState.from_dict(_json.loads(target.read_text(encoding="utf-8")))
    except (OSError, _json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug("param_learner_state ilegible: %s", exc)
        return LearnerState()


def save_state(state: LearnerState, path: Optional[_Path] = None) -> None:
    target = _Path(path) if path else _STATE_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("no se pudo escribir param_learner_state: %s", exc)


def run_learner_step(
    *,
    data_dir: Optional[_Path] = None,
    now: Optional[datetime] = None,
    min_sample: int = _MIN_SAMPLE,
    rng: Optional[random.Random] = None,
    since=None,
) -> LearnerState:
    """Un paso completo del learner, listo para llamar desde el runner.

    1. Carga estado + trades cerrados.
    2. Avanza el learner (decide gather/promote/revert).
    3. Despliega los parametros activos al store (para que las señales los lean).
    4. Persiste el estado.
    """
    from optionsdesk.performance.attribution import load_all_closed_trades
    from optionsdesk.backtest.param_store import write_active_params

    state_path = _Path(data_dir) / "param_learner_state.json" if data_dir else _STATE_FILE
    state = load_state(state_path)
    trades = load_all_closed_trades(data_dir, since=since)

    state = advance_learner(state, trades, now=now, min_sample=min_sample, rng=rng)

    # Desplegar lo que esta operando en vivo (campeon o retador) al store.
    write_active_params(state.deployed_params(), data_dir=data_dir, source=f"learner:{state.active}")
    save_state(state, state_path)
    return state


def load_learner_status(path: Optional[_Path] = None) -> Optional[dict]:
    """Lee el estado del learner para el dashboard (sin instanciar la clase)."""
    target = _Path(path) if path else _STATE_FILE
    if not target.exists():
        return None
    try:
        return _json.loads(target.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return None
