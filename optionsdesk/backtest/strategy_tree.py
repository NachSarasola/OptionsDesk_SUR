"""Busqueda en arbol del espacio de estrategias, condicionada por contexto.

No busca "la mejor estrategia global" — busca la MEJOR ESTRATEGIA POR REGIMEN.
Cada genoma evaluado reporta P&L por contexto (strategy_backtest); el arbol guarda,
para cada regimen, el genoma especialista que mejor rindio ahi. El resultado es una
POLITICA `contexto → estrategia`: en vivo se mira el regimen actual y se despliega
el genoma adecuado.

Por que arbol/evolutivo y no grid: el espacio (7 params continuos/enteros + 3 toggles)
es demasiado grande para grilla. Se expande por mutacion desde la frontera de
los mejores nodos — incluyendo los ESPECIALISTAS de cada regimen, no solo el mejor
global — para preservar diversidad. Asi una estrategia mediocre en promedio pero
excelente en 'bajista/vol-alta' sobrevive y se refina en su nicho.

"Disminuye cualquier trade off": en vez de un genoma de compromiso, se mantiene una
frontera de especialistas (uno por contexto) y la fitness penaliza el drawdown — el
optimo de cada nicho no es un config explosivo.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from optionsdesk.backtest.param_store import TUNABLES, FLAGS, write_active_params
from optionsdesk.backtest.strategy_backtest import (
    GenomeResult, RegimeResult, StrategyGenome, evaluate_genome,
)
from optionsdesk.backtest.strategy_context import all_regimes, regime_label_es

logger = logging.getLogger(__name__)

_TREE_FILE = Path("data") / "strategy_tree.json"
_MIN_REGIME_SAMPLE = 6     # trades minimos para confiar en un especialista
_MIN_DEPLOY_FITNESS = 0.0
_MIN_DEPLOY_TRADES = 12
_MUTATE_PARAM_PROB = 0.5
_MUTATE_FLAG_PROB = 0.25


# ── Mutacion ────────────────────────────────────────────────────────────────

def mutate(genome: StrategyGenome, rng: random.Random, sigma: float = 0.4) -> StrategyGenome:
    """Genera un hijo perturbando params y flags del genoma padre.

    Garantiza: (a) al menos un cambio, (b) al menos un setup activo (no genoma vacio).
    """
    params = dict(genome.params)
    flags = dict(genome.flags)
    changed = False

    for name, spec in TUNABLES.items():
        if rng.random() < _MUTATE_PARAM_PROB:
            step = rng.gauss(0.0, max(sigma, 0.05)) * spec.scale
            new_val = spec.clamp(params.get(name, spec.default) + step)
            if new_val != params.get(name, spec.default):
                params[name] = new_val
                changed = True

    for name in FLAGS:
        if rng.random() < _MUTATE_FLAG_PROB:
            flags[name] = not flags.get(name, True)
            changed = True

    # No permitir un genoma sin setups (no operaria nada).
    if not any(flags.values()):
        flags[rng.choice(list(FLAGS))] = True

    if not changed:
        name = rng.choice(list(TUNABLES))
        spec = TUNABLES[name]
        params[name] = spec.clamp(
            params.get(name, spec.default) + rng.choice([-1.0, 1.0]) * sigma * spec.scale
        )

    return StrategyGenome(params=params, flags=flags)


# ── Nodos y politica ──────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    id: int
    parent_id: Optional[int]
    genome: StrategyGenome
    result: GenomeResult
    depth: int


@dataclass
class StrategyPolicy:
    """Mapeo regimen → genoma especialista + metadatos para el dashboard/deploy."""
    by_regime: dict           # regime -> {"genome": StrategyGenome, "metrics": RegimeResult, "specialist": bool}
    best_overall: StrategyGenome
    overall_fitness: float
    evaluated: int

    def genome_for(self, regime: str) -> StrategyGenome:
        slot = self.by_regime.get(regime)
        if slot is not None:
            return slot["genome"]
        return self.best_overall


# ── Busqueda ───────────────────────────────────────────────────────────────────

@dataclass
class SearchOutcome:
    policy: StrategyPolicy
    nodes: list           # list[TreeNode]
    best_overall: TreeNode


def _update_best_by_regime(best: dict, result: GenomeResult) -> None:
    for regime, rr in result.by_regime.items():
        if rr.n <= 0:
            continue
        cur = best.get(regime)
        if cur is None or rr.fitness > cur[1].fitness:
            best[regime] = (result.genome, rr)


def _assemble_policy(
    best_by_regime: dict,
    best_overall_node: TreeNode,
    evaluated: int,
) -> StrategyPolicy:
    by_regime: dict = {}
    for regime in all_regimes():
        slot = best_by_regime.get(regime)
        # Especialista valido: muestra suficiente y edge positivo en ESE contexto.
        if slot is not None and slot[1].n >= _MIN_REGIME_SAMPLE and slot[1].fitness > 0:
            by_regime[regime] = {"genome": slot[0], "metrics": slot[1], "specialist": True}
        else:
            # Sin especialista confiable → cae al mejor global (no a un perdedor).
            by_regime[regime] = {
                "genome": best_overall_node.genome,
                "metrics": slot[1] if slot else None,
                "specialist": False,
            }
    return StrategyPolicy(
        by_regime=by_regime,
        best_overall=best_overall_node.genome,
        overall_fitness=best_overall_node.result.overall.fitness,
        evaluated=evaluated,
    )


def search(
    histories: dict,
    *,
    weeklies: Optional[dict] = None,
    max_evals: int = 40,
    beam: int = 3,
    children: int = 3,
    sigma: float = 0.4,
    rng: Optional[random.Random] = None,
    seed_genomes: Optional[list] = None,
    step: int = 1,
    max_days: Optional[int] = None,
) -> SearchOutcome:
    """Explora el espacio de estrategias y devuelve la politica por regimen.

    histories/weeklies: dict[symbol -> DataFrame] (de strategy_backtest.load_histories).
    max_evals: tope de genomas evaluados (cada eval replaya el historico).
    beam: cuantos mejores nodos expandir por iteracion (+ los especialistas).
    """
    rng = rng or random.Random()
    nodes: list[TreeNode] = []
    evaluated_keys: dict = {}
    best_by_regime: dict = {}
    _next_id = 0

    def _eval(genome: StrategyGenome, parent_id, depth) -> Optional[TreeNode]:
        nonlocal _next_id
        k = genome.key()
        if k in evaluated_keys:
            return evaluated_keys[k]
        res = evaluate_genome(genome, histories, weeklies=weeklies, step=step, max_days=max_days)
        node = TreeNode(id=_next_id, parent_id=parent_id, genome=genome, result=res, depth=depth)
        _next_id += 1
        nodes.append(node)
        evaluated_keys[k] = node
        _update_best_by_regime(best_by_regime, res)
        return node

    # Semillas: defaults + cualquier genoma sembrado (ej. campeon del param_learner).
    roots = [StrategyGenome.default()] + list(seed_genomes or [])
    for g in roots:
        if len(evaluated_keys) >= max_evals:
            break
        _eval(g, None, 0)

    # Expansion evolutiva.
    while len(evaluated_keys) < max_evals:
        ranked = sorted(nodes, key=lambda nd: nd.result.overall.fitness, reverse=True)
        specialists = [
            evaluated_keys[g.key()]
            for g, _ in best_by_regime.values()
            if g.key() in evaluated_keys
        ]
        # Frontera: top-beam global + especialistas (dedup por id).
        seen_ids = set()
        frontier: list[TreeNode] = []
        for nd in ranked[:beam] + specialists:
            if nd.id not in seen_ids:
                seen_ids.add(nd.id)
                frontier.append(nd)

        progressed = False
        for parent in frontier:
            for _ in range(children):
                if len(evaluated_keys) >= max_evals:
                    break
                child = mutate(parent.genome, rng, sigma)
                before = len(evaluated_keys)
                _eval(child, parent.id, parent.depth + 1)
                if len(evaluated_keys) > before:
                    progressed = True
        if not progressed:
            break   # espacio agotado (todo duplicado)

    best_overall_node = max(nodes, key=lambda nd: nd.result.overall.fitness)
    policy = _assemble_policy(best_by_regime, best_overall_node, len(evaluated_keys))
    return SearchOutcome(policy=policy, nodes=nodes, best_overall=best_overall_node)


# ── Persistencia + deploy ──────────────────────────────────────────────────────

def _genome_to_dict(g: StrategyGenome) -> dict:
    return {"params": g.params, "flags": g.flags}


def _genome_from_dict(d: dict) -> StrategyGenome:
    base = StrategyGenome.default()
    return StrategyGenome(
        params={**base.params, **(d.get("params") or {})},
        flags={**base.flags, **(d.get("flags") or {})},
    )


def save_policy(outcome: SearchOutcome, path: Optional[Path] = None) -> None:
    """Persiste la politica + un resumen del arbol para el dashboard."""
    target = Path(path) if path else _TREE_FILE
    pol = outcome.policy
    regimes_out = {}
    for regime, slot in pol.by_regime.items():
        m: Optional[RegimeResult] = slot.get("metrics")
        regimes_out[regime] = {
            "label": regime_label_es(regime),
            "genome": _genome_to_dict(slot["genome"]),
            "specialist": slot["specialist"],
            "n": m.n if m else 0,
            "fitness": m.fitness if m else None,
            "total_pnl_ars": m.total_pnl_ars if m else None,
            "win_rate": m.win_rate if m else None,
        }
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluated": pol.evaluated,
        "overall_fitness": pol.overall_fitness,
        "overall_n": outcome.best_overall.result.overall.n,
        "deployable": (
            pol.overall_fitness > _MIN_DEPLOY_FITNESS
            and outcome.best_overall.result.overall.n >= _MIN_DEPLOY_TRADES
        ),
        "best_overall": _genome_to_dict(pol.best_overall),
        "regimes": regimes_out,
        "tree": [
            {
                "id": nd.id, "parent": nd.parent_id, "depth": nd.depth,
                "fitness": nd.result.overall.fitness, "n": nd.result.overall.n,
                "regimes": {
                    reg: {
                        "n": rr.n,
                        "fitness": rr.fitness,
                        "win_rate": rr.win_rate,
                        "profit_factor": rr.profit_factor,
                        "drawdown_ars": rr.max_drawdown_ars,
                    }
                    for reg, rr in nd.result.by_regime.items()
                    if rr.n > 0
                },
            }
            for nd in outcome.nodes
        ][:200],
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("no se pudo escribir strategy_tree: %s", exc)


def load_policy_doc(path: Optional[Path] = None) -> Optional[dict]:
    """Lee el documento de politica (para dashboard y deploy en vivo)."""
    target = Path(path) if path else _TREE_FILE
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def policy_deployability(
    doc: dict,
    *,
    min_fitness: Optional[float] = None,
    min_trades: Optional[int] = None,
) -> tuple[bool, str]:
    """Verifica si una politica puede pasar de research a runtime."""
    try:
        fitness = float(doc.get("overall_fitness", 0.0) or 0.0)
    except (TypeError, ValueError):
        fitness = 0.0
    try:
        n = int(doc.get("overall_n", 0) or 0)
    except (TypeError, ValueError):
        n = 0
    min_fit = _MIN_DEPLOY_FITNESS if min_fitness is None else float(min_fitness)
    min_n = _MIN_DEPLOY_TRADES if min_trades is None else int(min_trades)
    if fitness <= min_fit:
        return False, f"fitness {fitness:.0f} <= {min_fit:.0f}"
    if n < min_n:
        return False, f"muestra {n} < {min_n}"
    return True, "deployable"


def deploy_for_regime(
    regime: str,
    *,
    data_dir: Optional[Path] = None,
    policy_path: Optional[Path] = None,
    require_deployable: bool = True,
) -> Optional[str]:
    """Despliega al store el genoma de la politica para el regimen dado.

    Lo llama el runner cada ciclo con el regimen vivo. Devuelve el genoma desplegado
    como string (para log/status) o None si no hay politica.
    """
    doc = load_policy_doc(policy_path)
    if not doc:
        return None
    if require_deployable:
        try:
            from optionsdesk.config.settings import settings as _s
            min_fitness = getattr(_s, "strategy_tree_min_deploy_fitness", _MIN_DEPLOY_FITNESS)
            min_trades = getattr(_s, "strategy_tree_min_deploy_trades", _MIN_DEPLOY_TRADES)
        except Exception:
            min_fitness, min_trades = _MIN_DEPLOY_FITNESS, _MIN_DEPLOY_TRADES
        ok, reason = policy_deployability(
            doc, min_fitness=min_fitness, min_trades=min_trades,
        )
        if not ok:
            logger.info("strategy_tree no desplegado: %s", reason)
            return None
    regimes = doc.get("regimes", {})
    slot = regimes.get(regime) or {"genome": doc.get("best_overall")}
    genome = slot.get("genome") if slot else doc.get("best_overall")
    if not genome:
        return None
    params = genome.get("params", {})
    flags = genome.get("flags", {})
    write_active_params(params, flags=flags, data_dir=data_dir, source=f"tree:{regime}")
    return f"{regime} ({'especialista' if slot.get('specialist') else 'global'})"


# ── CLI: correr la optimizacion del arbol ──────────────────────────────────────

def run_optimization(
    symbols: Optional[list[str]] = None,
    *,
    days: int = 240,
    max_evals: int = 40,
    seed: Optional[int] = None,
    allow_synthetic: bool = False,
) -> SearchOutcome:
    """Carga el historico, corre la busqueda y persiste la politica. Devuelve el outcome.

    Sembra la busqueda con el campeon del param_learner (si existe) para no arrancar
    de cero. Pensado para correr nightly / on-demand (cada eval replaya el historico).
    """
    from optionsdesk.backtest.strategy_backtest import load_histories
    from optionsdesk.config.settings import settings as _s

    syms = symbols or sorted(s.upper() for s in _s.stock_universe_symbols()) or ["GGAL"]
    dailies, weeklies = load_histories(syms, days=days, allow_synthetic=allow_synthetic)
    if not dailies:
        raise RuntimeError("Sin historico para optimizar (revisa la fuente de datos).")

    seeds = []
    try:
        from optionsdesk.backtest.param_learner import load_state
        champ = load_state().champion
        seeds.append(StrategyGenome(params=champ, flags=StrategyGenome.default().flags))
    except Exception:
        pass

    outcome = search(
        dailies, weeklies=weeklies, max_evals=max_evals,
        rng=random.Random(seed), seed_genomes=seeds,
    )
    save_policy(outcome)
    return outcome


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Optimizador de estrategias por contexto (arbol).")
    p.add_argument("--symbols", type=str, default="", help="lista separada por comas; vacio = universo")
    p.add_argument("--days", type=int, default=240, help="ruedas de historico")
    p.add_argument("--evals", type=int, default=40, help="genomas a evaluar")
    p.add_argument("--seed", type=int, default=None, help="semilla RNG (reproducible)")
    p.add_argument("--allow-synthetic", action="store_true", help="solo smoke tests; no usar para research")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    outcome = run_optimization(
        syms,
        days=args.days,
        max_evals=args.evals,
        seed=args.seed,
        allow_synthetic=args.allow_synthetic,
    )

    print(f"\nEvaluados: {outcome.policy.evaluated} genomas")
    print(f"Mejor fitness global: {outcome.best_overall.result.overall.fitness:,.0f}\n")
    print("Estrategia optima por regimen:")
    for regime, slot in outcome.policy.by_regime.items():
        m = slot.get("metrics")
        if m and m.n > 0:
            tag = "especialista" if slot["specialist"] else "global (fallback)"
            print(f"  {regime_label_es(regime):24s}  n={m.n:3d}  fitness={m.fitness:10,.0f}  [{tag}]")
    print("\nPolitica guardada en data/strategy_tree.json")


if __name__ == "__main__":
    main()
