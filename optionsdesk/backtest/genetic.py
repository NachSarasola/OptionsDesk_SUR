"""Motor Genético para Optimización Forward-Testing en Merval.

Mantiene una población de PaperBots iterando sobre el mercado en vivo.
"""
from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from optionsdesk.backtest.stock_demo import StockDemoPosition
from optionsdesk.data.macro import get_merval_trend, get_ccl_momentum

logger = logging.getLogger(__name__)

POPULATION_FILE = Path("data/genetic_population.json")


@dataclass
class Genome:
    """ADN del bot con parámetros mutables."""
    atr_stop_mult: float      # Ej: 1.5 a 3.0
    atr_tp_mult: float        # Ej: 1.5 a 5.0
    min_rr: float             # Reward-Risk exigido (ej: 1.5)
    require_merval_bull: bool # ¿Filtra si macro no es alcista?
    require_ccl_stable: bool  # ¿Evita operar en devaluación fuerte?
    
    def mutate(self) -> "Genome":
        """Genera un nuevo genoma con pequeñas mutaciones."""
        return Genome(
            atr_stop_mult=round(max(0.5, self.atr_stop_mult * random.uniform(0.8, 1.2)), 2),
            atr_tp_mult=round(max(1.0, self.atr_tp_mult * random.uniform(0.8, 1.2)), 2),
            min_rr=round(max(1.0, self.min_rr * random.uniform(0.9, 1.1)), 2),
            # 10% chance de mutar los booleanos
            require_merval_bull=not self.require_merval_bull if random.random() < 0.1 else self.require_merval_bull,
            require_ccl_stable=not self.require_ccl_stable if random.random() < 0.1 else self.require_ccl_stable,
        )

    @classmethod
    def random_genesis(cls) -> "Genome":
        return cls(
            atr_stop_mult=round(random.uniform(1.0, 3.0), 2),
            atr_tp_mult=round(random.uniform(2.0, 6.0), 2),
            min_rr=round(random.uniform(1.2, 3.0), 2),
            require_merval_bull=random.choice([True, False]),
            require_ccl_stable=random.choice([True, False])
        )


@dataclass
class PaperBot:
    """Individuo que opera en papel."""
    id: str
    generation: int
    genome: Genome
    capital: float = 1_000_000.0
    initial_capital: float = 1_000_000.0
    open_positions: list[StockDemoPosition] = field(default_factory=list)
    closed_trades: list[dict] = field(default_factory=list)
    
    def pnl_pct(self) -> float:
        return ((self.capital - self.initial_capital) / self.initial_capital) * 100.0
        
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "genome": self.genome.__dict__,
            "capital": self.capital,
            "initial_capital": self.initial_capital,
            "open_positions": [p.__dict__ for p in self.open_positions],
            "closed_trades": self.closed_trades
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperBot":
        bot = cls(
            id=data["id"],
            generation=data.get("generation", 0),
            genome=Genome(**data["genome"]),
            capital=data.get("capital", 1_000_000.0),
            initial_capital=data.get("initial_capital", 1_000_000.0),
            closed_trades=data.get("closed_trades", [])
        )
        bot.open_positions = [StockDemoPosition(**p) for p in data.get("open_positions", [])]
        return bot


class EvolutionEngine:
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.bots: list[PaperBot] = []
        self._load()
        if not self.bots:
            self._genesis()

    def _load(self):
        if not POPULATION_FILE.exists():
            return
        try:
            raw = json.loads(POPULATION_FILE.read_text(encoding="utf-8"))
            self.bots = [PaperBot.from_dict(b) for b in raw]
        except Exception as e:
            logger.error(f"Error cargando poblacion genetica: {e}")

    def _save(self):
        POPULATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        raw = [b.to_dict() for b in self.bots]
        POPULATION_FILE.write_text(json.dumps(raw, default=str, indent=2), encoding="utf-8")

    def _genesis(self):
        """Generación 0."""
        self.bots = [
            PaperBot(id=str(uuid.uuid4())[:8], generation=0, genome=Genome.random_genesis())
            for _ in range(self.population_size)
        ]
        self._save()

    def step(self, signals: list, quotes: dict):
        """Avanza un paso de la simulación. Los bots evalúan señales."""
        merval_trend = get_merval_trend()
        ccl_mom = get_ccl_momentum()
        
        for bot in self.bots:
            self._bot_tick(bot, signals, quotes, merval_trend, ccl_mom)
            
        self._save()

    def _bot_tick(self, bot: PaperBot, signals: list, quotes: dict, merval_trend: str, ccl_mom: str):
        # 1. Update open positions (trailing stops, etc) - simplificado
        survivors = []
        for pos in bot.open_positions:
            q = quotes.get(pos.symbol)
            if not q:
                survivors.append(pos)
                continue
                
            current_price = q.last or q.bid
            
            # Check TP / SL
            if current_price >= pos.target_price or current_price <= pos.stop_price:
                pnl = (current_price - pos.entry_price) * pos.quantity
                bot.capital += pnl
                bot.closed_trades.append({
                    "symbol": pos.symbol,
                    "entry_price": pos.entry_price,
                    "exit_price": current_price,
                    "pnl": pnl
                })
            else:
                survivors.append(pos)
        
        bot.open_positions = survivors
        
        # 2. Filter new signals
        open_symbols = {p.symbol for p in bot.open_positions}
        for sig in signals:
            if sig.symbol in open_symbols:
                continue
                
            # Aplicar filtros genéticos
            if bot.genome.require_merval_bull and merval_trend != "alcista":
                continue
            if bot.genome.require_ccl_stable and ccl_mom == "apreciacion":
                continue
                
            # Dinámica de Stops (SMC baseline)
            # Re-calculamos TP y SL usando los multiplicadores del genoma
            # Asumimos que signal.stop_price original era ~1 ATR
            risk_base = abs(sig.entry_price - sig.stop_price)
            if risk_base == 0:
                continue
                
            new_stop = sig.entry_price - (risk_base * bot.genome.atr_stop_mult)
            new_tp = sig.entry_price + (risk_base * bot.genome.atr_tp_mult)
            
            new_rr = abs(new_tp - sig.entry_price) / abs(sig.entry_price - new_stop)
            if new_rr < bot.genome.min_rr:
                continue
                
            # Buy! (10% del capital)
            qty = (bot.capital * 0.10) / sig.entry_price
            if qty < 1:
                continue
                
            pos = StockDemoPosition(
                id=str(uuid.uuid4())[:8],
                symbol=sig.symbol,
                strategy="GENETIC",
                signal_type=sig.signal_type,
                entry_ts=datetime.now(),
                entry_price=sig.entry_price,
                quantity=int(qty),
                stop_price=new_stop,
                target_price=new_tp,
                entry_fee_ars=0.0,
                time_stop_min=0,
                max_holding_days=30,
                rationale=f"Gen:{bot.generation}",
                score=sig.score
            )
            bot.open_positions.append(pos)

    def prune_and_breed(self):
        """Mata a los perdedores y clona a los ganadores."""
        # Ordenar por PnL
        self.bots.sort(key=lambda b: b.pnl_pct(), reverse=True)
        
        half = self.population_size // 2
        winners = self.bots[:half]
        
        new_generation = []
        # Conservar ganadores
        for w in winners:
            new_generation.append(w)
            
            # Crear un hijo mutante
            child_genome = w.genome.mutate()
            child = PaperBot(
                id=str(uuid.uuid4())[:8],
                generation=w.generation + 1,
                genome=child_genome,
                capital=1_000_000.0,
                initial_capital=1_000_000.0
            )
            new_generation.append(child)
            
        self.bots = new_generation
        self._save()
        logger.info("Genetic pruning complete.")
