from dataclasses import dataclass, field
from typing import List

@dataclass
class BudgetConfig:
    B_TASK_TOTAL: float = 0.10
    
    P_IN: float = 0.5e-6
    P_OUT: float = 3.0e-6

    ALPHA: float = 1.2
    
    B_IN_BUCKETS: List[int] = field(default_factory=lambda: [64, 128, 256, 384, 512, 768])

@dataclass
class HESNConfig:
    K_VALUES: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 8])
    MAX_INPUT_TOKENS: int = 8196
    MAX_GOAL_TOKENS: int = 64
    DEFAULT_TOTAL_STEPS: int = 5
    TOP_K_CANDIDATES: int = 20
    SET_TRANSFORMER_LAYERS: int = 2
    HIDDEN_DIM: int = 1024
    TASK_EMBED_DIM: int = 32
    NUM_HEADS: int = 4

@dataclass
class CompressorConfig:
    FALLBACK_TEXT_LEN: int = 500
    DEFAULT_PARAGRAPHS: int = 4
    FALLBACK_WINDOW_LINES: int = 15
    FALLBACK_LOG_LINES: int = 20

@dataclass
class PoolConfig:
    MAX_POOL_SIZE: int = 200
    
    RECENCY_DECAY_WEEKS: int = 7

@dataclass
class MSIRConfig:
    AVG_STEP_COST: float = 0.00125
    DEFAULT_TOTAL_BUDGET: float = 0.01

@dataclass
class AnytimeConfig:
    NUM_RESEND_CLUES: int = 2
    RESEND_CLUE_LENGTH: int = 200

@dataclass
class LLMConfig:
    API_KEY: str = ""
    BASE_URL: str = ""

@dataclass
class SystemConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    msir: MSIRConfig = field(default_factory=MSIRConfig)
    anytime: AnytimeConfig = field(default_factory=AnytimeConfig)
    hesn: HESNConfig = field(default_factory=HESNConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

config = SystemConfig()
