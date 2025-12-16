"""
MedForge Configuration

Global settings for the parallel LLM document processing pipeline.
"""

import os
from pathlib import Path

# =============================================================================
# Directory Configuration
# =============================================================================

# Base directories - configure these for your environment
ROOT_DIR = Path(os.environ.get("MEDFORGE_ROOT", Path(__file__).parent))
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = ROOT_DIR / "data"

# Subdirectories
PPT_DIR = DATA_DIR / "ppt"
TEXTBOOK_DIR = DATA_DIR / "textbooks"
EXERCISES_DIR = DATA_DIR / "exercises"

# =============================================================================
# Output Naming
# =============================================================================

# Chapter-level markdown suffix (without extension) for solved exercises output.
# Keep as a single source of truth to avoid downstream mismatches.
EXERCISES_CHAPTER_SUFFIX = "_exercises_full"

# Backward-compatible suffixes that may exist from older versions.
LEGACY_EXERCISES_CHAPTER_SUFFIXES = ("_exercises_solved",)

# =============================================================================
# API Configuration
# =============================================================================

# Load API keys from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# =============================================================================
# Parallel Execution Settings
# =============================================================================

NUM_PROCESSES = int(os.environ.get("MEDFORGE_PROCESSES", 8))
THREADS_PER_PROCESS = int(os.environ.get("MEDFORGE_THREADS", 4))

# =============================================================================
# LLM Model Configuration
# =============================================================================

# Default model for processing
DEFAULT_MODEL = "gemini-1.5-pro"

# Fallback models with priority order (highest to lowest)
FALLBACK_MODELS = [
    {
        "name": "gemini-1.5-pro",
        "provider": "google",
        "env_var": "GEMINI_API_KEY",
    },
    {
        "name": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "env_var": "ANTHROPIC_API_KEY",
    },
    {
        "name": "gpt-4o",
        "provider": "openai",
        "env_var": "OPENAI_API_KEY",
    },
]

# =============================================================================
# Smart Routing Configuration
# =============================================================================

# Retry settings per model before switching to fallback
RETRIES_PER_MODEL = 3
MODEL_RETRY_DELAY = 2  # seconds

# Keywords indicating quota exhaustion
QUOTA_KEYWORDS = ["quota", "insufficient", "balance", "credit", "rate_limit", "429"]

# Routing strategy
ROUTING_CONFIG = {
    "requests_per_retry": 10,      # Try primary model again after N requests
    "min_cooldown": 30,            # Minimum cooldown (seconds)
    "max_cooldown": 1800,          # Maximum cooldown (30 minutes)
    "fallback_retry_delay": 600,   # Try primary again after 10 minutes on fallback
}

# =============================================================================
# Processing Configuration
# =============================================================================

MAX_RETRIES = 3
RETRY_DELAY = 2

# Subject configuration file
SUBJECT_CONFIG_FILE = OUTPUT_DIR / "subject_config.json"

# Logging
LOG_DIR = ROOT_DIR / "logs"
ERROR_LOG = LOG_DIR / "error_log.txt"

# =============================================================================
# Initialization
# =============================================================================

def init_directories():
    """Create necessary directories if they don't exist."""
    for directory in [OUTPUT_DIR, DATA_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Auto-initialize on import
init_directories()
