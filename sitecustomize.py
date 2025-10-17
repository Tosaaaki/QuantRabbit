"""Process-wide defaults for local/dev execution."""
import os

os.environ.setdefault("GRPC_ALTS_ENABLED", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
