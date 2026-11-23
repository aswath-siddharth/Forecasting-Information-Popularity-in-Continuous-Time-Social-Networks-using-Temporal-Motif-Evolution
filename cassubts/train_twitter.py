"""CasSubTS Twitter training entrypoint."""
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_w.train_twitter import main

if __name__ == "__main__":
    main()
