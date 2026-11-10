"""Run a small example by default, or pass options shown by --help."""
from datetime import datetime
from pathlib import Path
import sys
from cfsc.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        output = Path("outputs") / ("demo-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        args = ["--scenario", "zipf", "--n", "1000", "--d", "32",
                "--epsilon", "2", "--seeds", "0", "1", "2", "--output", str(output)]
        print(f"Results: {output}", flush=True)
    raise SystemExit(main(args))
