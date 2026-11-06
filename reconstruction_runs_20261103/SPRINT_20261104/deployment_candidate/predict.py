from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from sprint.deploy import predict_main
if __name__ == "__main__":
    raise SystemExit(predict_main())
