#!/usr/bin/env python
"""Phase-II landing training (tracker alone, or the PS2 policy through the landing CIL).

See ``ps2rl/phase2_ps2/landing_ps2_entry.py`` for the details and
``docker_batch/run_landing_phase2_train.sh`` for the GPU batch that trains both references.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps2rl.phase2_ps2.landing_ps2_entry import main  # noqa: E402

if __name__ == "__main__":
    main()
