from execution.position_manager import PositionManager
import json
pm = PositionManager()
pos = pm.get_open_positions()
print(json.dumps(pos, ensure_ascii=False, indent=2))
