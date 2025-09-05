###################################################################################################
###################################################################################################

# GOAL: 

###################################################################################################
###################################################################################################

# Basic imports
import os, sys, socket, subprocess
from pathlib import Path

# Host-dependent paths
hostname = socket.gethostname()
if hostname == "Andreas-MacBook-Pro.local":
    os.environ["PROJECT_PATH"]  = "/Users/andreascalenghe/Documents/GitHub/Insurance_Demand_Jax"
    os.environ["OVERLEAF_PATH"] = ""
    os.environ["DROPBOX_PATH"]  = "/Users/andreascalenghe/Library/CloudStorage/Dropbox/PreDoc/Insurance_Demand_Jax"
elif hostname == "ECON-STAFF-ASSOC-2022MACSTUDIO-JYN7WTQ240":
    os.environ["PROJECT_PATH"]  = "/Users/as7596/Documents/GitHub/Insurance_Demand_Jax"
    os.environ["OVERLEAF_PATH"] = ""
    os.environ["DROPBOX_PATH"]  = "/Users/as7596/Dropbox/PreDoc/Insurance_Demand_Jax"
else:
    os.environ["PROJECT_PATH"] = "/Users/andreascalenghe/Documents/GitHub/prepostACA"

# Params
os.environ["SEED"] = str(12345)
os.environ["N"] = str(5000)

# Resolve paths relative to this script's location
# This allows running from any working directory without hardcoding paths
here = Path(__file__).resolve().parent
py = sys.executable
env = os.environ.copy()

# Resolve script paths relative to this file
script01 = (here / "01_Demand_graphs.py")

# Fallback to repo layout if not colocated
proj = Path(os.environ["PROJECT_PATH"])
if not script01.exists():
    script01 = proj / "01_Demand_graphs.py"

# Fail if paths are wrong
for p in [script01]:
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")

# Run as subprocess
subprocess.run([py, str(script01)], check=True, env=env, cwd=str(script01.parent))