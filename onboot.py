import subprocess
import sys


try:
    import pandas
    import json
    import pytz
    import matplotlib
    import numpy
    import scipy
except ImportError:
    print("Some packages are missing. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

