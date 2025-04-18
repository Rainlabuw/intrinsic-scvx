import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ClassicPoweredDescentModel
import matplotlib.pyplot as plt
from src.parameters import PoweredDescentParameters
from src.algorithms import ClassicSCvx

params = PoweredDescentParameters()

system = ClassicPoweredDescentModel(params=params)
algo = ClassicSCvx(params, system)

x, u = system.initialize_trajectory()

x_opt, u_opt = algo.run_proximal_method(x, u, verbose=True)