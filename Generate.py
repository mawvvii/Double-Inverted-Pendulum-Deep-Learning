import numpy as np
import random
from casadi import *
from casadi.tools import *
from pendulum_class import *
from controller import template_mpc

# initial state of the pendulum
i=0
while i < 1000:
    try:

        x0 = np.array([random.uniform(-0.9, 0.9),random.uniform(-np.pi,np.pi),random.uniform(-np.pi,np.pi),0.0,0.0,0.0]) # initial state of the pendulum

        # Load the pendulum
        pendulum = pendulum_simulator(x0)

        # load mpc controller and set initial state and guess
        mpc = template_mpc()
        mpc.x0 = x0
        mpc.set_initial_guess()

        #  simulate pendulum for random inputs
        for _ in range(250):

            # compute optimal control input via MPC
            u0 = mpc.make_step(x0)

            # Simulate pendulum
            x0 = pendulum.simulate(u0)
            if max(abs(x0[1:]))<1e-3:
                break;
            

        # Export data for learning
        pendulum.export_data('Train{}'.format(i))
    except:
        print("skipped")
        i=i-1
    i=i+1
