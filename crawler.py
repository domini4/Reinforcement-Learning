# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random
import numpy as np

def sarsa(env, logger):
    """
    Implement SARSA to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    # linear decay
    decay = eps/max_iterations
    #########################
#    import matplotlib.pyplot as plt
    # to hold Q(s,a)
    q = np.zeros((NUM_STATES, NUM_ACTIONS))
    exp = 0
#    lis=[]
    # collect eposides
    while exp < max_iterations:
        s = env.reset()
        # determine action using eps-greedy
        a = np.random.choice([np.random.choice([0,1,2,3],p=[.25,.25,.25,.25]), np.argmax(q[s])], p=[eps,1-eps])
        terminalFlag = False
        # collect an experience
        while not terminalFlag:
            s_, r, terminal, info = env.step(a)
            a_ = np.random.choice([np.random.choice([0,1,2,3],p=[.25,.25,.25,.25]), np.argmax(q[s_])], p=[eps,1-eps])
            if exp == max_iterations:
                break
#            lis.append(eps)
            # update q value after each experience
            if terminal == True:
                terminalFlag = True
                q[s][a] = q[s][a] + alpha*(r - q[s][a])
            else:
                q[s][a] = q[s][a] + alpha*(r + (gamma*(q[s_][a_])) - q[s][a])
            if eps > 0.1:
                eps -= decay
            s = s_
            a = a_
            exp += 1
        # determine best policy and values after each episode
        for st in range(NUM_STATES):
            pi[st] = np.argmax(q[st])
            v[st] = np.max(q[st])
        # Visualize the initial value and policy
        logger.log(exp, v, pi)
#    plt.plot(np.linspace(0,len(lis)-1,len(lis)),lis)    return None

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    # linear decay
    decay = eps/max_iterations
    #########################
#    import matplotlib.pyplot as plt
    # to hold Q(s,a)
    q = np.zeros((NUM_STATES, NUM_ACTIONS))
    exp = 0
#    lis=[]
    # collect eposides
    while exp < max_iterations:
        s=env.reset()
        terminalFlag = False
        # collect experiences
        while not terminalFlag:
            # determinig action using eps-greedy
            a = np.random.choice([np.random.choice([0,1,2,3],p=[.25,.25,.25,.25]), np.argmax(q[s])], p=[eps,1-eps])
            # collect the experience
            s_, r, terminal, info = env.step(a)
            if exp == max_iterations:
                break
#            lis.append(eps)
            exps = []
            # handle for terminal states
            if terminal == True:
                terminalFlag = True
                q[s][a] = q[s][a] + alpha*(r - q[s][a])
            else:
                for a_ in range(NUM_ACTIONS):
                    exps.append(q[s_][a_])
                q[s][a] = q[s][a] + alpha*(r + (gamma*max(exps)) - q[s][a])
            # lineraly decay the eps to 10% value
            if eps > 0.1:
                eps -= decay
            s = s_
            exp += 1
        # determin police and value
        for st in range(NUM_STATES):
            pi[st] = np.argmax(q[st])
            v[st] = np.max(q[st])
        # Visualize the initial value and policy
        logger.log(exp, v, pi)
#    plt.plot(np.linspace(0,len(lis)-1,len(lis)),lis)
    return None


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
         "SARSA": sarsa
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()