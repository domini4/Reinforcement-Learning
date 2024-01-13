# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
#
#
# Modified by: James Dominic for CPSC 8100 Deep Reinforcement learning Project 1
#
#
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In mc_control, sarsa, q_learning, and double q-learning once a terminal state is reached, 
the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import math
import numpy as np

def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the infinity norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    pi = [0] * NUM_STATES
    # initialize np arrays to store state k and k-1
    vk = np.zeros((1, NUM_STATES))
    vk1 = np.zeros((1, NUM_STATES))
    # impliment value iteration algorith. 
    #This for loop controls no. of iterations
    for k in range(max_iterations):
        # reseting vk before recalculating
        vk = np.zeros((1, NUM_STATES))
        # this for loop goes through each state
        for s in range(NUM_STATES):
            # initialize array to store reward values for all possible state-action combination.
            acVals = []
            # this for loop calculates rewards for each action possible in state s.
            for a in range(NUM_ACTIONS):
                t = TRANSITION_MODEL[s][a]
                vAction = 0
                # calculates rewards for each state-action
                for j in t:
                    # handling terminal condition
                    if len(t) == 1:
                        vAction +=  j[0]*(j[2])
                    else:
                        vAction +=  j[0]*(j[2] + gamma*vk1[0][j[1]])
                acVals.append(vAction)
            # selects the max reward from all possible rewards
            vk[0][s] = max(acVals)
            # saves policy (action) that gives max reward
            pi[s] = acVals.index(max(acVals))
        # checks for convergance of value.
        if np.linalg.norm(vk1-vk, ord=np.inf) < 0.0001:
            break
        # storing values for k-1
        vk1=vk
        # Visualize the value and policy
        logger.log(k+1, list(vk), pi)
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the infinity norm 
        norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of values by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    pi = np.random.randint(0, NUM_ACTIONS, NUM_STATES)
    # initialize np arrays to store state k and k-1
    vk = np.zeros((1, NUM_STATES))
    vk1 = np.zeros((1, NUM_STATES))
    pik = np.random.randint(0, NUM_ACTIONS, NUM_STATES)
    # impliment value iteration algorith. 
    #This for loop controls no. of iterations
    for k in range(max_iterations):
        scheck=1.0
        # value optimization for current policy
        while scheck > 0.0001:
            # reinitializing current state space.
            vk = np.zeros((1, NUM_STATES))
            # this for loop goes through each state
            for s in range(NUM_STATES):
                t = TRANSITION_MODEL[s][pik[s]]
                vAction = 0
                # calculate total reward for current policy
                for j in t:
                    # handling terminal cases
                    if len(t) == 1:
                        vAction +=  j[0]*(j[2])
                    else:
                        vAction +=  j[0]*(j[2] + gamma*vk1[0][j[1]])
                vk[0][s] = vAction
            # checking for convergence of value for current policy
            scheck = np.linalg.norm(vk1-vk, ord=np.inf)
            # saving k-1 values
            vk1=vk
        # reinitializing policy
        pi = list(np.random.randint(0, NUM_ACTIONS, NUM_STATES))
        # policiy improvement algorithm
        for s2 in range(NUM_STATES):
            acVals2 = []
            # this for loop calculates rewards for each action possible in state s.
            for a2 in range(NUM_ACTIONS):
                t2 = TRANSITION_MODEL[s2][a2]
                vAction2 = 0
                # calculates rewards for each state-action
                for j2 in t2:
                    #handling terminal cases
                    if len(t2) == 1:
                        vAction2 +=  j2[0]*(j2[2])
                    else:
                        vAction2 +=  j2[0]*(j2[2] + gamma*vk1[0][j2[1]])
                acVals2.append(vAction2)
            # selects the max reward from all possible rewards
            pi[s2] = acVals2.index(max(acVals2))
        # checking for policy convergence
        if (pik==pi).all():
            break
        # saving k-1 policy
        pik = np.array(pi)
        # Visualize the initial value and policy
        logger.log(k+1, list(vk), list(pi))
    return list(pi)

def on_policy_mc_control(env, gamma, max_iterations, logger):
    """
    Implement on-policy first visiti Monte Carlo control to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1.0
    # linear decay
    decay = eps/max_iterations
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################
    # to hold Q(s,a)
    q = np.zeros((NUM_STATES, NUM_ACTIONS))
    # initialize experience count
    exp = 0
    # preparing actions and probabilities for exploration
    aEqual = list(range(0,NUM_ACTIONS))
    pEqual = 1/NUM_ACTIONS
    # run eposides
    while exp < max_iterations:
        s = env.reset()
        terminalFlag = False
        exps = []
        # collect experiences
        while not terminalFlag:
            # get eps-greedy action
            a = np.random.choice([np.random.choice(aEqual,p=[pEqual] * NUM_ACTIONS), np.argmax(q[s])], p=[eps,1-eps])
            # get details of next step
            s_, r, terminal, info = env.step(a)
            if exp == max_iterations:
                break
            # decay esp until 90%
            if eps > 0.1:
                eps -= decay
            # storing the experience
            exps.append([s,a,r,s_])
            exp += 1
            if terminal == True:
                terminalFlag = True
            # select next state
            s=s_
        # list to check new experiences
        stateNew = []
        # calculate Gt and Q values for new experiences
        for k in range(len(exps)):
            # check if experience is new
            if ([exps[k][0],exps[k][1]]) not in stateNew:
                stateNew.append([exps[k][0],exps[k][1]])
                # calculate Gt
                gt = 0
                for t in range(len(exps)):
                    gt += exps[t][2]*np.power(gamma, t)
                # calculate Q value
                q[exps[k][0]][exps[k][1]] = q[exps[k][0]][exps[k][1]] + alpha*(gt - q[exps[k][0]][exps[k][1]])
        # calculate policy and V values
        for st in range(NUM_STATES):
            pi[st] = np.argmax(q[st])
            v[st] = np.max(q[st])
        # Visualize the initial value and policy
        logger.log(exp, v, pi)
    return pi


def sarsa(env, gamma, max_iterations, logger):
    """
    Implement SARSA to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # linear decay
    decay = eps/max_iterations
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################
    # to hold Q(s,a)
    q = np.zeros((NUM_STATES, NUM_ACTIONS))
    exp = 0
    # preparing actions and probabilities for exploration
    aEqual = list(range(0,NUM_ACTIONS))
    pEqual = 1/NUM_ACTIONS
    # collect eposides
    while exp < max_iterations:
        s = env.reset()
        # determine action using eps-greedy
        a = np.random.choice([np.random.choice(aEqual,p=[pEqual] * NUM_ACTIONS), np.argmax(q[s])], p=[eps,1-eps])
        terminalFlag = False
        # collect an experience
        while not terminalFlag:
            s_, r, terminal, info = env.step(a)
            a_ = np.random.choice([np.random.choice(aEqual,p=[pEqual] * NUM_ACTIONS), np.argmax(q[s_])], p=[eps,1-eps])
            if exp == max_iterations:
                break
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
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # linear decay
    decay = eps/max_iterations

    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################
    # to hold Q(s,a)
    q = np.zeros((NUM_STATES, NUM_ACTIONS))
    exp = 0
    # collect eposides
    while exp < max_iterations:
        s=env.reset()
        terminalFlag = False
        # preparing actions and probabilities for exploration
        aEqual = list(range(0,NUM_ACTIONS))
        pEqual = 1/NUM_ACTIONS
        # collect experiences
        while not terminalFlag:
            # determinig action using eps-greedy
            a = np.random.choice([np.random.choice(aEqual,p=[pEqual] * NUM_ACTIONS), np.argmax(q[s])], p=[eps,1-eps])
            # collect the experience
            s_, r, terminal, info = env.step(a)
            if exp == max_iterations:
                break
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
    return pi

def double_q_learning(env, gamma, max_iterations, logger):
    """
    Implement double Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # linear decay
    decay = eps/max_iterations
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################
    q = np.zeros((NUM_STATES, NUM_ACTIONS))
    qa = np.zeros((NUM_STATES, NUM_ACTIONS))
    qb = np.zeros((NUM_STATES, NUM_ACTIONS))
    exp = 0
    # collect eposides
    while exp < max_iterations:
        s = env.reset()
        terminalFlag = False
        # preparing actions and probabilities for exploration
        aEqual = list(range(0,NUM_ACTIONS))
        pEqual = 1/NUM_ACTIONS
        # collect experiences
        while not terminalFlag:
            # calculating (Qa(s,.) + Qb(s,.))/2
            qaval = [qa[s][i] for i in range(NUM_ACTIONS)]
            qbval = [qb[s][i] for i in range(NUM_ACTIONS)]
            qavg = [(qaval[i]+qbval[i])/2 for i in range(NUM_ACTIONS)]
            # collect acion sample
            a = np.random.choice([np.random.choice(aEqual,p=[pEqual] * NUM_ACTIONS), qavg.index(max(qavg))], p=[eps,1-eps])
            # collect one experience
            s_, r, terminal, info = env.step(a)
            if exp == max_iterations:
                break
            # choose randomly between UPDATE(A) and UPDATE(B)
            randUpdate = np.random.choice([1,2], p=[.5,.5])
            # if UPDATE(A)
            if randUpdate == 1:
                astar = np.argmax(qa[s_,:])
                if terminal == True:
                    terminalFlag = True
                    qa[s][a] = qa[s][a] + alpha*(r - qa[s][a])
                else:
                    qa[s][a] = qa[s][a] + alpha*(r + (gamma*qb[s_][astar]) - qa[s][a])
            # else UPDATE(B)
            else:
                bstar = np.argmax(qb[s_,:])
                if terminal == True:
                    terminalFlag = True
                    qb[s][a] = qb[s][a] + alpha*(r - qb[s][a])
                else:
                    qb[s][a] = qb[s][a] + alpha*(r + (gamma*qa[s_][bstar]) - qb[s][a])
            # linearly decaying eps
            if eps > 0.1:
                eps -= decay
            s = s_
            exp += 1
        # average q values
        q = (qa + qb)/2
        # determin police and value
        for st in range(NUM_STATES):
            pi[st] = np.argmax(q[st])
            v[st] = np.max(q[st])
        # Visualize the initial value and policy
        logger.log(exp, v, pi)
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "On-policy MC Control": on_policy_mc_control,
        "SARSA": sarsa,
        "Q-Learning": q_learning,
        "Double Q-Learning": double_q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        #"world2": lambda : [
        #    [10, "s", "s", "s", 1],
        #    [-10, -10, -10, -10, -10],
        #],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()