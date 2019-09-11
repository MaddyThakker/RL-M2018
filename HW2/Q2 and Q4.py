#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pickle


# In[36]:


def get_reward(state, action):
    # given the state and action, returns the next_state and reward obtained.
    if(state[0] == 0 and state[1] == 1):
        next_state = [4, 1]
        return 10, next_state
    elif(state[0] == 0 and state[1] == 3): 
        next_state = [2, 3]
        return 5, next_state
    elif(action == 0):
        next_state = [state[0] - 1, state[1]]
    elif(action == 1):
        next_state = [state[0], state[1]-1]
    elif(action == 2):
        next_state = [state[0] + 1, state[1]]
    elif(action == 3):
        next_state = [state[0], state[1]+1]
    reward = 0
    if(next_state[0] < 0 or next_state[0] >= 5 or next_state[1] < 0 or next_state[1] >= 5):
        reward = -1
        next_state = state
    return reward, next_state


# In[57]:


def ind(s):
    # a hash function to store the 2D states as 1D in map.
    return s[0]*100+s[1]
gamma = 0.9
idx = 0
state = [[i, j] for i in range(5) for j in range(5)]

ind_s = {} # dictionary that maps states to indices. 
for s in state:
    ind_s[ind(s)] = idx
    idx+=1


# ### Figure 3.5

# In[58]:


#Reference - http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf
prob = np.zeros([25, 25])
reward = np.zeros([25])
for s in state:
    for action in range(4):
        r, s_dash = get_reward(s, action) # for each state and each action, get next state and corresponding reward
        src = ind_s[ind(s)] 
        reward[src] += 0.25*r # added in reward of source
        des = ind_s[ind(s_dash)]
        prob[src][des] += 0.25 # probabilities of going from state a to state b


# In[59]:


# v(s) = E[R_{t+1} + gamma*v(s_dash) | s_t = s]
# v(s) = R_s + gamma*sum(prob[s][s_dash]*v(s_dash))
# v = r + gammma*p*v
# (I - gamma*p)*v = r
# v = inv((I - gamma*p))*r
values = np.matmul(np.linalg.inv(np.identity(25) - gamma*(prob)), reward)


# In[60]:


values.round(1).reshape(5,5)


# ### Figure 3.5

# In[71]:


action_star = [ [ [] for i in range(5) ] for j in range(5) ] # optimal actions
value_star = np.zeros([5, 5]) # optimal_values
theta = 1e-20 # convergence after theta


# In[72]:


chars = np.array(['^', '<', 'd', '>'])
while True:
    delta = 0
    for s in state:
        v_iter = []
        all_actions = [0, 1, 2, 3]
        for action in all_actions:
            reward, next_state = get_reward(s, action)
            v_iter.append((reward + gamma*value_star[next_state[0]][next_state[1]])) 
        v = max(v_iter)
        v_iter = np.array(v_iter) #has reward for each action for this state
        actions = chars[np.argwhere(v_iter == np.amax(v_iter)).ravel()] # choosing the best action
        delta = max(delta, np.abs(v - value_star[s[0]][s[1]]))
        value_star[s[0]][s[1]] = v
        action_star[s[0]][s[1]] = actions
        
    if(delta < theta):
        print("breaking", delta, theta)
        break
        


# In[73]:


print(np.around(value_star, decimals=1))


# In[84]:


for x in action_star:
    for y in x:
        print(y, end= ',')
    print()


# In[ ]:




