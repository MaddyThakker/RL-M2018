#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt


# ### Policy Iteration

# In[2]:


chars = np.array(['^', '<', 'd', '>'])

def get_next_state(state, action):
    reward = -1
    if(state[0] == 0 and state[1] == 0):
        return 0, state
    if(state[0] == 3 and state[1] == 3):
        return 0, state
    if(action == 0):
        next_state = [state[0] - 1, state[1]]
    if(action == 1):
        next_state = [state[0], state[1]-1]
    if(action == 2):
        next_state = [state[0] + 1, state[1]]
    if(action == 3):
        next_state = [state[0], state[1]+1]
    if(next_state[0] < 0 or next_state[0] >= 4 or next_state[1] < 0 or next_state[1] >= 4):
        next_state = state
    return reward, next_state


# In[29]:


theta = 1e-10
gamma = 1 # undiscounted task
value = np.zeros([4, 4])
state = [[i, j] for i in range(4) for j in range(4)]
all_actions = [ [ [0, 1, 2, 3] for i in range(4) ] for j in range(4) ]
policies = np.zeros(value.shape)
while True:
    print_iter = 5

    while True:
        delta = 0
        
        new_value = np.zeros([4, 4])
        
        
        for s in state:            
            v_iter = []
            for action in all_actions[s[0]][s[1]]:
                reward, next_state = get_next_state(s, action)
                new_value[s[0]][s[1]] += (1/len(all_actions[s[0]][s[1]]))*(reward + gamma*value[next_state[0]][next_state[1]])
        
        delta = np.sum(abs(value - new_value))
        
        value = new_value
        if(print_iter > 0):
            print_iter-=1
            print("Policy Evaluation at Work")
            print(value)
        if(delta < theta):
            print("breaking", delta, theta)
            break
                
    print(value)      
    for action_lis in all_actions:
        for specific_action in action_lis:
            print(chars[specific_action], end = ' ')
        print()
        
        
    policy_stable = True
    
    for s in state:
        
        old_action = policies[s[0]][s[1]]
        v_iter = []
        
        for action in [0, 1, 2, 3]:
            reward, next_state = get_next_state(s, action)
            v_iter.append((reward + gamma*value[next_state[0]][next_state[1]]))
        
        v = max(v_iter)
        
        v_iter = np.array(v_iter)
        best_action = np.argmax(v_iter)

        policies[s[0]][s[1]]  = best_action
        all_actions[s[0]][s[1]] = [best_action]
        
        if policy_stable and best_action != old_action:
            policy_stable = False
    
    if(policy_stable):
        print("policy stable, breaking")
        break


# In[30]:


for x in policies:
    for y in x:
        print(chars[int(y)], end = '  ')
    print()


# #### fix to the bug mentioned in Exercise 4.4
# 
# #### Q -The algorithm may never terminate if the policy continually switches between two or more policies that are equally good
# 
# #### Ans - Suppose that there are 2 policies that are equally good, the difference between the 2 policies is just for a single state, 2 different actions can be selected (both being equally good). The algorithm takes the argmax which inturn can select any of the equally likely actions for any state. If they form a cycle such that different actions are selected in every iteration; the algorithm may never terminate. 
# #### The fix is just to take the maximum action for each state such that it occurs at the smallest index in the list of actions to prevent different actions being selected. np.argmax() does exaclty this.

# ### Value Iteration

# In[55]:


theta = 1e-10
gamma = 1 # undiscounted task
value = np.zeros([4, 4])
all_actions = [ [ [0, 1, 2, 3] for i in range(4) ] for j in range(4) ]
while True:
    print("New Iteration - ")
    print(value)
    delta = 0
    for s in state:
        v = value[s[0]][s[1]]
        v_iter = []
        for action in all_actions[s[0]][s[1]]:
            reward, next_state = get_next_state(s, action)
            v_iter.append(reward + gamma*value[next_state[0]][next_state[1]])
        value[s[0]][s[1]] = np.max(v_iter)
        delta = max(delta, abs(v - value[s[0]][s[1]]))
    if(delta < theta):
        print("breaking", delta, theta)
        break
                


# In[53]:


policy = np.chararray([4, 4],unicode=True)
for s in state:
    v_iter = []
    for action in all_actions[s[0]][s[1]]:
        reward, next_state = get_next_state(s, action)
        v_iter.append(reward + gamma*value[next_state[0]][next_state[1]])
    policy[s[0]][s[1]] = chars[(np.argmax(v_iter))]


# In[54]:


print(policy)


# #### As we can see, with every iteration Value Function is improving; also optimal policy is printed
