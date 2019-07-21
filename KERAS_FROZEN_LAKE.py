#!/usr/bin/env python
# coding: utf-8

# In[66]:


import gym
import numpy as np
import tensorflow as tf
import random
env = gym.make('FrozenLake-v0')
from tqdm import tqdm


# In[85]:


env = gym.make('FrozenLake-v0')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128,input_dim=16))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
lr = 0.85
yr = 0.45


# In[26]:


#model = tf.keras.models.load_model('keras_frozen_lake-highest.h5')


# In[110]:

print "Testing the neural  net without training"
r_all = []
for i in tqdm(range(1000)):
    s = env.reset()
    s = random.randint(0,15)
    reset = False
    while(reset != True):
        states = np.zeros((1,16))
        states[0][s] = 1
        action = np.argmax(model.predict(states))
        new_s ,reward ,reset,_ = env.step(action)
        r_all.append(reward)
        s = new_s
print('Success rate  = ',(sum(r_all)/1000)*100)
        
        


# In[87]:


rall = []
for i in tqdm(range(4000)):
    s = env.reset()
    e = 1-(i/2000)
    if(random.uniform(0,1) < e):
        s = random.randint(0,15)
    reset = False
    while(reset != True):
        states = np.zeros((1,16))
        states[0][s] = 1
        action = np.argmax(model.predict(states))
        new_s ,reward ,reset,_ = env.step(action)
        
        if(reset == True and reward != 1):
            reward = -1.5
        
        if(reward == 1):
            rall.append(reward)
        
        states_new = np.zeros((1,16))
        states_new[0][new_s] = 1
        target = lr*(reward + yr*(np.max(model.predict(states_new) - np.max(model.predict(states)))))
        target_val = np.zeros((1,4))
        target_val[0][action] = target
        model.fit(states,target_val,epochs=1,verbose=0)
        s = new_s
print(sum(rall))


# In[88]:

print "Testing the neural  net after training"
r_all = []
for i in tqdm(range(500)):
    s = env.reset()
    reset = False
    while(reset != True):
        states = np.zeros((1,16))
        states[0][s] = 1
        action = np.argmax(model.predict(states))
        new_s ,reward ,reset,_ = env.step(action)
        r_all.append(reward)
        s = new_s
print 'Success rate  = ',(sum(r_all)/500)*100
        


# In[48]:


model.save('keras_frozen_lake.h5')

