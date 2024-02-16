import keras
keras.__version__

import tensorflow as tf
from keras.layers import Dense, Activation, Conv2D, Input
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, TimeDistributed , Conv2D, MaxPooling2D
from tensorflow.keras import layers
import numpy as np
import sys



class ReplayBuffer(object):
    def __init__(self, max_size,input_shape, n_actions, discrete= False ):
        self.mem_size = max_size
        self.mem_cntr = 0 #mem counter 
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape)) 
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, ):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal, 

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):

    
    model = Sequential([
            layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(200,300,4)),
            # THIS IS SUPER DUMB THAT I USED POOLING
            # TODO DO NOT USE POOLING
            layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),

            layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu'),
            #Dense
            layers.Flatten(),

            layers.Dense(64, activation='relu'),
            layers.Dense(n_actions, activation='softmax')
    ])
        # Input(shape=(200,300,4)),
        # Dense(fc1_dims, ),
        # Activation('relu'),
        # Dense(fc2_dims),
        # Activation('relu'),
        # Dense(n_actions)

        # Conv2D(32, (3, 3), activation='relu', input_shape=(200,300,4)),
        # MaxPooling2D((2, 2)),
        # Conv2D(64, (3, 3), activation='relu'),
        # MaxPooling2D((2, 2)),
        # Conv2D(64, (3, 3), activation='relu'),

    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model



class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000000, fname='ddqn_model.h5', replace_target=100):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.model_file=fname
        self.gamma = gamma
        self.epsilon = epsilon 
        self.epsilon_dec = epsilon_dec
        self.epsilon_min  = epsilon_end
        self.batch_size = batch_size
        self.model = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done, ):
        self.memory.store_transition(state, action, reward, new_state, done, ) 

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state_3d = state.reshape(1, 200, 300, 4)
            actions = self.q_eval.predict(state_3d)
            print(actions)
            action = np.argmax(actions)


        return action

    def learn(self):
        

        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indeces = np.dot(action,action_values )

            fix = new_state.reshape(64,200, 300, 4)
            q_next = self.q_target.predict(fix)
            q_eval = self.q_eval.predict(fix)



            fix2 = state.reshape(64,200, 300, 4)
            q_pred = self.q_eval.predict(fix2)
            


            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indeces] = reward + self.gamma * q_next[batch_index, max_actions.astype(int)] *done


            _ = self.q_eval.fit(fix2, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()
    
    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()