import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import random
import matplotlib.image as mpimg
import glob

from PIL import Image

import cv2
# Let's do it without our previous price so that we always immidiately get a return for our action 

class CandleStickEnv(Env):
    def __init__(self,):
        # Actions we can take: Bullish treshold, Bearish treshold 
        self.action_space = Discrete(2)
        # Here the observation space will be created

        # This is the action that is currently held
        self.state = 0
        self.day= 1
        
        self.current_stock = 'ASIANPAINT'

        self.profit = 0
        self.purchase_price = []
        self.action_taken_before = 0
        self.numWins = 0
        self.numLosses = 0
        


    def step(self, action):

        # firsly we get the observation



        # 0 is hold 1 is buy and 2 is sell
        # Now we will do an if statement 
        # We will return info as well
        # if action == 0:

        path_str = f"images/ASIANPAINT/{self.day},*"
        img_path = glob.glob(path_str)
        img = mpimg.imread(img_path[0])

        

        array_splt = img_path[0].split(",")

        current_price = float(array_splt[1])  

        arr_name_of_new_stock = array_splt[2].split(".")
        name_of_new_stock = arr_name_of_new_stock[0]


        done = False

        # self.action_taken_before
        #action
        # There is 2 actions in the actions space

        treshold_bull = self.purchase_price + 50
        treshold_bear = self.purchase_price - 50

        if(self.day == 1 or self.current_stock != (name_of_new_stock)):
            # return  profit and everythng as zero except of the observation
            reward = 0
            info  = f"The EQUITY is changing to: {self.current_stock}, from {name_of_new_stock}"
            self.current_stock = name_of_new_stock
            self.purchase_price = []
            self.action_taken_before = 0
        else:
            if ( current_price > treshold_bull ):
                if ( self.action_taken_before == 0):  # 0 means bear,  1 means bull
                    reward = -100
                    self.numLosses = self.numLosses +1
                    info  = f"X Unsuccesful prediction at value: {current_price}."
                elif (self.action_taken_before == 1):
                    reward = 100
                    self.numWins = self.numWins +1
                    info  = f"✓ Succesful prediction at value: {current_price}."
            elif (current_price < treshold_bear):
                if ( self.action_taken_before == 0): 
                    reward = 100
                    self.numWins = self.numWins +1
                    info  = f"✓ Succesful prediction at value: {current_price}."
                elif (self.action_taken_before == 1):
                    reward = -100
                    self.numLosses = self.numLosses +1
                    info  = f"X Unsuccesful prediction at value: {current_price}."
            else: 
                reward = 0
                info  = f"Subtle change in price: {current_price}"




        
        self.profit = self.profit + reward
        info = f"{info} ----- The W/L ratio is: {(self.numWins/(self.numLosses+self.numWins))*100}%"
    
        self.day = self.day +1
        # self.purchase_price = current_price
        reward = reward

        



        def load_image(file):
            img = Image.open(file)
            img.load()
            data = np.asarray(img, dtype="int32")
            return data

        data_3d = load_image(img_path[0])
        data_1d = data_3d.reshape(-1)
        
        return data_1d,reward,done, info


    def reset(self):
        
        # img = mpimg.imread('images/TrainingDS/1,3216.3,ASIANPAINT.png')
        path_str = f"images/ASIANPAINT/{self.day},*"
        img_path = glob.glob(path_str)
        img = mpimg.imread(img_path[0])
        # now we have the image and the path

        array_splt = img_path[0].split(",")
        self.current_price = float(array_splt[1])

        self.current_stock = (array_splt[2].split("."))[0]
        
        # self.previous_price = self.current_price
        self.day = 1

        # return self.current_price
        self.profit = 0


        def load_image(file):
            img = Image.open(file)
            img.load()
            data = np.asarray(img, dtype="int32")
            return data

        data_3d = load_image(img_path[0])
        data_1d = data_3d.reshape(-1)

        return data_1d



