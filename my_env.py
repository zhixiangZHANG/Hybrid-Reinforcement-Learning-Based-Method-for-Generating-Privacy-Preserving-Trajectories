import gym
from gym.spaces import Box, Discrete, MultiBinary
from my_function import save_variable, load_variable, cor_to_ind, ind_to_cor, cor_to_nor, nor_to_cor, lat_to_cor, a_star_search
from my_ppo import PPO, Memory, ActorCritic, Critic, Actor
import numpy as np
import cv2

class MyEnv(gym.Env):
    def __init__(self, train = True):
        path = 'map2.png'
        im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        im_gray[im_gray==0] = 1 #  # Inavaliable places
        im_gray[im_gray!=1] = 0 #  # Avaliable places
        self.im_size = np.array(np.shape(im_gray))
        self.pp = 500/75
        self.pad_pix = 30
        self.vis_pix = 20
        pad_row = np.zeros([self.pad_pix,self.im_size[1]])+1
        pad_col = np.zeros([self.im_size[0]+2*self.pad_pix, self.pad_pix])+1
        im_padding = np.vstack([pad_row, im_gray, pad_row])
        self.im_padding = np.hstack([pad_col, im_padding, pad_col])
        val = np.array(np.where(self.im_padding==0))
        self.vis_size = self.pad_pix
        self.v_map = np.zeros(np.shape(self.im_padding))-1  # In v_map -1 is inavaliable
        self.ma = np.zeros(np.shape(self.im_padding)) + 1 # 1 is reachable
        self.target_nor = np.zeros((2,))
        self.start_nor = np.zeros((2,))
        self.action_space = Box(low=np.array([-5.0, -5.0]), high=np.array([5.0, 5.0]))
        self.max_speed = 18
        self.rank = 0
        self.dis = 0
    
        if train:
            self.df = load_variable('train_df')
        else:
            self.df = load_variable('test_df')

    def around(self, ind, r = 50): # Search radius 50m, Return 8 distances
        dd = (ind - ind.astype(int))*self.pp
        cha = np.zeros((8,)) # initial 8 directions
        cha[0] = self.pp - dd [1]; cha[4] = dd[1]    
        if sum(dd) > self.pp: cha[1] = 1.41*(self.pp-dd[1]); cha[5] = 1.41*(self.pp-dd[0])
        else: cha[1] = 1.41*dd[0]; cha[5] = 1.41*dd[1]
        cha[2] = dd[0]; cha[6] = self.pp - dd[0]
        if dd[1] > dd[0]: cha[3] = 1.41*dd[0]; cha[7] = 1.41*(self.pp - dd[1])
        else: cha[3] = 1.41*dd[1]; cha[7] = 1.41*(self.pp - dd[0])
        zzx = ind.astype(int)    
        a = np.zeros((8,))
        if self.v_map[zzx[0]][zzx[1]] == -1 :
            a = np.zeros((8,))
        else:
            a = cha + 0
            for i in range (1, int(r/self.pp)):
                if  self.v_map[zzx[0]][zzx[1]+i] != -1:  a[0] = a[0] +  self.pp
                else: break
            for i in range (0, int(r/self.pp/1.41)):
                if  self.v_map[zzx[0]-i][zzx[1]+i] != -1:  a[1] = a[1] + 1.41 *  self.pp
                else: break
            for i in range (0, int(r/self.pp)):
                if  self.v_map[zzx[0]-i][zzx[1]] != -1:  a[2] = a[2] +   self.pp
                else: break
            for i in range (0, int(r/self.pp/1.41)):
                if  self.v_map[zzx[0]-i][zzx[1]-i] != -1:  a[3] = a[3] +  1.41 * self.pp
                else: break
            for i in range (0, int(r/self.pp)):
                if  self.v_map[zzx[0]][zzx[1]-i] != -1:  a[4] = a[4] +   self.pp
                else: break
            for i in range (0, int(r/self.pp/1.41)):
                if  self.v_map[zzx[0]+i][zzx[1]-i] != -1:  a[5] = a[5] +  1.41 * self.pp
                else: break
            for i in range (0, int(r/self.pp)):
                if  self.v_map[zzx[0]+i][zzx[1]] != -1:  a[6] = a[6] +   self.pp
                else: break
            for i in range (0, int(r/self.pp/1.41)):
                if  self.v_map[zzx[0]+i][zzx[1]+i] != -1:  a[7] = a[7] +  1.41 * self.pp
                else: break
        a = a/r
        return a
    
    def step(self, a): # Update state and calculate the current rewawrds
        reward = 0 
        ob1 = self.state_1  # lx, ly, vx, vy, ax, ay, dx, dy, gx, gy ar, es1, es2
        ob2 = self.state_2 # Figure about the environment
        collision = False
        done = False
        ############################     Record States     ########################
        last_loc = ob1[0:2] + 0  # Location
        last_v = ob1[2:4] + 0    # Speed
        last_a = ob1[4:6] + 0    # Acceleration
        last_dis = ob1[6:8] + 0  
        last_d = np.linalg.norm(last_dis*[self.pp*self.im_size[1]/2, self.pp*self.im_size[0]/2], ord = 2) # Distance to end point
        last_gg = ob1[0:2] - self.start_nor
        last_g = np.linalg.norm(last_gg*[self.pp*self.im_size[1]/2, self.pp*self.im_size[0]/2], ord = 2)  # Distance to start point
        
        ############################     Update States     ########################
        ### Acceleration
        ob1[4:6] = a   # Update Acceleration
        ### Speed
        v_sup = ob1[2:4] + ob1[4:6]  # Speed in theory
        speed = np.linalg.norm(v_sup*5, ord = 2) # Calculate Speed 
        if speed < 30:  # Vehicle speed limit, If not exceed, Updata Speed
            ob1[2:4] = v_sup
        speed = np.linalg.norm(ob1[2:4]*5, ord = 2) # Calculate current Speed 
        ### Location
        ind_sup = np.squeeze(cor_to_ind(nor_to_cor(ob1[0:2]) + ob1[2:4]*5))
        ind_sup = ind_sup.astype(int)
        if ind_sup[0] - self.pad_pix >= self.im_size[0] or ind_sup[0] - self.pad_pix < 0: # Whether it is outside the map
            collision = True
        if ind_sup[1] - self.pad_pix>= self.im_size[1] or ind_sup[1] - self.pad_pix < 0:  # Whether it is outside the map
            collision = True
        if collision == False:
            if self.v_map[ind_sup[0]][ind_sup[1]] == -1: # Whether it is on the road
                collision == True
            else:
                ob1[0:2] = cor_to_nor(nor_to_cor(ob1[0:2]) + ob1[2:4]*5) # If it is on the raod, update the location
        ### Distance 
        ob1[6:8] = self.target_nor - ob1[0:2]  # Update distance to start point
        ob1[8:10] = ob1[0:2] - self.start_nor  # Update distance to end point
        ### The new states
        now_loc = ob1[0:2]
        now_v = ob1[2:4]
        now_a = ob1[4:6]
        now_d = np.linalg.norm(ob1[6:8]*[self.pp*self.im_size[1]/2, self.pp*self.im_size[0]/2], ord = 2)
        now_g = np.linalg.norm(ob1[8:10]*[self.pp*self.im_size[1]/2, self.pp*self.im_size[0]/2], ord = 2)
        ind_now = np.squeeze(cor_to_ind(nor_to_cor(ob1[0:2])))
        ind_now_int = ind_now.astype(int)
        ### Distances to road edges in eight direction
        ob2 = self.v_map[ind_now_int[0]-self.vis_pix:ind_now_int[0]+self.vis_pix, ind_now_int[1]- self.vis_pix :ind_now_int[1]+self.vis_pix] # 更新ob2
        att1 = self.around(ind_now) ## The current distance to road edge in eigth directions
        ind_next = np.squeeze(cor_to_ind(nor_to_cor(ob1[0:2]) + ob1[2:4]*5))
        att2 = self.around(ind_next) ## The estimate distance to road edge in eigth directions for one second later
        ind_nnext = np.squeeze(cor_to_ind(nor_to_cor(ob1[0:2]) + ob1[2:4]*10))
        att3 = self.around(ind_nnext) ## The estimate distance to road edge in eigth directions for two seconds later
        ob1[10:18] = att1  #
        ob1[18:26] = att2  #
        ob1[26:] = att3

        ob2 = ob2[np.newaxis,:]
        ob2 = ob2[np.newaxis,:]
        self.state_1, self.state_2 = ob1, ob2 # State 1 and State 2 in paper
        ############################     Setting Rewards     ########################
        ### when collision
        if collision == True:   
            reward -= 15 + speed**2
        ### when arrive the end point
        if now_d < 20: 
            done = True
            reward += 40
        valid_speed = np.clip(speed, 0,self.max_speed)
        ### whether in the center of road
        att1 = att1 + 0.0001  
        d0 = att1[0]/att1[4]
        d2 = att1[2]/att1[6]
        if d0 > 0.25 and d0 < 4 and d2 > 0.25 and d2 < 4:
            reward += valid_speed/self.max_speed           # 0.5-1
        else:
            reward -= 0
        ### speed too slow
        if speed < 3:
            reward -= 1 # -1
        ###  speed too fast
        if speed > self.max_speed:
            reward -= (speed-self.max_speed)/self.max_speed # 
        ### whether in right direction
        r1 = last_d - now_d
        r2 = now_g - last_g
        if r1 > 0 or r2 > 0:
            reward += valid_speed/self.max_speed
        else: 
            reward -= valid_speed/self.max_speed + 1
        return self.state_1, self.state_2, reward, done
        ### time penalty
        reward -= 0.5

    def reset(self, rand = -1): # Sample and prepare environment for training
        if rand == -1:
            self.v_map = np.zeros(np.shape(self.im_padding))-1
            a_star_path = np.array(self.df['Road_ind'][self.rank])
            start_ind = a_star_path[0]
            end_ind = a_star_path[-1]
            self.rank += 1
        else:
            self.v_map = np.zeros(np.shape(self.im_padding))-1
            a_star_path = np.array(self.df['Road_ind'][rand])
            start_ind = a_star_path[0]
            end_ind = a_star_path[-1]
        
        self.start_nor = cor_to_nor(ind_to_cor(start_ind))# 归一化坐标
        self.target_nor = cor_to_nor(ind_to_cor(end_ind))
        state_1 = np.append(self.start_nor, [0,0,0,0]) # 初始速度为 0, 加速度为 0 
        state_1 = np.append(state_1, self.target_nor-self.start_nor) # 归一化距离
        state_1 = np.append(state_1, [0,0])
        shot_size = 2
    
        for i in range(0,len(a_star_path)):
            tem_ind = a_star_path[i,:]
            tem_mask = self.ma[tem_ind[0]-shot_size:tem_ind[0]+shot_size+1, tem_ind[1]-shot_size:tem_ind[1]+shot_size+1]
            self.v_map[tem_ind[0]-shot_size:tem_ind[0]+shot_size+1, tem_ind[1]-shot_size:tem_ind[1]+shot_size+1] = tem_mask

        state_2 = self.v_map[int(start_ind[0])-self.vis_pix: int(start_ind[0])+self.vis_pix, int(start_ind[1])-self.vis_pix: int(start_ind[1])+self.vis_pix]
        att1 = self.around(start_ind)
        state_1 = np.append(state_1, att1)
        state_1 = np.append(state_1, att1)
        state_1 = np.append(state_1, att1)
        self.state_1 = state_1
        self.dis = np.linalg.norm(state_1[6:8]*[self.pp*self.im_size[1]/2, self.pp*self.im_size[0]/2], ord = 2)
        self.state_2 = state_2[np.newaxis,:]

        return self.state_1, self.state_2