import pickle
import numpy as np
import random
import cv2

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()

def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f) 
    f.close()
    return r

path = 'map2.png'
im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
im_gray[im_gray==0] = 1 # Inavaliable places
im_gray[im_gray!=1] = 0 # Avaliable places

im_size = np.array(np.shape(im_gray))
pp = 500/75  # The length of each pixel
pad_pix = 30

def cor_to_ind(cor):
    # Transform Coordinate to the Index in the image 
    new_im = np.zeros(np.shape(cor))
    if len(np.shape(cor))==2:
        new_im[:,1] = cor[:,0]/pp
        new_im[:,0] = im_size[0] - cor[:,1]/pp - 1
    elif len(np.shape(cor))==3:
        new_im[:,:,1] = cor[:,:,0]/pp
        new_im[:,:,0] = im_size[0] - cor[:,:,1]/pp -1
    elif len(np.shape(cor))==1:
        new_im[1] = cor[0]/pp
        new_im[0] = im_size[0] - cor[1]/pp - 1
    return new_im + pad_pix

def ind_to_cor(ind):
    # Transform the Index in the image to Coordinate
    ind = ind - pad_pix
    new_cor = np.zeros(np.shape(ind))
    if len(np.shape(ind))==2:
        new_cor[:,0] = ind[:,1]*pp
        new_cor[:,1] = (im_size[0] - ind[:,0])*pp
    elif len(np.shape(ind))==3:
        new_cor[:,:,0] = ind[:,:,1]*pp
        new_cor[:,:,1] = (im_size[0] - ind[:,:,0])*pp
    elif len(np.shape(ind))==1:
        new_cor[0] = ind[1]*pp
        new_cor[1] = (im_size[0] - ind[0])*pp
    return new_cor

def cor_to_nor(cor): 
    # Normalization of the Coordinate
    nor = np.zeros(np.shape(cor))
    if len(np.shape(cor))==1:
        nor[0] = np.interp(cor[0], [0,im_size[1]*pp], [-1,1])
        nor[1] = np.interp(cor[1], [0,im_size[0]*pp], [-1,1])
    elif len(np.shape(cor))==2:
        nor[:,0] = np.interp(cor[:,0], [0,im_size[1]*pp], [-1,1])
        nor[:,1] = np.interp(cor[:,1], [0,im_size[0]*pp], [-1,1])
    return nor

def nor_to_cor(nor):
    # Inverse normalization
    cor = np.zeros(np.shape(nor))
    if len(np.shape(nor))==1:
        cor[0] = np.interp(nor[0],[-1,1], [0,im_size[1]*pp])
        cor[1] = np.interp(nor[1],[-1,1], [0,im_size[0]*pp])
    elif len(np.shape(nor))==2:
        cor[:,0] = np.interp(nor[:,0],[-1,1], [0,im_size[1]*pp])
        cor[:,1] = np.interp(nor[:,1],[-1,1], [0,im_size[0]*pp])
    return cor

def lat_to_cor(lis):
    ori = [-8.6837, 41.3777]
    out = np.zeros([2,])
    out[0] = (lis[0] - ori[0])*(6546.67/(-8.6044 + 8.6837))
    out[1] = (lis[1] - ori[1])*(5080/(41.4239 - 41.3777))
    return out

def image_downsampling(image, factor):
    # 获取原始图像的尺寸
    height, width = image.shape[:2]
    # 计算降采样后的尺寸
    new_height = height // factor
    new_width = width // factor
    # 使用平均池化进行降采样
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return downsampled_image

def a_star_search(grid: list, begin_point: list, target_point: list, cost=1):
    shape = np.shape(grid)
    if grid[begin_point[0]][begin_point[1]] != 0: # If the start point is inavaliable, find the nearest avaliable point 
        for i in range(1, 100):
            if begin_point[1]+i < shape[1]:
                if grid[begin_point[0]][begin_point[1]+i] == 0: begin_point[1] += i ; break
            if begin_point[0]-i > 0:
                if grid[begin_point[0]-i][begin_point[1]] == 0: begin_point[0] -= i ; break
            if begin_point[0]+i < shape[0]:
                if grid[begin_point[0]+i][begin_point[1]] == 0: begin_point[0] += i ; break
            if begin_point[1]-i > 0:
                if grid[begin_point[0]][begin_point[1]-i] == 0: begin_point[1] -= i ; break
            
    if grid[target_point[0]][target_point[1]] != 0: # If the target point is inavaliable, find the nearest avaliable point 
        for i in range(1, 100):
            if target_point[1]+i < shape[1]:
                if grid[target_point[0]][target_point[1]+i] == 0: target_point[1] += i ; break
            if target_point[0]-i > 0:
                if grid[target_point[0]-i][target_point[1]] == 0: target_point[0] -= i ; break
            if target_point[0]+i < shape[0]:
                if grid[target_point[0]+i][target_point[1]] == 0: target_point[0] += i ; break
            if target_point[1]-i > 0:
                if grid[target_point[0]][target_point[1]-i] == 0: target_point[1] -= i ; break
            
    # Creat the H matrix
    heuristic = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            heuristic[i][j] = abs(i - target_point[0]) + abs(j - target_point[1])
            if grid[i][j] == 1:
                heuristic[i][j] = 999999  # added extra penalty in the heuristic matrix

    # the actions we can take
    delta = [[-1, 0],[0, -1],[1, 0],[0, 1],[-1, -1],[1, -1],[1, 1],[-1, 1]]  
    # delta = [[-1, 0],[0, -1],[1, 0],[0, 1]]  
    close_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the referrence grid
    close_matrix[begin_point[0]][begin_point[1]] = 1
    action_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the action grid
    x = begin_point[0]
    y = begin_point[1]
    g = 0
    f = g + heuristic[begin_point[0]][begin_point[1]]
    cell = [[f, g, x, y]]
    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand
    while not found and not resign:
        if len(cell) == 0:
            resign = True        
            return None, None
        else:
            cell.sort()  # to choose the least costliest action so as to move closer to the goal
            cell.reverse()
            next = cell.pop()
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]
            if x == target_point[0] and y == target_point[1]:
                found = True
            else:
                # delta have four steps
                for i in range(len(delta)):  # to try out different valid actions
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if i > 3:
                        cost = 1.41
                    else:
                        cost = 1
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        if close_matrix[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            f2 = g2 + heuristic[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            close_matrix[x2][y2] = 1
                            action_matrix[x2][y2] = i
    invpath = []
    x = target_point[0]
    y = target_point[1]
    invpath.append([x, y])  # we get the reverse path from here
    while x != begin_point[0] or y != begin_point[1]:
        x2 = x - delta[action_matrix[x][y]][0]
        y2 = y - delta[action_matrix[x][y]][1]
        x = x2
        y = y2
        invpath.append([x, y])

    path = []
    for i in range(len(invpath)):
        path.append(invpath[len(invpath) - 1 - i])
    return path

def find_nearest_zero(arr, target):
    # The coordinate transformation may result out side of road. This is to find the nearest avaliable place of the result.
    rows, cols = len(arr), len(arr[0])
    visited = [[False] * cols for _ in range(rows)]
    queue = [(target[0], target[1], 0)]
    while queue:
        row, col, dist = queue.pop(0)
        if arr[row][col] == 0:
            return (row, col)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c, dist + 1))
    return None