# Hybrid-Reinforcement-Learning-Based-Method-for-Generating-Privacy-Preserving-Trajectories

## Program Hardware Requirements
- **GPU:** RTX 3080 (10GB)
- **CPU:** 12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- **Python:** 3.8
- **Cuda:** V11.3.109

## Required Packages
- pandas
- gym
- opencv-python
- torch

## File Overview
### I. `my_function.py`
Contains custom functions:
1. Variable save and load functions: `save_variable`, `load_variable`
2. Coordinate transformation functions: `cor_to_ind`, `ind_to_cor`, `cor_to_nor`, `nor_to_cor`, `lat_to_cor`
3. Image downsampling function: `image_downsampling`
4. A* road extraction function

### II. `my_ppo.py`
Structure of the hybrid reinforcement learning based model
Classes in the file:
1. **Memory:**
   - Variables: actions, state_1, state_2, logprobs, rewards, is_terminals
2. **Actor:**
   - Structure of the actor
3. **Critic:**
   - Structure of the critic
4. **ActorCritic:**
   - Defines act and evaluate behaviors
5. **PPO:**
   - Contains select_action and update behaviors

### III. `my_env.py`
Mainly includes environment update, interaction with the agent, and reward definition.

### IV. Data Files
- **map2.png:** Test map
- **test_df:** 500 randomly generated scenes from the map
- **train_df:** 80,000 randomly generated scenes from the map

### V. The .ipynb files
- **train_agent.ipnyb:** Trainning Setting and Process
- **test_agent.ipnyb:** Test the trained Agent
