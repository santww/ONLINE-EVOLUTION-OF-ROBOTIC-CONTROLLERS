from activations import *

# List of seeds for each run
SEEDS = [0, 1, 2, 7, 13, 21, 42, 69, 100, 123, 256, 314, 512, 999, 1234, 2020, 2022, 2025, 2718, 31415, 16180, 12345, 54321, 99999, 314159265, 271828182, 161803398, 98765, 4242, 606, 808, 101010]

# Evolution parameters #
num_gens = 150 # 50 100 150 200 300
num_gens_old = 150 # Store the old number of generations
config = "neat_config.txt"
evolution = "online" # offline
initial_population = "random" # random or best_genomes
num_initial_best_genomes = 2 # 1 or 2
plots = False # True or False
experiment_name = None # Name of the experiment for saving results
run = None # Run number
fitness = "average" # normal or average or normal_best
fail_test = False # True or False
#

# Pybullet parameters #
terrain = "plane.urdf" # rough or plane.urdf 
terrainId = None
robot = "robots\\quadruped.urdf" # ant quadruped
robotId = None
gravity = -9.81
start_pos = None # [0, 0, 0.2] 256x256 [0, 0, 2.0] 512x512
start_orientation = None
GUI = False # True or False
#

# Dynamic Environment parameters #
dynamic_environment = False # True or False
terrain_file = "environments\\terrain_data.csv"
height_data = None
heightfield_data = None
terrain_shape = None
mesh_scale = [20, 20, 4] # [20, 20, 4] # [100, 100, 10]  
terrain_size = 512 # 1024x1024 512x512 256x256
terrain_scaling = 128 # 256
scale_hills = 0.05
scale_holes = 0.1
friction_coefficient = 1

# FIXED VALUES #
motor_max_force = 50 # 6.5 ant - 50. 25.
motor_joint_range = 1. # Can be adjusted if needed
simulation_fps = 60
delta_t = 1/simulation_fps # 30 Hz or 4.166 ms # 60 Hz or 1.666 ms
alpha = 0.3 # exponential moving average
#

# Duration of the simulation #
simulation_duration = 20 # 20s 
recovery_duration = 20 # 20s
racing_min_duration = 13 # 13s
#

# Number of Iterations (steps) #
simulation_length_default = int(simulation_duration / delta_t) # 1200 steps
simulation_length = simulation_length_default                                     
recovery_length = simulation_length # 1200 steps
racing_min_length = int(racing_min_duration / delta_t) # 780 steps
#

# NEAT's Improvement mechanisms #
train = True
racing = False
recoveryMechanism = False 
parameter_control = False
distance_track = False # Fitness function based on distance traveled (True) or average speed (False)
#

# Racing Coefficients #
# Hoeffding Inequality
hoeffding_alpha = 0.5 # 0.5 0.9
hoeffding_beta = 2 # 2 5
#

# H-Rule Parameter Control Coefficients #
hrule_target_success = 0.2
hrule_alpha = 0.1
hrule_alpha_max = 0.15                                          
hrule_alpha_min = 0.05
min_duration = 13 # 13s
max_duration = 30 # 30s
min_eval_time = int(min_duration / delta_t) # 780 steps                                            
max_eval_time = int(max_duration / delta_t) # 1800 steps
#

def apply_condition(k, v):
    globals()[k] = v