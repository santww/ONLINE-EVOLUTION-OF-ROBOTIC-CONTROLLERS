import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import numpy as np
import pybullet as p
import pybullet_data
import pickle
import neat
import time
import pydot
from noise import pnoise2 
import os
import random
import json
import collections

import constants as c

import warnings
warnings.filterwarnings("ignore")

def set_simulation_length(run, results_filename):
    # Get the last simulation length from the results file #

    try:
        # Get the simulation_length from the results files
        
        # Load results file
        with open(results_filename, "r") as f:
            results = json.load(f)
        
        for condition in results:
            if "simulation_length" in condition:
                sim_length_list = condition["simulation_length"][run]

                # Get the last simulation length from the current run
                last_sim_length = sim_length_list[-1]

                # Set the length constants
                c.simulation_duration = last_sim_length
                c.simulation_length = int(c.simulation_duration / c.delta_t)
                return

    except Exception as e:
        print(f"[ERROR] Failed to set simulation length for run {run}: {e}")
    
def set_results_file(filename, save=True):
    # Returns the file name that will contain the results #
    aux = filename.split(os.sep) 
    name_file = aux[-1].split('.')[0] # Get the name of the file without the extension
    if save: c.experiment_name = name_file # Set the experiment name in the constants
    results_filename = os.sep.join(aux[:-1] + [c.evolution, f"{name_file}_results.json"])
    
    # Check if the directory exists, if not, create it #
    dir_ = os.path.dirname(results_filename)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)     
        
    return results_filename, name_file

def update_results_file(results_filename, experiment, index=None):
    # Update the results file with the experiment's current results #
    
    # Check if the results file exists, if not, create it #
    if os.path.exists(results_filename):
        with open(results_filename, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []

    # Check if the experiment already exists in the results
    if index is None:
        try:
            index = [r["name"] for r in results].index(experiment.name)
        except ValueError:
            index = -1

    experiment.generate_results_dictionary()
    
    if index >= 0: # If the experiment already exists, update its results 
        results[index]["num_runs"] += 1
        for k in experiment.results:
            if isinstance(experiment.results[k], list):
                results[index][k].extend(experiment.results[k])
            elif isinstance(experiment.results[k], (int, float)):
                # Example logic: track max fitness
                if "fitness" in k and experiment.results[k] > results[index].get(k, float('-inf')):
                    results[index][k] = experiment.results[k]
            else:
                results[index][k] = experiment.results[k]
    else: # If the experiment does not exist, add it to the results
        results.append(experiment.results)

    # Save the updated results to the file 
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=4)

def visualize_height_data(pos, height_data):
    # Visualizes the height data of the terrain before and after modification #
    
    # Calculate the difference between the two terrains 
    diff_map = c.height_data - height_data   
    
    # Visualize the old and the new height data
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.title('Previous Height Data')
    plt.imshow(c.height_data, cmap='terrain')
    plt.colorbar()
    plt.scatter(pos[0], pos[1], color='red', marker='x',s=100, label='Robot')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.title('New Height Data')
    plt.imshow(height_data, cmap='terrain')
    plt.colorbar()
    plt.scatter(pos[0], pos[1], color='red', marker='x',s=100, label='Robot')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff_map, cmap='RdYlGn')
    plt.title('Difference Between Terrains')
    plt.colorbar()
    plt.scatter(pos[0], pos[1], color='red', marker='x',s=100, label='Robot')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def modify_terrain():
    # Creates a new terrain in the environment and try to merge with the previous terrain using weighted sum #
    
    robot_pos, robot_orn = p.getBasePositionAndOrientation(c.robotId) # In order to not put the hill/hole in the same area as the robot we need to get its position 
    
    # Temporarily raise robot out of the way
    safe_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 5.0]
    p.resetBasePositionAndOrientation(c.robotId, safe_pos, robot_orn)

    center_offset = c.terrain_size // 2 # Offset to center the terrain to find the position of the robot given the size of the terrain
    pos = [int(robot_pos[0]/0.1) + center_offset, int(robot_pos[1]/0.1) + center_offset] # Get the x and y position of the robot
    
    height_data = np.zeros((c.terrain_size, c.terrain_size)) #  table that will contain the new terrain 
    
    r = 10 # radius around the robot that we want to maintain the previous height data
    
    heights = [] # Stores the new heights in order to do the normalization
    idxs = [] # Stores the indexes of the new heights
    
    # Randomize parameters for more complex terrains
    octaves_hills = random.randint(3, 6)
    persistence_hills = random.uniform(0.3, 0.6)
    lacunarity_hills = random.uniform(1.8, 2.2)

    octaves_holes = random.randint(2, 4)
    persistence_holes = random.uniform(0.4, 0.7)
    lacunarity_holes = random.uniform(1.8, 2.2)
    
    # Generate the holes and hills
    for i in range(c.terrain_size):
        for j in range(c.terrain_size):
            # Calculate the distance from the robot
            dist = np.sqrt((i - pos[0]) ** 2 + (j - pos[1]) ** 2)
            
            # Check if the distance is less than the radius
            if dist <= r+1:   
                height_data[i][j] = c.height_data[i][j] # Keep the old height
            
            else: # Generate a new height for this position 
                hill_height = pnoise2(i * c.scale_hills, j * c.scale_hills, octaves=octaves_hills, persistence=persistence_hills, lacunarity=lacunarity_hills) # Generate noise to create Hills
                
                hole_effect = pnoise2(i * c.scale_holes, j * c.scale_holes, octaves=octaves_holes, persistence=persistence_holes, lacunarity=lacunarity_holes) # Generate additional noise for creating holes
                
                height_data[i][j] = (hill_height * 0.8) - (hole_effect * 0.5) # Combine both noise layers 
                
                heights.append(height_data[i][j])
                idxs.append((i, j))
                
    # Convert the new heights to a numpy array
    heights = np.array(heights)
    
    # Normalize the new height data
    height_data_norm = (heights - np.min(heights)) / (np.max(heights) - np.min(heights)) * 0.5
    
    # Random Weighted sum percentage 
    weighted_sum_percentage_new = random.uniform(0.2, 0.4)
    weighted_sum_percentage_old = 1 - weighted_sum_percentage_new
    
    # Update the height data with the normalized values
    for idx, (i, j) in enumerate(idxs):
        height_data[i][j] = height_data_norm[idx]

        # Weighted sum of the two terrains 
        height_data[i][j] = (weighted_sum_percentage_new * height_data[i][j]) + (weighted_sum_percentage_old * c.height_data[i][j])    

    if c.plots: 
        visualize_height_data(pos, height_data)

    c.height_data = height_data

    c.heightfield_data = c.height_data.flatten()

    c.terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        flags = 0,
        meshScale=c.mesh_scale,
        heightfieldTextureScaling=c.terrain_size/2,
        heightfieldData=c.heightfield_data,
        numHeightfieldRows=c.terrain_size,
        numHeightfieldColumns=c.terrain_size,
        replaceHeightfieldIndex = c.terrain_shape
    )
    # c.terrainId = p.createMultiBody(0, c.terrain_shape)
    # p.resetBasePositionAndOrientation(c.terrainId, [0, 0, 0], [0, 0, 0, 1])
    # #p.changeVisualShape(c.terrainId, -1, rgbaColor=[1, 1, 1, 1])
    # p.changeDynamics(c.terrainId, -1, lateralFriction=c.friction_coefficient)
    
    # Place robot safely on the terrain after re-creating it
    final_pos = [robot_pos[0], robot_pos[1], robot_pos[2]+100]  
    p.resetBasePositionAndOrientation(c.robotId, final_pos, robot_orn)
    
def generate_rough_terrain():
    # Generates a rough terrain with holes and hills #
    c.height_data = np.zeros((c.terrain_size, c.terrain_size))

    # Generate random values to create the holes and hills
    for i in range(c.terrain_size):
        for j in range(c.terrain_size):
            # Generate noise to create hills
            hill_height = pnoise2(i * c.scale_hills, j * c.scale_hills, octaves=4, persistence=0.4, lacunarity=2.0)
            
            # Generate additional noise for creating holes
            hole_effect = pnoise2(i * c.scale_holes, j * c.scale_holes, octaves=3, persistence=0.5, lacunarity=2.0)
            
            # Combine the two noise layers to create hills and holes
            c.height_data[i][j] = (hill_height * 0.8) - (hole_effect * 0.5)
        
    # Normalize the height data
    c.height_data = (c.height_data - np.min(c.height_data)) / (np.max(c.height_data) - np.min(c.height_data)) * 0.5
    
    # Save the terrain height data
    np.savetxt(c.terrain_file, c.height_data, delimiter=',')

def rough_terrain():
    # Loads the rough terrain in the simulation #
    
    # Loads the terrain
    if os.path.exists(c.terrain_file): # Checks if the file that contains the terrain data exists
        c.height_data = np.loadtxt(c.terrain_file, delimiter=',').astype(np.float32)
    else: # Generate new terrain
        generate_rough_terrain()
        
    c.heightfield_data = c.height_data.flatten()  # Flatten to be applied on the PyBullet's function createCollisionShape
    
    # Create the terrain shape 
    c.terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=c.mesh_scale,   # Scale the terrain size and height
        heightfieldTextureScaling=c.terrain_scaling,
        heightfieldData=c.heightfield_data,
        numHeightfieldRows=c.terrain_size,
        numHeightfieldColumns=c.terrain_size
    )
    
    # Create the terrain object
    c.terrainId = p.createMultiBody(0, c.terrain_shape)
    p.resetBasePositionAndOrientation(c.terrainId, [0, 0, 0], [0, 0, 0, 1]) # Position [0, 0, 5.5]
    p.changeVisualShape(c.terrainId, -1, rgbaColor=[1, 1, 1, 1])            # Color
    
    # Set the friction coefficient of the terrain
    p.changeDynamics(c.terrainId, -1, lateralFriction=c.friction_coefficient)
        
def load_terrain():
    # Loads the terrain in the simulation #
    if c.terrain == "rough": # If the terrain is rough, load the rough terrain
        # set the start position for the robot
        c.start_pos = [0, 0, -1.5] #
        
        # Load terrain
        rough_terrain()
    else: # If the terrain is plane, load the plane terrain
        c.start_pos = [0, 0, 0.2] # set the start position for the robot
        c.terrainId = p.loadURDF(c.terrain, globalScaling=200) # Load the world - In this case, we use global scaling to make the world bigger to deal with the problem where the robot would fall off the edge of the world

def load_world():
    # Loads the world and the robot in the simulation #
    p.resetSimulation() # Reset the simulation

    load_terrain() # Load the terrain
    p.setTimeStep(c.delta_t) # Set the time step of the simulation
    p.setRealTimeSimulation(0) # Set to not use real time simulation, since we are on the training mode
    p.setGravity(0, 0, c.gravity) # Set the gravity of the simulation
    
    # If GUI is activated, configure the visualizer
    if c.GUI:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Disable GUI elements
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # Disable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) # Disable shadows

    # Load Robot
    # Base positions & orientations
    c.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    c.robotId = p.loadURDF(c.robot, c.start_pos, c.start_orientation) 
    
    # If gui is activated, enable rendering
    if c.GUI:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

def initialize_population(config, experiment_file):
    # Initialize the population with the best genomes from a previous experiment #

    # Get the best genome and second best genome from the offline evolution
    best_file = f"genomes/offline/best_genome_{os.path.splitext(os.path.basename(experiment_file))[0]}.pkl"
    second_file = best_file.replace("best_genome", "second_best_genome")

    # Load files
    with open(best_file, "rb") as bf:
        best_genome = pickle.load(bf)
        best_genome.key = 0
        best_genome.fitness = best_genome.fitness
    
    if c.num_initial_best_genomes == 2:
        with open(second_file, "rb") as sf:
            second_best_genome = pickle.load(sf)
            second_best_genome.key = 1
            second_best_genome.fitness = second_best_genome.fitness
    
    # Initialize a custom population
    pop = neat.Population(config)
    
    # Manually assign genomes to the population
    if c.num_initial_best_genomes == 2: pop.population = {0: best_genome, 1: second_best_genome}
    else: pop.population = {0: best_genome}

    # Fill the rest of the population using crossover between the two best genomes
    for i in range(c.num_initial_best_genomes, config.pop_size):
        # Create a new genome ID
        new_genome = config.genome_type(i)

        # Perform crossover
        if c.num_initial_best_genomes == 2:
            # Crossover between the two best genomes
            new_genome.configure_crossover(best_genome, second_best_genome, config.genome_config)
        else: 
            # Crossover with the best genome
            new_genome.configure_crossover(best_genome, best_genome, config.genome_config)

        # Mutate the new genome to add variation
        new_genome.mutate(config.genome_config)
        
        new_genome.fitness = 0.0

        # Add to the population
        pop.population[i] = new_genome
        
    # Do speciation with the generated population
    pop.species.speciate(config, pop.population, pop.generation)
    
    return pop

def get_worst_fitness(population, gene_keys, species, specie_id): 
    # Returns the worst fitness in the population #
    worst_fitness = float('inf')
    for g in population.values():
        if g.fitness is not None and g.fitness != 0.0 and g.key not in gene_keys and species.get(g.key) == specie_id and g.fitness < worst_fitness:
            worst_fitness = g.fitness
    return worst_fitness

def get_best_fitness(population, species, specie_id): 
    # Returns the best fitness in the population #
    best_fitness = -float('inf')
    for g in population.values():
        if g.fitness is not None and species.get(g.key) == specie_id and g.fitness > best_fitness:
            best_fitness = g.fitness
    return best_fitness

def hoeffding_inequality(t, best_fit, worst_fit):
    #  Calculates the Hoeffding bound used to decide if further evaluation is worthwhile # 
    return math.sqrt( ( (best_fit - worst_fit)**2 / (c.hoeffding_beta * t) )
                      * math.log(2 / c.hoeffding_alpha) )

def get_foot_link_indices():
    # Returns the indices of the foot links in the robot #
    foot_link_names = ["FrontLowerLeg", "BackLowerLeg", "LeftLowerLeg", "RightLowerLeg"]
    foot_link_indices = []
    for i in range(p.getNumJoints(c.robotId)):
        joint_info = p.getJointInfo(c.robotId, i)
        link_name = joint_info[12].decode("utf-8")
        if link_name in foot_link_names:
            foot_link_indices.append(i)
    return foot_link_indices

def check_orientation():
    # Checks the orientation of the robot #
    # Returns true if the robot is upright, false otherwise

    # Get the base orientation of the robot
    _, orientation = p.getBasePositionAndOrientation(c.robotId)
    
    # Get the rotation matrix
    rot  = p.getMatrixFromQuaternion(orientation) 
    dot = np.array([rot[6], rot[7], rot[8]]).dot([0, 0, 1])
    
    # Check if the robot is upright 
    if dot > 0.5: # z-axis should be close to 0
        return True
    else:   # Case when the robot is sideways or upside down
        return False 

def eval_genomes(genomes, config, reporter, species): 
    # Evaluates the genomes in the population #
    population = dict(genomes)
    skip_count = 0  # Track how many we skip in this generation
    ind_lengths = [] # Track the duration of the skipped individual
    fitness_values = [] # Track the fitness of the skipped individual
    gene_keys = [] # Track the ids of the skipped individuals
    num_bad_orientations = 0 # Track the number of bad orientations
    generation = reporter.generation # Get the current generation from the reporter
    
    reporter.start_generation(species=species)
    
    # Applies dynamic environment if it is enabled
    if c.dynamic_environment:
        # For every 10 generations, modifies the terrain
        if generation % 5 == 0: #if reporter.generation % 1 == 0 and reporter.generation != 1: # 10
            # Get position of the robot
            # pos, orn = p.getBasePositionAndOrientation(c.robotId)
            # new_pos = [pos[0], pos[1], pos[2] + 50]
            
            # Modify the terrain
            modify_terrain() 
            
            # wait for the robot to touch the ground again
            touching = False
            foot_links = get_foot_link_indices() # Get the indices of the foot links
            while not touching:
                p.stepSimulation()
                # Check contact for any foot link
                for link_idx in foot_links:
                    contacts = p.getContactPoints(bodyA=c.robotId, linkIndexA=link_idx)
                    if contacts:
                        touching = True
                        break  
            
    for i, (genome_id, genome) in enumerate(population.items()):
        # Build the neural net for this genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # If the evolution is offline, reset the robot to its initial position
        # and set all joint states to 0.0
        if c.evolution == "offline":
            p.resetBasePositionAndOrientation(c.robotId, c.start_pos, c.start_orientation)
            num_joints = p.getNumJoints(c.robotId)
            for j in range(num_joints):
                p.resetJointState(c.robotId, j, targetValue=0.0, targetVelocity=0.0)

        # Implements the Recovery Mechanism
        if c.recoveryMechanism:
            for step in range(c.recovery_length):
                move_robot(net, generation)
                p.stepSimulation()
        
        # Initialize assessment parameters
        prevPos, _ = p.getBasePositionAndOrientation(c.robotId)
        total_distance = 0.0
        start_pos = prevPos
        
        # Simulation loop
        for step in range(c.simulation_length):
            move_robot(net, generation)
            p.stepSimulation()                
                
            # Check if the robot is not upright
            if not check_orientation(): 
                num_bad_orientations += 1
                
            current_pos, _ = p.getBasePositionAndOrientation(c.robotId)
            total_distance = np.linalg.norm(np.array(current_pos) - np.array(prevPos))
            t = (step+1) * c.delta_t
            #prevPos = current_pos
            
            if c.terrain == "rough":
                # check if it falls from the map
                if current_pos[2] < -50.0:  # e.g., threshold = -3.0
                    print(f"[WARNING] Robot fell off the terrain")
                    return

            # Partial fitness of the individual
            if c.distance_track:
                partial_fitness = total_distance # In meters
            else:
                partial_fitness = total_distance/t # In meters/seconds
            
            # Racing step - It only starts to evaluate the individual after a min duration
            if c.racing and step > c.racing_min_length: # and reporter.generation > 1:       
                specie_id = reporter.genome_to_species.get(genome_id)  
                species = reporter.genome_to_species        
                worst_fit = get_worst_fitness(population, gene_keys, species, specie_id)
                best_fit  = get_best_fitness(population, species, specie_id)
                
                hoeff_bound = hoeffding_inequality(t, best_fit, worst_fit)
                
                # print(f"[RACING] Gen {reporter.generation} | Genome {genome_id} | Step: {step} | t: {t:.2f}")
                # print(f"         Partial Fitness: {partial_fitness:.4f}, Worst Fit: {worst_fit:.4f}, Bound: {hoeff_bound:.4f}")
                
                #print(partial_fitness + 2*hoeff_bound, "----", worst_fit)
                    
                # skips the individual if it is most likely to not improve compared to the current worst individual in the population
                if partial_fitness + 2*hoeff_bound < worst_fit:
                    skip_count += 1
                    fitness_values.append(partial_fitness) # Register the fitness of the current individual
                    gene_keys.append(genome.key) # Register the id of the current individual
                    # print(f"--> SKIPPING Genome {genome_id} (partial_fitness + 2*bound = {(partial_fitness + 2*hoeff_bound):.4f}) < worst_fit = {worst_fit:.4f}")
                    break 
                
        ind_lengths.append(t) # Register the duration of the evaluation of the current individual
        
        current_pos, _ = p.getBasePositionAndOrientation(c.robotId)
        total_distance = np.linalg.norm(np.array(current_pos) - np.array(start_pos))
        new_fitness = 0.0
        
        # Assign the fitness (total distance) to the genome
        if c.distance_track:
            new_fitness = total_distance # In meters
        else: 
            new_fitness = total_distance/(c.simulation_length * c.delta_t) if not c.racing else partial_fitness # In meters/seconds
        
        if genome.fitness is None:
            genome.fitness = new_fitness
        else: # If the genome already has a fitness value, update it if the new fitness is better
            if c.fitness == "normal_best":
                if new_fitness > genome.fitness:  
                    genome.fitness = new_fitness
            
            elif c.fitness == "average":
                # exponential moving average
                genome.fitness = c.alpha * new_fitness + (1.0 - c.alpha) * genome.fitness
                
            else: # "normal"
                genome.fitness = new_fitness
            
        # print(f"Genome {genome_id} fitness: {genome.fitness}")
        
    if not ind_lengths:
        ind_lengths = [0.0]
    if not fitness_values:
        fitness_values = [0.0]
    reporter.register_generation(skip_count, ind_lengths, fitness_values, num_bad_orientations)
    # print(f"[SUMMARY] Generation {reporter.generation} | Skipped: {skip_count} individuals")

def get_observation(include_xy=False):
    # Gets and returns the quadruped's full observation vector #

    # Base position/orientation
    basePos, baseOrn = p.getBasePositionAndOrientation(c.robotId)

    info = {}
    if include_xy:
        torso_xy = list(basePos[:2])
        info["x_position"] = basePos[0]
        info["y_position"] = basePos[1]

    z = basePos[2]
    orn = list(baseOrn)  # quaternion: [x, y, z, w]

    # Base linear and angular velocity
    baseLinVel, baseAngVel = p.getBaseVelocity(c.robotId)
    linVel = list(baseLinVel)
    angVel = list(baseAngVel)

    # Joint states
    jointAngles = []
    jointVels = []
    jointTorques = []

    num_joints = p.getNumJoints(c.robotId)

    for j in range(num_joints):
        joint_state = p.getJointState(c.robotId, j)
        jointAngles.append(joint_state[0])     # position
        jointVels.append(joint_state[1])       # velocity
        jointTorques.append(joint_state[3])    # applied torque

    # Foot contact sensors
    foot_link_names = ["FrontLowerLeg", "BackLowerLeg", "LeftLowerLeg", "RightLowerLeg"]
    foot_link_indices = []
    for i in range(p.getNumJoints(c.robotId)):
        joint_info = p.getJointInfo(c.robotId, i)
        link_name = joint_info[12].decode("utf-8")
        if link_name in foot_link_names:
            foot_link_indices.append(i)

    footContacts = []
    for link in foot_link_indices:
        contact_points = p.getContactPoints(bodyA=c.robotId, linkIndexA=link)
        footContacts.append(1 if contact_points else 0)

    # Final observation vector
    obs = [z] + orn + jointAngles + linVel + angVel + jointVels + footContacts + jointTorques

    if include_xy:
        obs = torso_xy + obs
        return np.array(obs, dtype=np.float32), info
    else:
        return np.array(obs, dtype=np.float32)

def get_observation_2(include_xy=False):
    # Gets and returns the ant's observation vector #
    
    # Base position/orientation 
    basePos, baseOrn = p.getBasePositionAndOrientation(c.robotId)

    info = {}
    if include_xy:
        torso_xy = list(basePos[:2])
        info["x_position"] = basePos[0]
        info["y_position"] = basePos[1]
    
    # Use only the z-coordinate for observation (by default, x,y are excluded)
    z = basePos[2]
    orn = list(baseOrn)  # [x, y, z, w]

    # Get base velocities 
    baseLinVel, baseAngVel = p.getBaseVelocity(c.robotId)
    linVel = list(baseLinVel)  # [vx, vy, vz]
    angVel = list(baseAngVel)  # [wx, wy, wz]

    # Get joint angles and angular velocities
    jointAngles = []
    jointVels = []
    
    if (c.robot == "robots\\ant_feet.urdf"):
        num_joints = p.getNumJoints(c.robotId) - 4  # Exclude the 4 fixed joints
    else:
        num_joints = p.getNumJoints(c.robotId)
    for j in range(num_joints):
        joint_state = p.getJointState(c.robotId, j)
        jointAngles.append(joint_state[0])  # angle
        jointVels.append(joint_state[1])    # angular velocity

    # Build observation: positions first, then velocities
    obs = [z] + orn + jointAngles + linVel + angVel + jointVels
    if include_xy:
        obs = torso_xy + obs
        return np.array(obs, dtype=np.float32), info
    else:
        return np.array(obs, dtype=np.float32)

def move_robot(net, generation=0):
    # Moves the robot using an individual's brain #
    obs = get_observation()
    outputs = net.activate(obs)

    if (c.robot == "robots\\ant_feet.urdf"):
        num_joints = p.getNumJoints(c.robotId) - 4  # Exclude the 4 fixed joints
    else:
        num_joints = p.getNumJoints(c.robotId)
    
    for j in range(num_joints):
        # Apply a failure test for joints after a certain number of generations
        if c.fail_test and ((generation >= 50 and j == 0)): # or (generation >= 100 and j == 1)):
            joint_state = p.getJointState(c.robotId, j)
            joint_position = joint_state[0]
            output = joint_position 
        else:   
            output = outputs[j] * c.motor_joint_range
        
        p.setJointMotorControl2(
            bodyIndex=c.robotId,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition= output,
            force=c.motor_max_force,
            maxVelocity=1.0
        )
        
        if c.GUI:
            # Get camera position
            yaw = p.getDebugVisualizerCamera()[8]
            pitch = p.getDebugVisualizerCamera()[9]
            dist = p.getDebugVisualizerCamera()[10]
            
            # Change camera position
            p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=p.getBasePositionAndOrientation(c.robotId)[0])


def plot_mean_std_over_x(x_vals, y_vals, title, ax, ylabel="Fitness"):
    # Plot the mean and standard deviation of y_vals over x_vals #
    x_to_ys = collections.defaultdict(list)
    for x, y in zip(x_vals, y_vals):
        x_to_ys[x].append(y)

    sorted_x = sorted(x_to_ys.keys())
    mean_y = [np.mean(x_to_ys[x]) for x in sorted_x]
    std_y = [np.std(x_to_ys[x]) for x in sorted_x]

    sorted_x = np.array(sorted_x)
    mean_y = np.array(mean_y)
    std_y = np.array(std_y)

    ax.plot(sorted_x, mean_y, label=title)
    ax.fill_between(sorted_x, mean_y - std_y, mean_y + std_y, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Evaluation Length (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True)

def plot_mean_over_time(input_data = None, dataset=None, name = "name", x_label = "x_label", y_label="y_label", y_limit = None, show=False, title=None, second_dataset=None, color=None):
    # Plots the mean of each dataset over time with the confidence interval
    _, ax = plt.subplots() # generate figure and axes
    input_data = [np.array(x) for x in input_data if isinstance(x, list)]
    input_data = np.array(input_data)
    if isinstance(name, str): name = [name]; input_data = [input_data]

    if len(name) == 1:
        no_legend = True
    else:
        no_legend = False

    for index, this_name in enumerate(name):
        this_input_data = np.array(dataset[index], dtype=float)
        this_input_data[np.isinf(this_input_data)] = np.nan
        
        total_generations = this_input_data.shape[1]
        mean_vals = np.nanmean(this_input_data, axis=0) # mean across runs
        gens = np.arange(total_generations)

        std_vals = np.nanstd(this_input_data, axis=0)
        ax.fill_between(gens, mean_vals-std_vals, mean_vals+std_vals, alpha=0.3)

        # Plot the mean line
        ax.plot(gens, mean_vals, label=this_name)

        ax.set_xlabel(x_label) # add axes labels
        ax.set_ylabel(y_label)

        if y_limit: ax.set_ylim(y_limit[0],y_limit[1])
        if title is not None:
            plt.title(title)
        else:
            plt.title(y_label)

    # if the second dataset is given, plot it a line in the x axis
    if second_dataset is not None: 
        # Transform the dataset into an 1D array
        flattened = []
        for sublist in second_dataset:
            if isinstance(sublist, (list, np.ndarray)):
                for item in sublist:
                    if isinstance(item, (list, np.ndarray)):
                        flattened.extend(int(x) for x in item)
                    else:
                        flattened.append(int(item))
            else:
                flattened.append(int(sublist))
        
        second_dataset = sorted(set(flattened))  # remove duplicates
        
        # Plot dots
        if len(name) == 1:
            mean_vals = np.nanmean(np.array(dataset[0]), axis=0)
            for gen_idx in second_dataset:
                if 0 <= gen_idx < len(mean_vals):
                    ax.plot(gen_idx, mean_vals[gen_idx], 'ro', markersize=5)  # red dot
        
    if not no_legend:
        plt.legend(loc='best', fontsize=8) # add legend
    if show:
        plt.tight_layout()
        plt.show() 

def visualize_best_genome(file="best_genome_experiments.pkl", steps=2000, record=False, video_filename="simulation.mp4"): 
    # Visualize the best genome #
    
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.gravity)
    p.setTimeStep(1.0/240.0)
    
    # Record the simulation
    if record:
        vid_log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # Set the start position based on the terrain type and size
    if c.terrain == "rough": 
        c.start_pos = [0, 0, -0.2]
    else: 
        c.start_pos = [0, 0, 0.2] 

    p.loadURDF(c.terrain)
    c.robotId = p.loadURDF(c.robot, c.start_pos, p.getQuaternionFromEuler([0, 0, 0]))

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    with open(file, "rb") as f:
        genome = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, c.config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Variable for the distance traveled
    prevPos, _ = p.getBasePositionAndOrientation(c.robotId)

    for _ in range(steps):
        obs = get_observation()
        outputs = net.activate(obs)
        num_joints = p.getNumJoints(c.robotId)
        for j in range(num_joints):
            output = np.tanh(outputs[j])
            p.setJointMotorControl2(
                bodyIndex=c.robotId,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition= np.clip(output * c.motor_joint_range, -0.5, 0.5),
                force=c.motor_max_force
            )
        
        p.stepSimulation()
        
        # Get camera position
        yaw = p.getDebugVisualizerCamera()[8]
        pitch = p.getDebugVisualizerCamera()[9]
        dist = p.getDebugVisualizerCamera()[10]
        
        # Change camera position
        p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=p.getBasePositionAndOrientation(c.robotId)[0])

        time.sleep(1/240.0)
        
    # Calculate the distance traveled
    lastPos = p.getBasePositionAndOrientation(c.robotId)[0]
    step_distance = np.linalg.norm(np.array(lastPos) - np.array(prevPos))
    total_distance = step_distance
    avg_speed = total_distance / (steps * (1/240.0))  # Average speed in meters/second
    
    # Stop recording the simulation
    if record:
        p.stopStateLogging(vid_log)
    
    p.disconnect()
    
    print(f"Total distance traveled: {total_distance:.2f} meters")
    print(f"Average Speed: {avg_speed:.2f} meters")
    

def print_genome_info(genome_file):
    # Load genome
    with open(genome_file, "rb") as f:
        genome = pickle.load(f)
        
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        c.config
    )
    
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    all_nodes = set(genome.nodes.keys())
    
    hidden_nodes = list(all_nodes - set(input_nodes) - set(output_nodes))

    print(f"Input nodes: {len(input_nodes)}, Hidden nodes: {len(hidden_nodes)}, Output nodes: {len(output_nodes)}")

def draw_genome(config_file, genome_file, output_filename=None):
    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Load genome
    with open(genome_file, "rb") as f:
        genome = pickle.load(f)

    # Setup pydot graph
    graph = pydot.Dot(graph_type='digraph', rankdir="LR")

    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [n for n in genome.nodes if n not in input_nodes and n not in output_nodes]

    # Helper to add a node
    def add_node(nid, label, color):
        graph.add_node(pydot.Node(str(nid), label=label, style="filled", fillcolor=color))

    # Add input nodes
    for nid in input_nodes:
        add_node(nid, f"Input\n{nid}", "lightblue")

    # Add output nodes
    for nid in output_nodes:
        add_node(nid, f"Output\n{nid}", "lightgreen")

    # Add hidden nodes
    for nid in hidden_nodes:
        add_node(nid, f"Hidden\n{nid}", "orange")

    # Add connections
    for conn_key, conn in genome.connections.items():
        from_node, to_node = conn_key
        color = "black" if conn.enabled else "gray"
        style = "solid" if conn.enabled else "dotted"
        label = f"{conn.weight:.2f}"

        edge = pydot.Edge(str(from_node), str(to_node), label=label, color=color, style=style)
        graph.add_edge(edge)

    if output_filename is None:
        output_filename = genome_file.replace(".pkl", ".png")

    # Save the graph
    graph.write_png(output_filename)
    print(f"NEAT graph saved to: {output_filename}")

    
'''     
    FUNCTIONS ADAPTED FROM 
    https://github.com/CodeReclaimers/neat-python/blob/master/examples/openai-lander/visualize.py
'''        
def draw_net(config, genome_file="best_genome_experiments.pkl", filename=None, node_names=None, node_colors=None,
            fmt='png', view=False, node_size=800, node_colors_lookup=None, 
            input_node_color='lightsteelblue', output_node_color='lightsteelblue',
            hidden_node_color='lightsteelblue'):
    """ Visualizes a NEAT network in an organized, clean way.
    """
    
    if node_names is None:
        node_names = {}
    
    if node_colors is None:
        node_colors = {}
    
    node_colors_lookup = node_colors_lookup or {
        'input': input_node_color,
        'output': output_node_color,
        'hidden': hidden_node_color
    }
    
    # Create figure and layout
    plt.figure(figsize=(12, 9))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=0, hspace=0)
    
    with open(genome_file, "rb") as f:
        genome = pickle.load(f)
    
    # Get network info
    nodes = list(genome.nodes.keys())
    connections = [(conn.key[0], conn.key[1], conn.weight, conn.enabled) for conn in genome.connections.values()]
    
    # Get node positions
    node_positions = {}
    
    # Position input and output nodes
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [n for n in nodes if n not in input_nodes and n not in output_nodes]
    
    # Layout nodes
    for i, node_id in enumerate(input_nodes):
        node_positions[node_id] = (-1.0, (i + 1) / float(len(input_nodes) + 1))
    
    for i, node_id in enumerate(output_nodes):
        node_positions[node_id] = (1.0, (i + 1) / float(len(output_nodes) + 1))
    
    # Position hidden nodes in layers
    if len(hidden_nodes) > 0:
        # Get a graph representation of the connections
        connections_undir = set()
        for conn in connections:
            if conn[3]:  # If connection is enabled
                connections_undir.add((conn[0], conn[1]))
                connections_undir.add((conn[1], conn[0]))
        
        # For each hidden node, assign a layer
        node_layers = {}
        for node_id in hidden_nodes:
            # Find the shortest path from input to this node
            in_distance = float('inf')
            for input_id in input_nodes:
                dist = find_shortest_path_length(input_id, node_id, connections_undir)
                if dist is not None and dist < in_distance:
                    in_distance = dist
            
            # Find the shortest path from this node to any output
            out_distance = float('inf')
            for output_id in output_nodes:
                dist = find_shortest_path_length(node_id, output_id, connections_undir)
                if dist is not None and dist < out_distance:
                    out_distance = dist
            
            # Assign layer
            if in_distance == float('inf') or out_distance == float('inf'):
                # If node can't reach an input or output, place it in the middle
                layer = 0.5
            else:
                # Position between input and output based on distances
                total_distance = in_distance + out_distance
                layer = in_distance / total_distance if total_distance > 0 else 0.5
            
            node_layers[node_id] = layer
        
        # Position nodes within each layer
        layer_nodes = {}
        for node_id, layer in node_layers.items():
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(node_id)
        
        for layer, nodes_in_layer in layer_nodes.items():
            for i, node_id in enumerate(nodes_in_layer):
                horz_position = -1.0 + (layer * 2.0)
                vert_position = (i + 1) / float(len(nodes_in_layer) + 1)
                node_positions[node_id] = (horz_position, vert_position)
    
    # Draw the network
    plt.xlim(-1.2, 1.2)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Draw connections
    connecting_lines = []
    connection_colors = []
    
    for conn in connections:
        from_pos = node_positions[conn[0]]
        to_pos = node_positions[conn[1]]
        weight = conn[2]
        enabled = conn[3]
        
        # Calculate the line color based on weight and enabled status
        if enabled:
            if weight > 0:
                color = (0, 0, 0.5 + min(0.5, abs(weight)))
            else:
                color = (0.5 + min(0.5, abs(weight)), 0, 0)
        else:
            color = (0.5, 0.5, 0.5, 0.5)
        
        connecting_lines.append([from_pos, to_pos])
        connection_colors.append(color)
    
    # Draw the connections
    if connecting_lines:
        line_collection = LineCollection(connecting_lines, colors=connection_colors, alpha=0.8, linewidths=1.0)
        plt.gca().add_collection(line_collection)
    
    # Draw nodes
    for node_id in nodes:
        if node_id in node_positions:
            pos = node_positions[node_id]
            
            # Determine node type and color
            if node_id in input_nodes:
                node_type = 'input'
                shape = 'o'
            elif node_id in output_nodes:
                node_type = 'output'
                shape = 's'
            else:
                node_type = 'hidden'
                shape = 'o'
            
            color = node_colors.get(node_id, node_colors_lookup.get(node_type, 'black'))
            
            plt.scatter(pos[0], pos[1], s=node_size, color=color, marker=shape, zorder=2)
            
            # Draw node label
            node_label = node_names.get(node_id, str(node_id))
            plt.annotate(node_label, xy=pos, xytext=(-5, -5), textcoords='offset points',
                        ha='center', va='center', fontsize=12)
    
    # Add title
    plt.title(f"NEAT Network - {len(input_nodes)} inputs, {len(hidden_nodes)} hidden, {len(output_nodes)} outputs", 
            fontsize=16, pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=input_node_color, label='Input Nodes'),
        mpatches.Patch(color=hidden_node_color, label='Hidden Nodes'),
        mpatches.Patch(color=output_node_color, label='Output Nodes'),
        mpatches.Patch(color=(0, 0, 1), label='Positive Weight'),
        mpatches.Patch(color=(1, 0, 0), label='Negative Weight'),
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                ncol=3, frameon=False)
    
    # Save or show the figure
    if filename:
        plt.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
    
    if view:
        plt.show()
    
    plt.close()
    
def find_shortest_path_length(start_node, end_node, connections):
    """Finds shortest path length between two nodes in an undirected graph."""
    if start_node == end_node:
        return 0
    
    # BFS to find shortest path
    queue = [(start_node, 0)]
    visited = set([start_node])
    
    while queue:
        current, distance = queue.pop(0)
        
        next_nodes = [n2 for (n1, n2) in connections if n1 == current and n2 not in visited]
        
        for next_node in next_nodes:
            if next_node == end_node:
                return distance + 1
            
            visited.add(next_node)
            queue.append((next_node, distance + 1))
    
    return None  # No path found
