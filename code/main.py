import argparse
import numpy as np
import traceback
from tqdm import trange
import os
import json
import random
import pickle
import copy
import time
import sys
import math

import neat
from neat.checkpoint import Checkpointer
import pybullet as p
import pybullet_data

from experiment import Experiment
import constants as c

from fitnessHistoryReporter import FitnessHistoryReporter

from util import *

def main(args = None):
    try:
        # Get number of experiment runs, if exists
        if args and args.experiment_runs: 
            runs = int(args.experiment_runs)
        else:
            runs = 1
            
        # Get visualization mode
        if args and args.visualize:
            c.train = False

        # Get experiment file
        if args and args.experiment_file:
            experiment_file = args.experiment_file # use the defined file
        elif c.experiment_name is None:
            experiment_file = "experiments\\experiment.json" # default experiment file
        else:
            experiment_file = c.experiment_name
        
        # Check if the environment will be dynamic or static
        if args and args.dynamic_environment:
            c.dynamic_environment = True
            c.terrain = "rough"
                
        # Get the type of evolution that we are going to use
        if args and args.evolution_type: 
            c.evolution = "offline" # offline evolution

        # File that will contain the results
        results_filename, name_file = set_results_file(experiment_file)
        
        # File that will store the best genome
        aux = "online" if c.evolution == "online" else "offline"
        if c.run == None:
            best_brain_filename = f"genomes\\{aux}\\best_genome_{os.path.splitext(os.path.basename(experiment_file))[0]}.pkl"
        else:
            best_brain_filename = f"genomes\\{aux}\\best_genome_{os.path.splitext(os.path.basename(experiment_file))[0]}_run_{c.run}.pkl"
        second_brain_filename = best_brain_filename.replace("best_genome", "second_best_genome")
    
        # Variables to track the best genomes
        best_brain = None
        best_fitness = -math.inf
        second_best_brain = None
        second_best_fitness = -math.inf
        
        # Create experiment objects based on the experiment file
        name, controls, conditions = Experiment.load_conditions_file(experiment_file) # Get all the conditions on the experiment file
        experiments = [Experiment(condition, controls, args, runs) for condition in conditions] # For each condition in the experiment file creates an experiment object


        # Training mode #
        if c.train: 
            print("Training mode")
            
            # Checkpoint handling 
            if args and args.checkpoint_file:
                checkpoint_filename = args.checkpoint_file # Check if a checkpoint file is provided
            else:
                if c.run is None:
                    checkpoint_filename = f"checkpoints\\{aux}\\checkpoint_{name_file}_"
                else:
                    checkpoint_filename = f"checkpoints\\{aux}\\checkpoint_{name_file}_run_{c.run}_"
            
            # Check if the checkpoint directory exists
            checkpoint_dir = os.path.dirname(checkpoint_filename)
            if checkpoint_dir and not os.path.exists(checkpoint_dir): 
                os.makedirs(checkpoint_dir)  # Creates it
            checkpointer = Checkpointer(filename_prefix=checkpoint_filename) # Initialize the checkpointer

            results =[] # List that store the results of the experiments
            if os.path.exists(results_filename): 
                with open(results_filename, "r") as f:
                    try:
                        results = json.load(f)
                    except:
                        ...
                        
            if os.path.exists(results_filename):
                if multiprocessing.current_process().name == "MainProcess":
                    user_input = input(f"\n[!] Results file '{results_filename}' already exists.\nDo you want to wipe it? (y/n): ")
                    if user_input.strip().lower() == "y":
                        os.remove(results_filename)
                    else:
                        print("[!] Aborting.")
                        return
                else:
                    print(f"[Parallel] Skipping prompt and continuing with existing results file '{results_filename}'")
            
            if not os.path.exists(results_filename) or len(results)==0 and args and not args.checkpoint: # Creates the results file if it does not exist
                with open(results_filename, "w+") as f:
                    f.write("[\n")
                    for experiment in experiments:
                        experiment.generate_empty_results_dictionary()
                        r = experiment.results
                        r["num_runs"] = 0
                        json.dump(r, f, indent=4)
                        if experiment != experiments[-1]:f.write(",\n")
                        
                    f.write("\n]")
                    f.close()

            if args:
                c.GUI = args.gui_activated # Check if GUI is activated

            # Initialize PyBullet 
            if c.GUI: # Check if GUI argument is requested
                p.connect(p.GUI) # GUI mode
            else:
                p.connect(p.DIRECT) # Direct mode 
                
            p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Add to the search path the PyBullet data directory
            
            print(f"Running experiment: {name}")
            for i, experiment in enumerate(experiments): # For each experiment runs the evolution
                print("_"*80)
                print(f"\tCondition {i} ({experiment.name})")

                saved_results = results[i] if args and args.checkpoint and len(results) > i else None # Load the results from the previous runs of the experiment, if exists
                
                # Load the conditions from the experiment file and apply them on the constants.py variables
                experiment.apply_condition() 
                experiment.setup_arrays(saved_results)
                
                if args and args.checkpoint:
                    pop_checkpoint = Checkpointer.restore_checkpoint(checkpoint_filename) # Restore the population from the checkpoint file
                    experiment.start_gen = pop_checkpoint.generation 
                    if c.parameter_control:
                        set_simulation_length(c.run, results_filename)
                        
                    if args.num_gens:
                        c.num_gens_old = pop_checkpoint.generation # Store the old number of generations
                        c.num_gens = int(args.num_gens)
                        
                    base_name = os.path.basename(results_filename).split(".")[0]
                    base_name = base_name.replace("results", f"_run_{c.run}_results") # Get the base name of the results file
                    results_filename = os.path.join("experiments",c.evolution, f"{base_name}.json")
                    
                pbar = trange(runs)
                for run in pbar: # Run the current experiment for N runs
                    if c.run is None: c.run = int(run) # Convert run to int
                    
                    load_world() # Loads the world with the robot and the environment in PyBullet
                
                    # Given the stochastic nature of the algorithm, we need to set the seed for each run
                    if args and args.checkpoint and args.seed: 
                        seed = int(args.seed) # Get the seed from the args
                    else: 
                        seed = c.SEEDS[c.run] # Get the seed from the list of seeds defined in constants.py
                    np.random.seed(seed)
                    random.seed(seed)

                    experiment.current_run = run # set run number
                    c.simulation_length = c.simulation_length_default # reset simulation length

                    # Initialize NEAT with pybullet #
                    # check if checkpoint is activated
                    if args and args.checkpoint:
                        pop = copy.deepcopy(pop_checkpoint) # Copy the population from the original checkpoint  
                        config = pop.config # Get the configuration from the population
                        
                        # Checks if the population size can still be the same as the one in the checkpoint
                        if args and args.checkpoint_population: 
                            checkpoint_pop = int(args.checkpoint_population) # Get the population size from the args
                            config.pop_size = checkpoint_pop
                        
                            # Add new genomes
                            next_id = max(pop.population.keys()) + 1
                            while len(pop.population) < checkpoint_pop:
                                genome = config.genome_type(next_id)
                                genome.configure_new(config.genome_config)
                                pop.population[next_id] = genome
                                next_id += 1
                        
                            # Re-speciate the population to ensure that the species are still valid
                            pop.species.speciate(config, pop.population, pop.generation)
                        
                    else:
                        # Load the NEAT configuration file
                        config = neat.config.Config(
                            neat.DefaultGenome, 
                            neat.DefaultReproduction, 
                            neat.DefaultSpeciesSet, 
                            neat.DefaultStagnation, 
                            c.config) # Load NEAT config

                        # Initialize the population #
                        if c.initial_population == "best_genomes":
                            pop = initialize_population(config, experiment_file) # Create the population based on pre-trained genomes
                        else:
                            pop = neat.Population(config) # Create population

                    fitness_reporter = FitnessHistoryReporter() # Fitness reporter
                    pop.add_reporter(fitness_reporter) # Add the fitness reporter to the NEAT algorithm
                    #

                    # Evaluation function
                    def eval_wrapper(genomes, neat_config): 
                        eval_genomes(genomes, neat_config, fitness_reporter, pop.species)
                    
                    pop.run(eval_wrapper, c.num_gens) # Run NEAT for N generations               
                    
                    # Track the best genome from the current run
                    population = [g for g in list(pop.population.values()) if g.fitness is not None] # Get the population of genomes
                    population.sort(key=lambda g: g.fitness, reverse=True) # sort the population by fitness
                    current_best = population[0] # Get the best genome from the population
                    current_fitness = current_best.fitness # Get the fitness of the best genome
                    current_second  = population[1] # Get the second best genome from the population	
                    current_second_fitness = current_second.fitness # Get the fitness of the second best genome
                    
                    # Check if the best genome from the current run is better than the best genome from the previous runs
                    if current_fitness > best_fitness:
                        second_best_fitness = best_fitness
                        second_best_brain   = best_brain
                        best_fitness        = current_fitness
                        best_brain          = current_best
                        
                    # Check if the second best genome from the current run is better than the second best genome from the previous runs
                    if current_second is not None and current_second_fitness > second_best_fitness:
                            second_best_fitness = current_second_fitness
                            second_best_brain   = current_second

                    # Save the results of the current run to the experiment object
                    experiment.record_results(
                        individuals_skipped_over_time=np.array(fitness_reporter.skip_counts_over_time),
                        simulation_length=np.array(fitness_reporter.sim_length_history),
                        fitness_over_time=np.array(fitness_reporter.best_fitness_history),
                        diversity_over_time=np.array(fitness_reporter.diversity_over_time),
                        species_over_time=np.array(fitness_reporter.species_counts_over_time),
                        nodes_over_time=np.array(fitness_reporter.avg_nodes_history),
                        connections_over_time=np.array(fitness_reporter.avg_connections_history),
                        ind_length_over_time=np.array(fitness_reporter.ind_length_over_time),
                        fitness_racing_history=np.array(fitness_reporter.fitness_racing_history),
                        bad_orientations_gen=fitness_reporter.bad_orientations_gen # Track the generations where the robot is not oriented correctly
                    )
                    
                    experiment.generate_results_dictionary()  
                    
                    update_results_file(results_filename, experiment.results, index=i)
                    # End run #
                # End experiment #
            
            p.disconnect() # Disconnect PyBullet    
            
            # Save the best genomes from the experiments #
            # Check if directory exists
            genome_dir = os.path.dirname(best_brain_filename)
            if genome_dir and not os.path.exists(genome_dir):  # If the directory does not exist, create it
                os.makedirs(genome_dir)
            # Write the best genomes to the files
            with open(best_brain_filename, "wb") as bf: # Best genome
                pickle.dump(best_brain, bf)
            if aux == "offline":
                with open(second_brain_filename, "wb") as bf: # Second best genome
                    pickle.dump(second_best_brain, bf)
                
            # Save the checkpoint of the population #
            checkpointer.save_checkpoint(config, pop.population, pop.species, pop.generation)

            # Visualization mode #
            aux = "" if c.evolution == "online" else "-e"
            os.system(f"python main.py -v -f {experiment_file} {aux}") # Run the main file with the visualization mode
        else: 
            # Debug mode #
            if args and args.debug: 
                # Plot the correlation between fitness and simulation length
                if args and args.second_experiment_file: # check if a second experiment file is provided
                    second_file = args.second_experiment_file
                    second_results_file, _ = set_results_file(second_file, False)
                    os.system(f"python analyze_experiment.py -f {results_filename} -fs {second_results_file} -a") 
                else:
                    os.system(f"python analyze_experiment.py -f {results_filename} -a") 
            # Visualization mode #
            else: 
                if args and args.record:
                    record = True
                else:
                    record = False
                    
                if args and args.video_filename:
                    video_filename = args.video_filename
                else:
                    video_filename = "simulation.mp4"
                
                for i, experiment in enumerate(experiments): # For each experiment runs the evolution
                    print("_"*80)
                    print(f"\tCondition {i} ({experiment.name})")

                    # Load the conditions from the experiment file and apply them on the constants.py variables
                    experiment.apply_condition() 
                    experiment.setup_arrays()
                    
                    # # Print the best genome info
                    # print_genome_info(best_brain_filename)
                    
                    # Visualize the behavior of the best controller in the physics simulator 
                    visualize_best_genome(best_brain_filename, record=record, video_filename=video_filename)
                    
                    # # Visualize the nodes and connections of the best controller 
                    # config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, c.config)
                    # draw_net(config, best_brain_filename, view=True)
                    # draw_genome(c.config, best_brain_filename)
                    
                # Plot the results
                os.system(f"python analyze_experiment.py -f {results_filename} &") 

                # Disconnect PyBullet
                if p.isConnected(): 
                    p.disconnect()

    except KeyboardInterrupt:
        print("Stopping early...")
        
        # If training mode, save a checkpoint
        if c.train: 
            if 'checkpointer' in locals() and 'config' in locals() and 'pop' in locals():
                try:
                    checkpointer.save_checkpoint(config, pop.population, pop.species, pop.generation)
                except Exception as e:
                    print(f"[ERROR] Could not save checkpoint: {e}")

        if p.isConnected():
            p.disconnect()

    except Exception as e:
        traceback.print_exc()

        # Save a checkpoint
        if c.train: 
            if 'checkpointer' in locals() and 'config' in locals() and 'pop' in locals():
                try:
                    checkpointer.save_checkpoint(config, pop.population, pop.species, pop.generation)
                except Exception as e:
                    print(f"[ERROR] Could not save checkpoint: {e}")

        if p.isConnected():
            p.disconnect()

if __name__ == '__main__':
    
    # Parse args
    parser = argparse.ArgumentParser(description='Run robot\'s controller using NEAT.')
    parser.add_argument('-r',   '--experiment_runs',        action='store',         help='Number of experiment runs.')
    parser.add_argument('-d',   '--debug',                  action='store_true',    help='Show debug messages.')
    parser.add_argument('-f',   '--experiment_file',        action='store',         help='Experiment description file.')
    parser.add_argument('-v',   '--visualize',              action='store_true',    help='Visualize the best controller and plot the results.')
    parser.add_argument('-fs',  '--second_experiment_file', action='store',         help='Second Experiment description file.')
    parser.add_argument('-gui', '--gui_activated',          action='store_true',    help='Evolve the robot with GUI on.')
    parser.add_argument('-e',   '--evolution_type',         action='store_true',    help='Type of evolution: true for offline and false for online.')
    parser.add_argument('-rec', '--record',                 action='store_true',    help='Record the simulation.')
    parser.add_argument('-vf',  '--video_filename',         action='store',         help='Video filename.')
    parser.add_argument('-ch',  '--checkpoint',             action='store_true',    help='Use a checkpoint population to continue the evolution.')
    parser.add_argument('-cf',  '--checkpoint_file',        action='store',         help='Checkpoint file to be used.')
    parser.add_argument('-cp',  '--checkpoint_population',  action='store',         help='Population size to be use in the checkpoint')
    parser.add_argument('-s',   '--seed',                   action='store',         help='Seed for the random number generator.')
    parser.add_argument('-dy',  '--dynamic_environment',    action='store_true',    help='Use a dynamic environment for the simulation.')
    parser.add_argument('-ng', '--num_gens',                action='store',         help='Number of generations to run the NEAT algorithm')

    args = parser.parse_args() 
    
    main(args) # Run the main function with the parsed arguments 