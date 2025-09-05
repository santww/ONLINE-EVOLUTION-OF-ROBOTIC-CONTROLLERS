import numpy as np
import neat
import constants as c
import pickle
import os

class FitnessHistoryReporter(neat.reporting.BaseReporter):
    def __init__(self):
        super().__init__()
        self.generation = 0
        self.best_fitness_history = []
        self.best_genomes = []
        self.avg_nodes_history = []
        self.avg_connections_history = []
        self.sim_length_history = []
        self.decrease_count = 0
        self.last_gen_best_fitness = -float('inf')
        self.skip_counts_over_time = []
        self.species_counts_over_time = []
        self.diversity_over_time = []
        self.ind_length_over_time = []
        self.fitness_racing_history = []
        self.bad_orientations_gen = [] # Track the generations where the robot is not oriented correctly
        self.avg_fitness_history = []  # Track average fitness per generation

        self.current_gen_skip_count = 0
        self.best_genome = None
    def start_generation(self, generation=None, species=None):
        # Reset skip count when a new generation begins
        self.current_gen_skip_count = 0

        if generation is None: generation = self.generation + 1
        self.generation = generation
        
        # Species mapping
        if species is not None:
            self.genome_to_species = {}  
            for sid, s in species.species.items():
                for gid in s.members:
                    self.genome_to_species[gid] = sid

    def register_generation(self, num, lengths, fitnesses, num_bad_orientations):
        # Register information about the racing mechanism
        self.current_gen_skip_count = num
        self.ind_length_over_time.append(np.mean(lengths))
        self.fitness_racing_history.append(np.mean(fitnesses))
        if num_bad_orientations > 0: self.bad_orientations_gen.append(self.generation)
    
    def post_evaluate(self, config, population, species, best_genome):        
        # print(self.generation, " - ", len(population))
        
        # Sort the population by fitness
        sorted_genomes = sorted(population.values(), key=lambda g: g.fitness if g.fitness is not None else -float('inf'), reverse=True)
        
        # Get all the fitnesses of the population
        fitnesses = [genome.fitness for genome in population.values()]  # ordenar para ter o melhor fitness primeiro

        # Compute the best fitness
        best_fitness = sorted_genomes[0].fitness
        #best_fitness = best_genome.fitness
        
        # Get the best genome
        self.best_genome = sorted_genomes[0]
        
        # Compute the number of species
        num_species = len(species.species) 

        # Compute the number of nodes and connections in the population
        all_num_nodes = []
        all_num_connections = []
        for g in population.values():
            all_num_nodes.append(len(g.nodes) - len(config.genome_config.output_keys))
            all_num_connections.append(len(g.connections))
        
        # Compute diversity (number of unique genomes)
        fitnesses = [gen.fitness for gen in population.values() if gen.fitness is not None]
        if len(fitnesses) > 1: diversity = np.std(fitnesses)
        else: diversity = 0.0
        
        # Compute the average fitness of the population
        avg_fitness = np.mean(fitnesses)
                
        # Store to history
        self.best_fitness_history.append(best_fitness)
        self.best_genomes.append(self.best_genome)
        self.sim_length_history.append((c.recoveryMechanism*c.recovery_length*c.delta_t)+(c.simulation_length*c.delta_t))
        self.skip_counts_over_time.append(self.current_gen_skip_count)
        self.species_counts_over_time.append(num_species)
        self.avg_nodes_history.append(np.mean(all_num_nodes))
        self.avg_connections_history.append(np.mean(all_num_connections))
        self.diversity_over_time.append(diversity)
        self.current_gen_skip_count = 0 # Reset for next generation
        self.avg_fitness_history.append(avg_fitness)

        # Applies the parameter control
        if c.parameter_control and len(self.avg_fitness_history) >= 3:
            # Get the last three generations' average fitness
            avg_fits = self.avg_fitness_history[-3:]
            
            # If the average fitness is increasing
            if avg_fits[0] < avg_fits[1] < avg_fits[2]:
                self.adapt_evaluation_time(trend='increasing')
            elif avg_fits[0] > avg_fits[1] > avg_fits[2]: # Average fitness is decreasing
                self.adapt_evaluation_time(trend='decreasing')
            
        self.save_genome()
    
    def adapt_evaluation_time(self, trend=None):
        # Adapts the simulation length based on the average fitness of the generations #

        # There is no need to adapt the simulation length if the population is not improving
        if self.generation == 0: return

        prev_length = c.simulation_length 
        
        if trend == 'increasing': # If the average fitness is increasing, we want to make the task harder by decreasing the simulation length
            adjustment = 1.0 - np.random.uniform(c.hrule_alpha_min, c.hrule_alpha_max) * 0.1
            c.simulation_length = int(c.simulation_length * adjustment)
        elif trend == 'decreasing': # If the average fitness is decreasing, we want to make the task easier by increasing the simulation length
            adjustment = 1.0 + np.random.uniform(c.hrule_alpha_min, c.hrule_alpha_max) * 0.05
            c.simulation_length = int(c.simulation_length * adjustment)

        # Safeguard: Prevents the simulation to not decrease the simulation length dramatic
        if c.simulation_length < prev_length: self.decrease_count += 1
        else: self.decrease_count = 0

        if self.decrease_count >= 6:
            c.simulation_length = int(prev_length * 1.01)
            self.decrease_count = 0
            
        # Clamp the simulation length 
        c.simulation_length = np.clip(c.simulation_length, c.min_eval_time, c.max_eval_time)
        
    def save_genome(self):
        # Save the best genome of the current generation #
        
        # Directory path
        if c.run is None:
            dir_path = os.path.join("genomes", str(c.evolution), str(c.experiment_name))
        else:
            dir_path = os.path.join("genomes", str(c.evolution), str(c.experiment_name), str(c.run))
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        file_path = os.path.join(dir_path, f"gen_{self.generation}.pkl")
        
        # Save the genome
        with open(file_path, 'wb') as f:
            pickle.dump(self.best_genome, f)