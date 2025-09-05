import json
import numpy as np
import os
import constants as c
import math

class Experiment:
    def load_conditions_file(filename): 
        # Check if directory exists
        dir_name = os.path.dirname(filename)
        if dir_name and not os.path.exists(dir_name): # if the directory does not exist, create it
            os.makedirs(dir_name)

        # If file does not exist, create a JSON file
        if not os.path.exists(filename):
            data = {
                "name": "Single run",
                "controls": {},
                "conditions": [
                    {
                        "Run": {
                            "None": "None"
                        }
                    }
                ]
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)

        # load the file
        with open(filename) as f:
            json_data = json.load(f)

        # Get the data from the file
        controls = json_data["controls"]
        conditions = json_data["conditions"]
        name = json_data["name"]

        return name, controls, conditions

    def load_results(self, results):
        # Loads results from a previously saved experiment
        for run in range(self.num_runs):
            self.fitness_results[run][:self.start_gen] = results["fitness_results"][run][:self.start_gen]
            self.diversity_results[run][:self.start_gen] = results["diversity_results"][run][:self.start_gen]
            self.individuals_skipped_over_time[run][:self.start_gen] = results["individuals_skipped_over_time"][run][:self.start_gen]
            self.simulation_length[run][:self.start_gen] = results["simulation_length"][run][:self.start_gen]
            self.species_results[run][:self.start_gen] = results["species_results"][run][:self.start_gen]
            self.nodes_results[run][:self.start_gen] = results["nodes_results"][run][:self.start_gen]
            self.connections_results[run][:self.start_gen] = results["connections_results"][run][:self.start_gen]
            self.ind_length_over_time[run][:self.start_gen] = results["ind_length_over_time"][run][:self.start_gen]
            self.fitness_racing_history[run][:self.start_gen] = results["fitness_racing_history"][run][:self.start_gen]
            self.bad_orientations_gen[run] = results["bad_orientations_gen"][run]

    def __init__(self, condition, controls, args, num_runs=1) -> None: 
        self.name = list(condition.keys())[0] 
        self.controls = controls
        self.condition = condition[self.name]
        self.num_runs = num_runs
        self.individuals_skipped_over_time = np.zeros((num_runs, c.num_gens))
        self.simulation_length = np.zeros((num_runs, c.num_gens))
        self.fitness_results = np.zeros((num_runs, c.num_gens))
        self.diversity_results = np.zeros((num_runs, c.num_gens))
        self.species_results =   np.zeros((num_runs, c.num_gens))
        self.nodes_results = np.zeros((num_runs, c.num_gens))
        self.connections_results = np.zeros((num_runs, c.num_gens))
        self.ind_length_over_time = np.zeros((num_runs, c.num_gens)) 
        self.fitness_racing_history = np.zeros((num_runs, c.num_gens))
        self.bad_orientations_gen = []
        self.gens_to_find_solution = [math.inf] * num_runs
        self.current_run = 0 
        self.start_gen = 0 # used to track the generation number when resuming an experiment
        self.args = args.__dict__ if args is not None else {}
    
    def setup_arrays(self, results=None): 
        self.start_gen = c.num_gens_old
        total_gens = c.num_gens + self.start_gen 
        
        self.fitness_results = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.diversity_results = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.individuals_skipped_over_time = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.simulation_length = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.species_results = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.nodes_results = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.connections_results = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.ind_length_over_time = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.fitness_racing_history = [np.zeros(total_gens) for _ in range(self.num_runs)]
        self.bad_orientations_gen = [[] for _ in range(self.num_runs)]
        
        self.current_run = c.run
        
        if results:
            self.load_results(results)
        
    def record_results(self, individuals_skipped_over_time, simulation_length,fitness_over_time, diversity_over_time, species_over_time, nodes_over_time, connections_over_time, ind_length_over_time, fitness_racing_history, bad_orientations_gen): 
        end_gen = self.start_gen + len(fitness_over_time) # - 1
        
        self.bad_orientations_gen.append(bad_orientations_gen)
        
        self.individuals_skipped_over_time[self.current_run][self.start_gen:end_gen] = individuals_skipped_over_time
        self.simulation_length[self.current_run][self.start_gen:end_gen] = simulation_length
        self.fitness_results[self.current_run][self.start_gen:end_gen] = fitness_over_time
        self.diversity_results[self.current_run][self.start_gen:end_gen] = diversity_over_time
        self.species_results[self.current_run][self.start_gen:end_gen] = species_over_time
        self.nodes_results[self.current_run][self.start_gen:end_gen] = nodes_over_time
        self.connections_results[self.current_run][self.start_gen:end_gen] = connections_over_time
        self.ind_length_over_time[self.current_run][self.start_gen:end_gen] = ind_length_over_time
        self.fitness_racing_history[self.current_run][self.start_gen:end_gen] = fitness_racing_history
        
        self.current_run+=1
    
    def generate_empty_results_dictionary(self): 
        self.results = {}
        self.results["name"] = self.name
        self.results["condition"] = self.condition
        self.results["num_runs"] = self.num_runs
        self.results["fitness_results"] = []
        self.results["diversity_results"] = []
        self.results["individuals_skipped_over_time"] = []
        self.results["simulation_length"] = []
        self.results["species_results"] =[]
        self.results["nodes_results"] = []
        self.results["connections_results"] = []
        self.results["ind_length_over_time"] = []
        self.results["fitness_racing_history"] = [] 
        self.results["bad_orientations_gen"] = []
        self.results["args"] = self.args
        self.results["brain"] = {"fitness": 0}
        
    def generate_results_dictionary(self): 
        self.results = {}
        self.results["name"] = self.name
        self.results["condition"] = self.condition
        self.results["num_runs"] = self.num_runs
        self.results["bad_orientations_gen"] = self.bad_orientations_gen
        self.results["fitness_results"] = [arr.tolist() for arr in self.fitness_results]
        self.results["diversity_results"] = [arr.tolist() for arr in self.diversity_results]
        self.results["individuals_skipped_over_time"] = [arr.tolist() for arr in self.individuals_skipped_over_time]
        self.results["simulation_length"] = [arr.tolist() for arr in self.simulation_length]
        self.results["species_results"] = [arr.tolist() for arr in self.species_results]
        self.results["nodes_results"] = [arr.tolist() for arr in self.nodes_results]
        self.results["connections_results"] = [arr.tolist() for arr in self.connections_results]
        self.results["ind_length_over_time"] = [arr.tolist() for arr in self.ind_length_over_time]
        self.results["fitness_racing_history"] = [arr.tolist() for arr in self.fitness_racing_history]
        self.results["args"] = self.args
        self.results["brain"] = {"fitness": max(max(arr) for arr in self.fitness_results)}
            
    def apply_condition(self): 
        for k,v in self.controls.items():
            print("\t Control:", k, "->", v)
            c.apply_condition(k, v)
            if k == "num_runs":
                self.num_runs = v
            
        for k, v in self.condition.items():
            if k is not None:
                print(f"\t\tapply {k}->{v}")
                c.apply_condition(k, v)
                if k == "simulation_duration":
                    value = int(v / c.delta_t)
                    c.apply_condition("simulation_length_default", value)
                    c.apply_condition("simulation_length", value)