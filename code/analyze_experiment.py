import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from util import plot_mean_over_time, plot_mean_std_over_x

def get_experiments_with_condition(data, condition):
    # Get all experiments with a specific condition #
    experiments_with_condition = []

    for d in data: # for each experiment get the list of conditions
        cond = d.get("condition", {})  # or {} if missing
    
        # Check if the experiment has the condition that we want to get
        if cond.get(condition, False) == True:
            experiments_with_condition.append(d)

    return experiments_with_condition

def flatten_list(lst):
    # Flattens a list of lists into a single list #
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat

def main(args):
    
    # Gets the experiment file
    filename = ""
    if args.experiment_file:
        filename = args.experiment_file
    else:
        raise Exception("No experiment file specified")
    with open(filename) as f:
        data = json.load(f)
        
    plt.rc('font', size=20) # Set plot text size

    # Debug mode #
    if args.analyze_fitness_simulation_length: 
        if args.second_experiment_file: # Subplot the trend of the two experiment files
            with open(args.second_experiment_file) as f:
                data2 = json.load(f)

            _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            results = [data, data2]
            titles = ["Distance Traveled", "Average Speed"]
            ylabels = ["Distance (m)", "Speed (m/s)"]

            for i in range(2):
                all_eval_lengths = []
                all_fitnesses = []

                for run in results[i]:
                    sim_length_list = run.get("simulation_length", [])
                    fitness_list = run.get("fitness_results", [])

                    for s, f in zip(sim_length_list, fitness_list):
                        all_eval_lengths.extend(s)
                        all_fitnesses.extend(f)

                plot_mean_std_over_x(all_eval_lengths, all_fitnesses, titles[i], axs[i], ylabels[i])

            #plt.suptitle("Simulation Length vs Fitness")
            plt.tight_layout()
            plt.show()
        
        else: # Plots the trend of the single experiment file
            all_eval_lengths = []
            all_fitnesses    = []

            # Collect data from all runs/generations in flat lists
            for run in data:
                sim_length_list = run.get("simulation_length", [])
                fitness_list    = run.get("fitness_results", [])

                for s, f in zip(sim_length_list, fitness_list):
                    all_eval_lengths.append(s)
                    all_fitnesses.append(f)

            fig, ax = plt.subplots(figsize=(8, 6))
            plot_mean_std_over_x(all_eval_lengths, all_fitnesses, "Fitness vs Simulation Length", ax)
            plt.tight_layout()
            plt.show()
            
    # Visualization mode #
    else:
        # # Diversity 
        # plot_mean_over_time([np.array(c["diversity_results"]) for c in data], [np.array(
        #     c["diversity_results"]) for c in data], [c["name"] for c in data], "Generations", "Average diversity", title="Diversity")
        # Species
        plot_mean_over_time([np.array(c["species_results"]) for c in data], [np.array(
            c["species_results"]) for c in data], [c["name"] for c in data], "Generations", "N Species", title="Species")
        # # Nodes
        # plot_mean_over_time([np.array(c["nodes_results"]) for c in data], [np.array(c["nodes_results"]) for c in data], [
        #                                         np.array(c["name"]) for c in data], "Generations", "Number of Nodes", title="Number of Nodes")
        # # Connections
        # plot_mean_over_time([np.array(c["connections_results"]) for c in data], [np.array(
        #     c["connections_results"]) for c in data], [c["name"] for c in data], "Generations", "Number of Connections", title="Number of Connections")
        # Fitness
        plot_mean_over_time([np.array(c["fitness_results"]) for c in data], [np.array(
            c["fitness_results"]) for c in data], [c["name"] for c in data], "Generations", f"Best fit individual (m/s)", title="Best fitness Values")
        
        # plot_mean_over_time([np.array(c["fitness_results"]) for c in data], [np.array(
        #     c["fitness_results"]) for c in data], [c["name"] for c in data], "Generations", f"Best fitness (m/s)", title="Fitness Values", second_dataset=[np.array(c["bad_orientations_gen"]) for c in data])
        
        # Racing
        data_racing = get_experiments_with_condition(data, "racing")
        if len(data_racing) > 0:
            plot_mean_over_time([np.array(c["individuals_skipped_over_time"]) for c in data_racing], [np.array(
                    c["individuals_skipped_over_time"]) for c in data_racing], [c["name"] for c in data_racing], "Generations", f"Num Individuals", title="Number of Individuals Skipped by Racing")
            # Assessment Length
            plot_mean_over_time([np.array(c["ind_length_over_time"]) for c in data_racing], [np.array(
                    c["ind_length_over_time"]) for c in data_racing], [c["name"] for c in data_racing], "Generations", f"Duration (seconds)", title="Average Evaluation time through Generations")
            # Fitness Values of the skipped individuals
            plot_mean_over_time([np.array(c["fitness_racing_history"]) for c in data_racing], [np.array(
                    c["fitness_racing_history"]) for c in data_racing], [c["name"] for c in data_racing], "Generations", f"Average Speed (m/s)", title="Fitness Values of Skipped Individuals")
        
        # Simulation Length
        data_parameter_control = get_experiments_with_condition(data, "parameter_control")
        if len(data_parameter_control) > 0:
            plot_mean_over_time([np.array(c["simulation_length"]) for c in data_parameter_control], [np.array(
                    c["simulation_length"]) for c in data_parameter_control], [c["name"] for c in data_parameter_control], "Generations", f"Duration (seconds)", title="Evaluation time through Generations")


        # Evolution Time 
        experiment_names = []
        # all_run_times = []
        all_assessment_length = []
        all_last_gen_fitness = []

        for i, exp in enumerate(data):
            experiment_names.append(exp.get("name", f"Exp {i}"))
            
            # run_times = [float(rt) for rt in exp["run_time"] if rt != float('inf')]
            assessment_lengths = [rt for rt in flatten_list(exp["simulation_length"]) if rt != float('inf')]

            # all_run_times.append(run_times)
            all_assessment_length.append(assessment_lengths)
            
            last_gen_fitness = []
            for run in exp["fitness_results"]:
                last_gen_fitness.append(run[-1])  
            all_last_gen_fitness.append(last_gen_fitness)

        # Boxplots #
        
        # # Evolution Time 
        # plt.figure(figsize=(8,6))
        # plt.boxplot(all_run_times, patch_artist=True)

        # plt.xlabel("Experiment")
        # plt.ylabel("Run Time (seconds)")
        # plt.title("Experiments Run Time")
        # plt.xticks(ticks=range(1, len(experiment_names)+1), labels=experiment_names, rotation=90, fontsize=8)
        # plt.tight_layout()

        # Assessment Length            
        plt.figure(figsize=(8,6))
        plt.boxplot(all_assessment_length, patch_artist=True)
        
        plt.xlabel("Experiment")
        plt.ylabel("Length (seconds)")
        plt.title("Assessment Length")
        plt.xticks(ticks=range(1, len(experiment_names)+1), labels=experiment_names, rotation=90, fontsize=8)
        plt.tight_layout()
        
        # Last Generation Fitness
        plt.figure(figsize=(8, 6))
        plt.boxplot(all_last_gen_fitness, patch_artist=True)
        plt.title("Last Generation Best Fitness Values")
        plt.xlabel("Experiment")
        plt.ylabel("Average Speed (m/s)")
        plt.xticks(ticks=range(1, len(experiment_names)+1), labels=experiment_names, rotation=90, fontsize=8)
        plt.tight_layout()
        
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run search on the robot controller.')
    parser.add_argument('-f', '--experiment_file',                      action='store',         help='Experiment results file.',        required=True)
    parser.add_argument('-a', '--analyze_fitness_simulation_length',    action='store_true',    help='Analyze the correlation between fitness and simulation length.')
    parser.add_argument('-fs', '--second_experiment_file',              action='store',         help='Second experiment results file.', required=False)
    args = parser.parse_args()
    
    # Run the main function with the parsed arguments
    main(args)