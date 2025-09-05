# University of Coimbra - Informatics Department - Master's in Data Science and Engineering

Master's thesis implementation - "Online Evolution of Robotic Controllers"

# Steps

## Setup

Install requirements:
`pip install -r requirements.txt`

## Implementation

Run the NEAT for a single run:
`python main.py`

Run the NEAT for a specific configuration:
`python main.py -f experiments\experiments_file.json`

For more options:
`python main.py --help`

## Visualization

Visualize the results of an experiment:
`python main.py -v -f <results_file.json>`.

## Credits

This project uses the [NEAT](https://github.com/CodeReclaimers/neat-python) library and was inspired by [Jackson Dean](https://github.com/jacksonsdean)'s [work](https://github.com/jacksonsdean/evolutionary-robotics).

## Example of training

`python main.py -f experiments\experiments.json -r 5`
