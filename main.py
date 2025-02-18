import json
from compare_clustering_solutions import evaluate_clustering


def analyze_unrecognized_requests(data_file, output_file, num_representatives, min_size):
    # todo: implement this function
    #  you are encouraged to split the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file

    pass


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])
