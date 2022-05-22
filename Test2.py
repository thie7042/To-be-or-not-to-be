import multiprocessing
import os
import pickle
import random
#import cart_pole
import myneat

import numpy as np

from myneat import visualize



runs_per_net = 2 #500
#simulation_seconds = 60.0

def setup():
    setup.target = [0,1,0,1,0,0,0,0,1,0,1]

    setup.chars = [0,1]



setup()

# Lets create the phrase class

class phrase():
    def __init__(self):
        # Create a random binary array
        self.genes = [None] * len(setup.target)
        for i in range(len(setup.target)):
            self.genes[i] = random.choice(setup.chars)

class document_results():
    def __init__(self):
        # Create a random binary array
        self.best_fitness = 0
        self.solution = 0

    def update_data(self,fitness,solution):
        self.best_fitness = fitness
        self.solution = solution

    def print_results(self):
        print("Best recorded solution: ", self.solution)
        print("Best recorded fitness: " , self.best_fitness)
library = document_results()


# Evaluate one network a few times per generation
def eval_genome(genome, config):

    net = myneat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    phrases = []


    # For each 'run' that we are going to execute on each net
    for runs in range(runs_per_net):
        # Canidate performers
        phrases.append(phrase())

        #sim = cart_pole.CartPole()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0

        #done = False
        #while not done:
        # Lets allow our network to make 10 changes at most
        for change in range(11):
            # Here we are running our trials
            # Lets pull the current state of the phrase

            observation = phrases[runs].genes

            # action = output from neural network based on inputs
            # Output is either 0 or 1 (2 output nodes)
            output = net.activate(observation)

            for k in range(len(output)):
                if output[k] > 0.5:
                    # We change this digit
                    if phrases[runs].genes[k] == 0:
                        phrases[runs].genes[k] = 1
                    else:
                       phrases[runs].genes[k] = 0
                    #phrases[runs].genes[k] = random.choice(setup.chars)

           # action = np.argmax(net.activate(observation))

            fitness = 0
            for k in range(len(phrases[runs].genes)):

                if phrases[runs].genes[k] == setup.target[k]:
                    #print("MATCH!")
                    fitness += 1


            # Store the best solution we find
            if fitness > library.best_fitness:
                print("New Best Solution")
                print("Fitness: " + str(fitness))
                print("lib fit: " + str(library.best_fitness))

                library.best_fitness = fitness
                library.solution = phrases[runs].genes

                # Write Solution and Fitness
                t_file = open("Best_Solution.txt", 'w')
                t_file.write(str(phrases[runs].genes) + ","+ str(library.best_fitness))
                t_file.close()




            # Update
            # Reward = number of correct digits
            #observation, reward = .....(action)

            #fitness = reward
        # When solutions terminate, append fitnesses
        fitnesses.append(fitness)
    #print("Average Fitness of run = ",np.mean(fitnesses))


    # The genome's fitness is its average performance across all runs.
    return np.mean(fitnesses)
    #return np.max(fitnesses)




# Define the fitness of each network within the population
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = myneat.Config(myneat.DefaultGenome, myneat.DefaultReproduction,
                         myneat.DefaultSpeciesSet, myneat.DefaultStagnation,
                         config_path)

    pop = myneat.Population(config)
    stats = myneat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(myneat.StdOutReporter(True))

    pe = myneat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)


    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)
    print("________")



    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                      filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    run()