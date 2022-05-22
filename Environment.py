import myneat
import os
import random

from myneat import visualize

chars = 'abcdefghijklmnopqrstuvwxyz '
#print(len(chars))


def recording():
    recording.current_best_score = 0
    recording.best_phrase = ""
recording()

def setup():
    # Objective phrase
    #setup.target = "to be or not to be"
    setup.target = "40334567890"
    # Characters to choose from
    #setup.chars = 'abcdefghijklmnopqrstuvwxyz '
    setup.chars = '0123456789'
    # Size of population
    setup.populationsize = 150

    # Number of generations
    setup.generations = 300

setup()

class phrase():
    def __init__(self):
        self.fitness = 0

        # Create a random phrase
        self.genes = [None] * len(setup.target)
        for i in range(len(setup.target)):
            self.genes[i] = random.choice(setup.chars)










# Evaluate each solution within a generation
def eval_genomes(genomes,config):

    # A list to store all phrases in each generation
    phrases = []

    # A list to store all genes of each solution
    ge = []

    # A list to store all neural networks in each generation
    nets = []

    # Index is the index of the phrase
    index = 0
    for genome_id, genome in genomes:
        phrases.append(phrase())

        # Set the preliminary fitness
        genome.fitness = 0 #4.0

        # construct the neural network based on the genes within genome
        net = myneat.nn.FeedForwardNetwork.create(genome, config)

        # Store in list
        nets.append(net)

        # Store genomes in list
        ge.append(genome)

        #
        for i in range(len(phrases[index].genes)):
            # Calculate fitness
            if phrases[index].genes[i] == setup.target[i]:
                genome.fitness += 1

        if genome.fitness > recording.current_best_score :
            recording.best_phrase = phrases[index].genes

        index += 1


   # print(phrases)



    # We only need to calculate neural network outputs of each phrase for each generation (for this scheme)
    # Input/output controls
    for genome_id, genome in enumerate(phrases): #genomes:

        #print(genome.genes[0])
        #  Go through each digit and check if we are changing it or not
        for i in range(len(genome.genes)):
            output = nets[genome_id].activate((0,int(genome.genes[i])))

            if output[0] > 0.5 :
                # Randomly change the character
                genome.genes[i] = random.choice(setup.chars)

        # Let's collect the 'inputs'
        # Here we pass our neural network our inputs

        """for letter in len(genome.genes):
            output = net.activate(letter)
            genome.fitness
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2"""



def run(config_file):
    # Load configuration.
    config = myneat.Config(myneat.DefaultGenome, myneat.DefaultReproduction,
                         myneat.DefaultSpeciesSet, myneat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NoneEAT run.
    p = myneat.Population(config)


    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(myneat.StdOutReporter(True))
    stats = myneat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(myneat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = myneat.nn.FeedForwardNetwork.create(winner, config)

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    myneat.visualize.draw_net(config, winner, True, node_names=node_names)
    myneat.visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    myneat.visualize.plot_stats(stats, ylog=False, view=True)
    myneat.visualize.plot_species(stats, view=True)


    p = myneat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)