
import random



##############################
#       Project Set-up       #
##############################
import random





#######



mutation_rate = 0.01  # Mutates 1% of the time


def setup():
    setup.target = "to be or not to be"

    setup.chars = 'abcdefghijklmnopqrstuvwxyz '
    setup.populationsize = 150

    setup.popindex = 0

setup()

popindex = 0
# DNA stores the genetic information of members of a population
class DNA:
    # Initialize random genes
    def __init__(self):
        global popindex
        popindex += 1
        self.genes=[None]*len(setup.target)
        for i in range(len(setup.target)):
            self.genes[i] = random.choice(setup.chars)
        #print("Creating creature: ", popindex)

        #print(self.genes)

    # Calculate the fitness
    def fitness(self):
        self.score = 0

        # Check if each character in gene matches our target
        for i in range(len(self.genes)):
            if self.genes[i] == setup.target[i]:
                self.score += 1

        # Scale the score depending on the size of the target
        self.score = self.score / len(setup.target)

    def mutation(self, mutation_rate):

        # Check if mutating
        for i in range(len(self.genes)):
            if random.uniform(0, 1) < mutation_rate:
                # Perform some mutation
                self.genes[i] = random.choice(setup.chars)


# Create our initial population
population = [None]*setup.populationsize
for i in range(setup.populationsize):
    population[i] = DNA()


##############################
#         Functions          #
##############################

# Get each member of the population to calculate its fitness
"""def scoring():
    for i in range(len(population)):
        population[i].fitness()

scoring()"""


# Crossover of genes
def crossover(parentA, parentB):
    child = DNA()

    # Select a random cut point to split genes
    splitmid = random.randint(0, len(setup.target))

    # Take genes from both parents at random divide
    # This is not a great way to do it. "Coin flip' for each gene may be better, but also need to think about NEAT
    for i in range(len(setup.target)):
        if i < splitmid:
            child.genes[i] = parentA.genes[i]
        else:
            child.genes[i] = parentB.genes[i]

    return child




#testing = fitness("too")
#
#print(testing)
generations = 600

best = 0
index = 0
for i in range(generations):
    # Start of generation

    # Calculate fitness
    for j in range(len(population)):
        population[j].fitness()

        if population[j].score > best:
            current_best = population[j].genes
            best = population[j].score
            index = i

    # Print the best solution
    print("".join(current_best), ", Generation: ", i+1, "Fitness: ", best )

    # Build mating pool

    matingPool = []
    # Create mating pool (% chance of being selected as a parent)
    for i in range(setup.populationsize):

        n = population[i].score * 100
        for j in range(int(n)):
            matingPool.append(population[i]);


    # Reproduction
    for i in range(len(population)):
        A_index = random.randint(0, len(matingPool)-1)
        B_index = random.randint(0, len(matingPool)-1)
        # make sure that you have unique indexes
        while A_index == B_index:
            B_index = random.randint(0, len(matingPool))

        ParentA = matingPool[A_index]
        ParentB = matingPool[B_index]

        # Get genetic code for child
        child = crossover(ParentA, ParentB)
        # Mutate genes (random)
        child.mutation(mutation_rate)

        population[i] = child

print("___________________")
print("Generation of best solution: ", index)