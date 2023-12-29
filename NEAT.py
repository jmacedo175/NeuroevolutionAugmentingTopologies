import NN
import ActivationFunctions
import random
import numpy as np


def rankParents(A,B, maximisation=True, adjusted_fitness=True):
    if adjusted_fitness:
        if maximisation:
            return (A,B) if A.adjusted_fitness > B.adjusted_fitness else (B,A)
        else:
            return (A,B) if A.adjusted_fitness < B.adjusted_fitness else (B,A)
        
    else:
        if maximisation:
            return (A,B) if A.fitness > B.fitness else (B,A)
        else:
            return (A,B) if A.fitness < B.fitness else (B,A)

'''
def best(A,B, maximisation=True):
    if maximisation:
        return A if A.fitness > B.fitness else B
    else:
        return A if A.fitness < B.fitness else B
    
    

def crossover(A,B, maximisation=True):
    ##made up crossover (works, but does not follow the original NEAT paper)
    bestParent = best(A,B, maximisation)
    size = bestParent.genotype.size()

    input_genesA = [g for g in A.genotype.node_genes if g.type == 'input']
    input_genesB = [g for g in B.genotype.node_genes if g.type == 'input']
    output_genesA = [g for g in A.genotype.node_genes if g.type == 'output']
    output_genesB = [g for g in B.genotype.node_genes if g.type == 'output']
    hidden_genesA = [g for g in A.genotype.node_genes if g.type == 'hidden']
    hidden_genesB = [g for g in B.genotype.node_genes if g.type == 'hidden']
    
    genes_off = []
    connection_genesOff =  []
    for i in range(len(input_genesA)):
        if random.random()<0.5:
            genes_off.append(input_genesA[i])
            connection_genesOff += [c for c in A.genotype.connection_genes if c.in_node == input_genesA[i].index]
        else:
            genes_off.append(input_genesB[i])
            connection_genesOff += [c for c in B.genotype.connection_genes if c.in_node == input_genesB[i].index]
    for i in range(len(output_genesA)):
        if random.random()<0.5:
            genes_off.append(output_genesA[i])
            connection_genesOff += [c for c in A.genotype.connection_genes if c.in_node == output_genesA[i].index]
        else:
            genes_off.append(output_genesB[i])
            connection_genesOff += [c for c in B.genotype.connection_genes if c.in_node == output_genesB[i].index]
    for i in range(size-len(input_genesA)-len(output_genesA)):
        if i<len(hidden_genesA) and i<len(hidden_genesB):
            if random.random()<0.5:
                genes_off.append(hidden_genesA[i])
                connection_genesOff += [c for c in A.genotype.connection_genes if c.in_node == hidden_genesA[i].index]
            else:
                genes_off.append(hidden_genesB[i])
                connection_genesOff += [c for c in B.genotype.connection_genes if c.in_node == hidden_genesB[i].index]
        elif i<len(hidden_genesA):
            genes_off.append(hidden_genesA[i])
            connection_genesOff += [c for c in A.genotype.connection_genes if c.in_node == hidden_genesA[i].index]
        elif i<len(hidden_genesB):
            genes_off.append(hidden_genesB[i])
            connection_genesOff += [c for c in B.genotype.connection_genes if c.in_node == hidden_genesB[i].index]

    return NN.Individual(0, NN.Genotype(connection_genesOff,genes_off), 0)
    '''
    
def crossover(A,B, maximisation=True, fitness_sharing=True):
    ##crossover as per the original NEAT paper
    best, worst = rankParents(A,B, maximisation, adjusted_fitness=fitness_sharing)
    
    nodeGenes_off = [g for g in best.genotype.node_genes]
    connectionGenes_off = [g for g in best.genotype.connection_genes]
    
    inds_nodes_worst = [g.historical_innovation_number for g in worst.genotype.node_genes]
    nodes_worst = [g for g in worst.genotype.node_genes]
    inds_connections_worst = [g.historical_innovation_number for g in worst.genotype.connection_genes]
    connections_worst = [g for g in worst.genotype.connection_genes]

    for i in range(len(nodeGenes_off)):
        if nodeGenes_off[i].historical_innovation_number in inds_nodes_worst and random.random()<0.5:
            nodeGenes_off[i] = nodes_worst[inds_nodes_worst.index(nodeGenes_off[i].historical_innovation_number)]
        nodeGenes_off[i] = nodeGenes_off[i].copy()

    for i in range(len(connectionGenes_off)):
        if connectionGenes_off[i].historical_innovation_number in inds_connections_worst and random.random()<0.5:
            connectionGenes_off[i] = connections_worst[inds_connections_worst.index(connectionGenes_off[i].historical_innovation_number)]
        connectionGenes_off[i] = connectionGenes_off[i].copy()

    return NN.Individual(0, NN.Genotype(connectionGenes_off,nodeGenes_off), 0)

def compute_similarity(A,B, c1, c2, c3):
    matching_node_genes, non_matching_node_genes = 0, 0
    matching_connection_genes, non_matching_connection_genes = 0, 0
    weight_difference = 0

    nodeGenesA = [g.historical_innovation_number for g in A.genotype.node_genes]
    nodeGenesB = [g.historical_innovation_number for g in B.genotype.node_genes]
    for n in nodeGenesA:
        if n in nodeGenesB:
            matching_node_genes += 1
        else:
            non_matching_node_genes += 1
    non_matching_node_genes += len(nodeGenesB) - matching_node_genes
            
    connectionGenesA = [g.historical_innovation_number for g in A.genotype.connection_genes]
    connectionWeightsA = [g.weight for g in A.genotype.connection_genes]
    connectionGenesB = [g.historical_innovation_number for g in B.genotype.connection_genes]
    connectionWeightsB = [g.weight for g in B.genotype.connection_genes]

    for n in connectionGenesA:
        if n in connectionGenesB:
            matching_connection_genes += 1
            weight_difference += abs(connectionWeightsA[connectionGenesA.index(n)] - connectionWeightsB[connectionGenesB.index(n)])
        else:
            non_matching_connection_genes += 1
    non_matching_connection_genes += len(connectionGenesB) - matching_connection_genes
    
    N = max(len(nodeGenesA)+len(connectionGenesA), len(nodeGenesB)+ len(connectionGenesB))
    return c1*(non_matching_node_genes + non_matching_connection_genes)/N + c3 * weight_difference
            
def adjust_fitness(pop, c1, c2, c3, thresh, alpha, maximisation=True):
    dists = np.zeros((len(pop), len(pop)))  
    for i in range(len(pop)):
        for j in range(i+1, len(pop)):
            dists[i,j] = dists[j,i] = compute_similarity(pop[i], pop[j], c1, c2, c3)
    for i in range(len(pop)):
        neighs = 0
        for j in range(len(dists)):
            if(dists[i][j] <= thresh):
                neighs+=1-(dists[i][j]/thresh)**alpha
        if maximisation:
            pop[i].adjusted_fitness = pop[i].fitness/neighs
        else:
            pop[i].adjusted_fitness = pop[i].fitness*neighs


def MSE(A,B):
    return sum((A-B)**2 )/len(A)
    #return sum([(a-b)**2 for a,b in zip(A,B)])/len(A)

def evaluate(ind, X, Y, debug=False):
    preds = [ind.interpret(x) for x in X]
    preds = np.array(preds)
    if debug:
        print('preds',preds, 'target', Y)
    ind.adjusted_fitness = ind.fitness = MSE(preds, Y)
    if debug:
        print('fitness', ind.adjusted_fitness)
    return ind.adjusted_fitness

def tournament(pop, size = 2, maximisation=True, fitness_sharing=True):
    cands = random.choices(pop, k=size)
    best = cands[0]
    for i in range(1,size):
        best = rankParents(best, cands[i], maximisation=maximisation, adjusted_fitness=fitness_sharing)[0]
    return best

def sort(pop, maximisation=True):
    return sorted(pop, key=lambda x: x.adjusted_fitness, reverse=maximisation)


def problem(name):
    if name=='XOR':
        X,Y = [[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]]
        X, Y, eval_function, maximisation = np.array(X), np.array(Y), evaluate, False
    return X,Y, eval_function, maximisation

if __name__ == '__main__':
    random.seed(0)
    INOVATION_NUMBER = 0
    ADMISSIBLE_ACTIVATION_FUNCTIONS = [ActivationFunctions.logistics, ActivationFunctions.sinusoid, ActivationFunctions.relu, ActivationFunctions.tanh, ActivationFunctions.linear, ActivationFunctions.leaky_relu, ActivationFunctions.binary_step,  ActivationFunctions.arctan, ActivationFunctions.elu, ActivationFunctions.softplus]
    #ADMISSIBLE_ACTIVATION_FUNCTIONS = [ActivationFunctions.logistics, ActivationFunctions.sinusoid, ActivationFunctions.relu, ActivationFunctions.tanh, ActivationFunctions.linear]#, ActivationFunctions.leaky_relu, ActivationFunctions.binary_step,  ActivationFunctions.arctan, ActivationFunctions.elu, ActivationFunctions.softplus]
    #ADMISSIBLE_MUTATIONS = '[(self.addRandomConnection,0.15), (self.disableRandomConnection,0.05), (self.changeActivationFunction,0.15), (self.addNodeMutation,0.2),(self.changeConnectionWeight,0.2), (self.reEnableRandomConnection,0.05), (self.changeBias,0.2)]'
    ADMISSIBLE_MUTATIONS = '[(self.addRandomConnection,0.1), (self.disableRandomConnection,0.1), (self.changeActivationFunction,0.1), (self.addNodeMutation,0.1),(self.changeConnectionWeight,0.25), (self.reEnableRandomConnection,0.1), (self.changeBias,0.25)]'
    #ADMISSIBLE_MUTATIONS = '[(self.addRandomConnection,0.15), (self.disableRandomConnection,0.1), (self.addNodeMutation,0.1),(self.changeConnectionWeight,0.275), (self.reEnableRandomConnection,0.1), (self.changeBias,0.275)]'
    
    overall_best = None

    pop_size = 1000
    generations = 1000
    p_mut=0.3
    p_cross=0.7
    survival_selection = 'elitist'
    elite_size = 2
    stagnation = 250
    fitness_sharing = False
    c1, c2, c3, thresh, alpha = 1, 1, 0.4, 3, 1 ##original parameters as per the NEAT paper
        
    connectionWeightDomain = (-1,1)
    biasDomain = (-1,1)
    
    X, Y, fitness_function, maximisation = problem('XOR')
    hidden_layers_shape= []
    
    shape = [len(X[0])] + hidden_layers_shape + [len(Y[0])]
    
    inovation_archive = {'nodes':{}, 'connections':{}}
    
    pop = [NN.Individual(i, None, 0, shape, connectionWeightDomain=connectionWeightDomain, biasDomain=biasDomain, ADMISSIBLE_ACTIVATION_FUNCTIONS=ADMISSIBLE_ACTIVATION_FUNCTIONS) for i in range(pop_size)]
    INOVATION_NUMBER = max(pop[0].genotype.connection_genes[-1].historical_innovation_number, pop[0].genotype.node_genes[-1].historical_innovation_number)

    for ind in pop:
        for c in ind.genotype.connection_genes:
            inovation_archive['connections'][(c.in_node, c.out_node)] = c.historical_innovation_number

    for i in range(pop_size):
        fitness_function(pop[i], X, Y)
        
        pop[i].creation = -1
        if overall_best is None or rankParents(overall_best, pop[i], maximisation, adjusted_fitness=False)[0] == pop[i]:
            overall_best = pop[i]

    if fitness_sharing:
        adjust_fitness(pop, c1, c2, c3, thresh, alpha, maximisation=maximisation)
    else:
        for ind in pop:
            ind.adjusted_fitness = ind.fitness
    
    for gen in range(generations):
        offspring = []
        for i in range(pop_size):
            A, B = tournament(pop, maximisation=maximisation), tournament(pop, maximisation=maximisation)
            crossed = False
            if random.random()<p_cross:
                offspring.append(crossover(A,B, maximisation=maximisation,fitness_sharing=fitness_sharing))
                crossed=True                
            else:
                offspring.append(A.copy())
                
            if random.random()<p_mut or not crossed:
                INOVATION_NUMBER = offspring[-1].randomMutation(INOVATION_NUMBER, inovation_archive, ADMISSIBLE_MUTATIONS)
            
            fitness_function(offspring[-1], X, Y)
            offspring[-1].creation = gen
    
        
        if survival_selection=='elitist':
            if fitness_sharing:
                elite = sort(pop, maximisation=maximisation)[:elite_size] 
                new_pop = pop + offspring
                adjust_fitness(new_pop, c1, c2, c3, thresh, alpha, maximisation=maximisation)
                new_pop = new_pop[len(pop):]
                new_pop = sort(new_pop, maximisation=maximisation)
                
                pop = elite + new_pop[:-elite_size]
                adjust_fitness(pop, c1, c2, c3, thresh, alpha, maximisation=maximisation)
            else:
                offspring = sort(offspring, maximisation=maximisation)
                pop = pop[:elite_size]+ offspring[:-elite_size]
                    
            pop = sort(pop, maximisation=maximisation)[:pop_size]
        elif survival_selection=='merge':    
            pop = sort(pop+offspring, maximisation=maximisation)[:pop_size]
        elif survival_selection=='generational':
            pop = offspring
        
        for i in range(len(pop)):
            if rankParents(overall_best, pop[i], maximisation, adjusted_fitness=False)[0] == pop[i]:
                overall_best = pop[i]
        print('Generation',gen,'best adjusted fitness:', pop[0].adjusted_fitness, 'original fitness', pop[0].fitness, 'creation',pop[0].creation, 'overall best fitness:', overall_best.fitness, overall_best.creation)
        if gen-overall_best.creation >= stagnation:
            print('Stagnation')
            break
    print('Last pop best adjusted fitness:', pop[0].adjusted_fitness, 'original fitness', pop[0].fitness, 'creation', pop[0].creation)
    print(pop[0].genotype)
    print(fitness_function(pop[0], X, Y, debug=True))   
    
    print('Overal best adjusted fitness:', overall_best.adjusted_fitness, 'original fitness', overall_best.fitness, 'creation', overall_best.creation)
    print(overall_best.genotype)
    print(fitness_function(overall_best, X, Y, debug=True))   
    