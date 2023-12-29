from typing import Any
import ActivationFunctions
import random

class Individual:
    def __init__(self, id, genotype, fitness, shape=None, connectionWeightDomain=(-1,1), biasDomain=(-1,1), ADMISSIBLE_ACTIVATION_FUNCTIONS=[ActivationFunctions.logistics, ActivationFunctions.sinusoid, ActivationFunctions.relu, ActivationFunctions.tanh, ActivationFunctions.linear]):
        self.id = id
        if genotype is not None:
            self.genotype = genotype
        else:
            self.genotype = Genotype(None, None, shape=shape, connectionWeightDomain=connectionWeightDomain, ADMISSIBLE_ACTIVATION_FUNCTIONS=ADMISSIBLE_ACTIVATION_FUNCTIONS)
        self.fitness = fitness
        self.connectionWeightDomain = connectionWeightDomain
        self.biasDomain = biasDomain
        self.admissibleActivationFunctions = ADMISSIBLE_ACTIVATION_FUNCTIONS
        self.adjusted_fitness = fitness
        
    def copy(self):        
        return Individual(self.id, self.genotype.copy(), self.fitness, shape= None, connectionWeightDomain=self.connectionWeightDomain, biasDomain=self.biasDomain)
    
    def interpret(self, input, debug = False, reset = True):
        output = []
        if reset:
            for g in self.genotype.node_genes:
                g.value = 0
                g.activation_value = 0
        for g in self.genotype.node_genes:
            val = g.bias #Important to keep g.value untouched due to self connections
            if g.type == 'input':
                val += input[g.index]

            if debug:
                print('node: ', g.index, 'bias: ', g.bias, 'value: ', val)
            for c in self.genotype.connection_genes:
                if c.out_node == g.index and c.enabled:
                    try:
                        val += c.weight * self.genotype.node_genes[c.in_node].activation_value
                    except:
                        ##Something went wrong, remove connection
                        self.genotype.connection_genes.remove(c)
                    if debug:
                        print('\tconnection: ', c.in_node, '->', c.out_node, 'weight: ', c.weight, 'value: ', self.genotype.node_genes[c.in_node].activation_value, 'result: ', c.weight * self.genotype.node_genes[c.in_node].activation_value, 'total: ', val)
            g.value = val
            g.activation_value = g.squash(g.value)
            if debug:
                print('\tsquash: ', g.activation_value)
            if g.type == 'output':
                output.append(g.activation_value)


        return output
        
    def addRandomConnection(self, INOVATION_NUMBER=0, inovation_archive={}):
        #get all possible connections
        possible_connections = []
        for i in range(len(self.genotype.node_genes)):
            for j in range(len(self.genotype.node_genes)):
                possible_connections.append((i,j))
        #remove connections that already exist
        for c in self.genotype.connection_genes:
            try:
                possible_connections.remove((c.in_node, c.out_node))
            except:
                pass
        #add a random connection
        if len(possible_connections)>0:
            c = random.choice(possible_connections)
            ino, INOVATION_NUMBER = self.getInnovationNumber(inovation_archive, INOVATION_NUMBER, c, mode='connections')
            self.genotype.connection_genes.append(ConnectionGene(ino, c[0], c[1], self.genotype.uniformWeight(self.connectionWeightDomain), -1, True))
            
        else:
           pass
           #print('No possible connections to add')
        return INOVATION_NUMBER
        
    def reEnableRandomConnection(self, INOVATION_NUMBER=0, inovation_archive={}):
        cands = [c for c in self.genotype.connection_genes if not c.enabled]
        if len(cands)>0:
            random.choice(cands).enabled = True
        return INOVATION_NUMBER
    
    def disableRandomConnection(self, INOVATION_NUMBER=0, inovation_archive={}):
        ##disable only
        c = random.choice(self.genotype.connection_genes)
        c.enabled = False
        return INOVATION_NUMBER
    
    def removeRandomConnection(self, INOVATION_NUMBER=0, inovation_archive={}):
        if len(self.genotype.connection_genes)>1:
            #really remove
            c = random.choice(self.genotype.connection_genes)
            self.genotype.connection_genes.remove(c)
        else:
            pass
            #print('No connections to remove')
        return INOVATION_NUMBER

    def changeActivationFunction(self, INOVATION_NUMBER=0, inovation_archive={}):
        c = random.choice(self.genotype.node_genes)
        c.activation_function = random.choice(self.admissibleActivationFunctions)
        return INOVATION_NUMBER
    
    def changeConnectionWeight(self, INOVATION_NUMBER=0, inovation_archive={}):
        #print(self.genotype)
        c = random.choice(self.genotype.connection_genes)
        c.weight += random.gauss(0,0.1*(self.connectionWeightDomain[1]-self.connectionWeightDomain[0]))
        #c.weight = self.genotype.uniformWeight(self.connectionWeightDomain)
        return INOVATION_NUMBER

    def changeBias(self, INOVATION_NUMBER=0, inovation_archive={}):
        #print(self.genotype)
        c = random.choice(self.genotype.node_genes)
        c.bias += random.gauss(0,0.1*(self.biasDomain[1]-self.biasDomain[0]))
        return INOVATION_NUMBER

    def getInnovationNumber(self, inovation_archive, INOVATION_NUMBER, c, mode='nodes'):
        if c not in inovation_archive[mode]:
            ino = INOVATION_NUMBER
            inovation_archive[mode][c] = INOVATION_NUMBER
            INOVATION_NUMBER+=1

        else:
            ino = inovation_archive[mode][c]
        return ino, INOVATION_NUMBER

    def randomMutation(self, INOVATION_NUMBER, inovation_archive, ADMISSIBLE_MUTATIONS):
        ADMISSIBLE_MUTATIONS = eval(ADMISSIBLE_MUTATIONS)
        #return random.choice(ADMISSIBLE_MUTATIONS)(INOVATION_NUMBER=INOVATION_NUMBER, inovation_archive=inovation_archive) #uniform probabilites
        r = random.random()
        s = 0.0
        for m, p in ADMISSIBLE_MUTATIONS:
            s+=p
            #print(m, p, s, r)    
            if r<=s:
                return m(INOVATION_NUMBER=INOVATION_NUMBER, inovation_archive=inovation_archive) #weighted probabilites
        
    def addNodeMutation(self, admissibleFunctions=[ActivationFunctions.logistics, ActivationFunctions.sinusoid, ActivationFunctions.relu, ActivationFunctions.tanh, ActivationFunctions.linear],INOVATION_NUMBER=0, inovation_archive={}):
        c = random.choice(self.genotype.connection_genes)
        self.genotype.connection_genes.remove(c)
        
        node_index =  len(self.genotype.node_genes)
        
        ino, INOVATION_NUMBER = self.getInnovationNumber(inovation_archive, INOVATION_NUMBER, (c.in_node, c.out_node), mode='nodes')
        self.genotype.node_genes.append(NodeGene(ino,node_index, 'hidden', self.genotype.uniformWeight(self.biasDomain), random.choice(admissibleFunctions)))
        
        ino, INOVATION_NUMBER = self.getInnovationNumber(inovation_archive, INOVATION_NUMBER, (c.in_node, node_index), mode='connections')
        #self.genotype.connection_genes.append(ConnectionGene(ino, c.in_node, node_index, 1, -1))# this is like in the original paper, but it does not work well. The idea was to minimise the impact of the new node on the network
        self.genotype.connection_genes.append(ConnectionGene(ino, c.in_node, node_index, self.genotype.uniformWeight(self.connectionWeightDomain), -1))# this line is more disruptive, but it works better
        
        ino, INOVATION_NUMBER = self.getInnovationNumber(inovation_archive, INOVATION_NUMBER, (node_index, c.out_node), mode='connections')
        self.genotype.connection_genes.append(ConnectionGene(ino, node_index, c.out_node, c.weight, -1))
        
        return INOVATION_NUMBER
        

class Genotype:
    def __init__(self, connection_genes, node_genes, shape=None, connectionWeightDomain=(-1,1), biasDomain=(-1,1), ADMISSIBLE_ACTIVATION_FUNCTIONS=[ActivationFunctions.logistics, ActivationFunctions.sinusoid, ActivationFunctions.relu, ActivationFunctions.tanh, ActivationFunctions.linear]):
        #by default, create a fully connected network with the given shape
        if connection_genes is not None and node_genes is not None:
            self.connection_genes = connection_genes
            self.node_genes=node_genes
            
        else :
            self.connection_genes = []
            self.node_genes = []
            if shape is not None:
                node_index = 0
                historical_innovation_number = 0
                for l in range(len(shape)):
                    for i in range(shape[l]):
                        if l==0:
                            typ = 'input'
                            afunc = ActivationFunctions.linear
                            #afunc = ActivationFunctions.modified_sigmoid
                        elif(l==len(shape)-1):
                            typ = 'output'
                            #afunc = ActivationFunctions.relu
                            #afunc = ActivationFunctions.modified_sigmoid
                            afunc = random.choice(ADMISSIBLE_ACTIVATION_FUNCTIONS)
                        else:
                            typ = 'hidden'
                            #afunc = ActivationFunctions.relu
                            #afunc = ActivationFunctions.modified_sigmoid
                            afunc = random.choice(ADMISSIBLE_ACTIVATION_FUNCTIONS)
                            
                        self.node_genes.append(NodeGene(historical_innovation_number, node_index, typ, self.uniformWeight(biasDomain), afunc))
                        node_index+=1
                        historical_innovation_number+=1
                        
                        
                    if l>0:
                        for i in range(len(self.node_genes)-shape[l], len(self.node_genes)):
                            for j in range(len(self.node_genes)-shape[l]-shape[l-1], len(self.node_genes)-shape[l]):
                                self.connection_genes.append(ConnectionGene(historical_innovation_number, j, i, self.uniformWeight(connectionWeightDomain), -1))
                                historical_innovation_number+=1

    def uniformWeight(self,connectionWeightDomain):
        return random.random()*(connectionWeightDomain[1]-connectionWeightDomain[0])+connectionWeightDomain[0]

    def size(self):
        return len(self.node_genes)+len(self.connection_genes)
    
    def copy(self):
        connection_genes = [g.copy() for g in self.connection_genes]
        node_genes = [g.copy() for g in self.node_genes]
        return Genotype(connection_genes, node_genes)

    def __str__(self) -> str:
        s = '-'*5+'Genotype'+'-'*5
        s+='\n'+'Node genes: '
        for c in self.node_genes:
            s+='\n\t'+c.__str__()

        s+='\n'+'Connection genes: '
        for c in self.connection_genes:
            s+='\n\t'+c.__str__()
        s+='\n'+'-'*20+'\n'
        return s

            
class Gene:
    def __init__(self, history_id):
        self.historical_innovation_number = history_id

    def __str__(self) -> str:
        return 'Gene ' + str(self.historical_innovation_number)
    
    def copy(self):
        return self.__class__(self.historical_innovation_number)
    
class NodeGene(Gene):
    def __init__(self, history_id, index, type, bias, squash):
        super().__init__(history_id)
        self.index = index
        self.type = type
        self.bias = bias
        self.squash = squash
        self.value = 0
        self.activation_value = 0
    
    def copy(self):
        return NodeGene(self.historical_innovation_number, self.index, self.type, self.bias, self.squash)
    
    def __str__(self) -> str:
        return 'Node '+super().__str__() + ' index:' + str(self.index) + ' type:' + self.type + ' bias:' + str(self.bias) + ' squash:' + str(self.squash)+ ' value:' + str(self.value)

class ConnectionGene(Gene):
    def __init__(self, history_id, in_node, out_node, weight, gater, enabled=True):
        super().__init__(history_id)
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.gater = gater
        self.enabled= enabled

    def copy(self):
        return ConnectionGene(self.historical_innovation_number, self.in_node, self.out_node, self.weight, self.gater)


    def __str__(self) -> str:
        return 'Connection '+super().__str__() + ' in_node:' + str(self.in_node) + ' out_node:' + str(self.out_node) + ' weight:' + str(self.weight) + ' gater:' + str(self.gater)
if __name__ == '__main__':
    ind = Individual(0, None, 0, [2,3,1])
    print(ind.genotype)
    print(ind.interpret([1,2]))
    