"""
epidemic_helper.py: Package for epidemic simulation. 

COMMENTS

There is still plenty of room for improvement here. The obvious one would be to have SIR and SIRV inherit from an abstract class. Moreover I could divide the classes into separte packages and make a nicer graphical interface. 

In terms off efficiency some lists could be replaced by arrays & some array and CSR matrices should be replaced by lil matricses. THis should speed up some of the computations.

The optimizer class is in its current pratice quite overkill for the work it's doing. I implemented it anyways so it would be easier to experiment with different learning rates. 

Author: Axel Sjöberg

"""

# Packages used
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import networkx as nx
import matplotlib.pyplot as plt
import operator
import itertools
import random as rand


class SIR(object):
    """
    Attributes
    G: nx.Graph 
    number_of_nodes: int
    beta: float
    rho: float
    inf_time: np.ndarray
    rec_time: np.ndarray
    W: Sparce Matrix
    infected_at_t: list
    """
    
    def __init__(self, G_loc, beta, rho,print_out=True):
        
        """"Arguments:

        G : networkx.Graph()
                Graph over which the epidemic propagates
        beta : float
            infection rate (must be positive)
        rho : float
            recovery rate (must be positive)
        print_out : bool (default: True)
            prints the status during thew simulation
        
        """
        if not isinstance(G_loc, nx.Graph):
            raise ValueError('Invalid graph type, should be networkx.Graph')
        if not list(G_loc.nodes()) == sorted(G_loc.nodes()):
            raise ValueError('NODES NOT IN ORDER')
        
        # Save the graph object
        self.G = G_loc
        
        # Set the number of nodes
        self.number_nodes = len(G_loc.nodes())
        
        # Infection rate
        if beta < 0:
            raise ValueError('Error, must be negative beta')
        self.beta = beta
        
        # Recovery rate
        if rho < 0:
            raise ValueError('Error, must be negative rho')
        self.rho = rho
        
        # Vectors for infection time and recovery time. 
        self.inf_time = np.full(self.number_nodes, np.inf) #np.array(np.inf)*self.number_nodes
        self.rec_time = np.full(self.number_nodes, np.inf) #np.array(np.inf)*self.number_nodes
        
        #Adjecency Matrix 
        self.W = nx.to_scipy_sparse_matrix(G_loc)
        
        # Indicating the number of infected neighboors
        self.inf_neigh = csr_matrix((len(G_loc.nodes()),1), dtype=np.int8)
        
        # The number of infected at time t
        self.infected_at_t = []
        
        # Creating the status printer
        self.print_out = print_out
    
    # Checks the node status for node_id at time
    def check_node_status(self, node_id, time):
        if self.inf_time[node_id] > time:
            return 0
        elif self.rec_time[node_id] > time:
            return 1 
        else:
            return 2
    
    # Returns the infection time vector. (Takes value np.inf if never infected)
    def get_inf_time(self):
        return self.inf_time
    
    # Returns the recovery time vector. (Takes value np.inf if never recovered)
    def get_rec_time(self):
        return self.rec_time
    
    # Returns the number of new infections at time t
    def get_new_infected_at_t(self):
        return self.infected_at_t
    
    # Returns the total number of infected at time t 
    def get_total_infected_at_t(self,time):
        return len(list(filter(lambda n: self.check_node_status(n,time) == 1, range(self.number_nodes))))
        
    # Returns the total number of recovered at time t 
    def get_total_recovered_at_t(self,time):
        return len(list(filter(lambda n: self.check_node_status(n,time) == 2, range(self.number_nodes))))
        
    # Returns the total number of susceptible at time t 
    def get_total_susceptible_at_t(self,time):
        return len(list(filter(lambda n: self.check_node_status(n,time) == 0, range(self.number_nodes))))

        
    # Get the current infected     
    def get_current_infected(self):
        a = lil_matrix(self.inf_time, dtype=np.int8).nonzero()[1]
        b = lil_matrix(self.rec_time, dtype=np.int8).nonzero()[1]
        cols = np.setdiff1d(a,b)
        rows=np.zeros((len(cols)),dtype=np.int8)
        data = np.ones((len(cols)),dtype=float)
        return csr_matrix((data, (cols, rows)), shape=(self.number_nodes, 1))

    # Get the probability of infection for node_id
    def probability_of_infection(self,node_id):
        if self.rec_time[node_id] < np.inf:
            return 0
        elif self.inf_time[node_id] < np.inf:
            return 0
        else:
            return 1- np.power((1-self.beta),self.inf_neigh[node_id][0].toarray()[0][0])

    # Get the probability of recovery for node_id
    def probability_of_recovery(self,node_id):
        if self.rec_time[node_id] < np.inf:
            return 0
        elif self.inf_time[node_id] < np.inf:
            return self.rho
        else:
            return 0
    
    # Set changes 
    def set_change(self, transitions, current_time, rec=True):
        if rec == True:
            for i in np.where(transitions)[0]:
                self.rec_time[i] = current_time
        else:
            for i in np.where(transitions)[0]:
                self.inf_time[i] = current_time

    # Status printer
    def printer(self,current_time,max_time, end=0):
        temp_inf = np.round(100 * (np.count_nonzero(self.inf_time < np.inf) / self.number_nodes),4)
        temp_rec = np.round(100 * (np.count_nonzero(self.rec_time < np.inf) / self.number_nodes),4)
        #temp_suc = np.round(100 - temp_inf  - temp_rec,4)
        temp_suc = np.round(100 * (self.get_total_susceptible_at_t(current_time) / self.number_nodes),4)
        infected = 'infected: ' + str(temp_inf) + ' %'
        recovered = 'recovered: ' + str(temp_rec) + ' %'
        susceptible = 'susceptible: ' + str(temp_suc) + ' %'
        if self.print_out is True:
 
            if end == 1:
                print('Epidemic terminated in week ' + str(current_time) + ' | ' + susceptible + ' | ' + ' | ' + infected + ' | ' + recovered, end="\r", flush=True)

            elif end == 2:
                print('Epidemic still progressing in week ' + str(current_time) + ' | ' + susceptible + ' | ' + ' | ' + infected + ' | ' + recovered, end="\r", flush=True)

            else:
                print('Week ' + str(current_time) + ' out of ' + str(max_time) + ' | ' + susceptible + ' | ' + ' | ' + infected + ' | ' + recovered,end="\r", flush=True)


    # Process of one time unit in the epidemic progression
    def _process(self,current_time):
        I = self.get_current_infected() # Gets the currently infected
        self.inf_neigh = np.dot(self.W,I) # Gets the number of infected neighbours for each node
        
        #Gets the probabilty of infection for every node and picks the nodes to infect given these probabilities
        infection_probability = np.fromiter(map(self.probability_of_infection,range(0,self.number_nodes)),dtype=np.float)
        new_infected = np.fromiter(map(lambda x: x > rand.random(),infection_probability),dtype=np.bool)
        
        self.infected_at_t.append(np.sum(new_infected))# Records how many people were infected in this time step. 
        
        #Gets the probabilty of recovery for every node and picks the nodes to recover given these probabilities      
        recovery_probability = np.fromiter(map(self.probability_of_recovery,range(0,self.number_nodes)),dtype=np.float)
        new_recovered = np.fromiter(map(lambda x: x > rand.random(),recovery_probability),dtype=np.bool)

        self.set_change(new_infected,current_time,rec=False) # Infects the selected nodes
        self.set_change(new_recovered,current_time) # "recovers" the selected nodes

    # Set the infection of the source nodes
    def set_children(self,source,current_time):
        for i in source:
            self.inf_time[i] = current_time
    
    # Checks if the epidemic has terminated
    def check_infection_stopped(self):
        return np.count_nonzero(self.inf_time < np.inf) == np.count_nonzero(self.rec_time < np.inf)
    
    # Starts the simulation
    def simulate_epidemic(self, source, max_time = 100):
        current_time = 1
        self.set_children(source,current_time) # Sets the sorce of infection to infected in the network
        while current_time < max_time:
            self._process(current_time) #Run one process, i.e run one week.
            current_time = current_time + 1
            
            # Check if the infection has reached a dead state, no need to continue the simulation -> BREAK
            if self.check_infection_stopped() is True:
                self.printer(current_time, max_time,end=1)
                break
            
            # Print the progression of the epidemic
            if self.print_out == True:
                self.printer(current_time, max_time)
        
        # Print the end information of the epidemic 
        if self.check_infection_stopped() is False:
            self.printer(current_time, max_time, end =2)




class SIRV(object):
    """
    Attributes
    G: nx.Graph 
    number_of_nodes: int
    beta: float
    rho: float
    inf_time: np.ndarray
    rec_time: np.ndarray
    W: Sparce Matrix
    infected_at_t: list
    vaccinated_at_t: list
    V:list
    """
   

    def __init__(self, G_loc, beta, rho, V, print_out=True):
        
        """"Arguments:

        G : networkx.Graph()
            Graph over which the epidemic propagates
        beta : float
            infection rate (must be positive)
        rho : float
            recovery rate (must be positive)
        print_out : bool (default: True)
            prints the status during thew simulation
        V: list
            list over the number infected each week. 
        """
        
        if not isinstance(G_loc, nx.Graph):
            raise ValueError('Invalid graph type, should be networkx.Graph')
        
        if not list(G_loc.nodes()) == sorted(G_loc.nodes()):
            raise ValueError('NODES NOT IN ORDER')
        
        #Save the graph object
        self.G = G_loc
        
        # Set the number of nodes
        self.number_nodes = len(G_loc.nodes())
        # Infection rate
        if beta < 0:
            raise ValueError('Error, must be negative beta')
        self.beta = beta
        
        # Recovery rate
        if rho < 0:
            raise ValueError('Error, must be negative rho')
        self.rho = rho
        
        # Vectors for infection time, recovery time and vacination time
        self.inf_time = np.full(self.number_nodes, np.inf) #np.array(np.inf)*self.number_nodes
        self.rec_time = np.full(self.number_nodes, np.inf) #np.array(np.inf)*self.number_nodes
        self.vac_time = np.full(self.number_nodes, np.inf)
        
        #Adjecency Matrix 
        self.W = nx.to_scipy_sparse_matrix(G_loc)
        self.inf_neigh = csr_matrix((len(G_loc.nodes()),1), dtype=np.int8)
        
        # Indicating the number of infected neighboors
        self.infected_at_t = []
        
        # Indicating the number of vacianted neighboors
        self.vaccinated_at_t = []
        
        # Creating the status printer
        self.print_out = print_out
        
        # Sets V as the number of new vaccinated (in %) per week
        self.V = [V[0]] + list(np.round(np.diff(V),3))
    
    # Checks the node status for node_id at time
    def check_node_status(self, node_id, time):
        if self.inf_time[node_id] < time:
            return 3  
        elif self.inf_time[node_id] > time:
            return 0
        elif self.rec_time[node_id] > time:
            return 1 
        else:
            return 2
    
    # Checks the node status, ignoring the vaccination for node_id at time
    def check_node_status2(self, node_id, time):
        if self.inf_time[node_id] > time:
            return 0
        elif self.rec_time[node_id] > time:
            return 1 
        else:
            return 2
    
    # Returns the infection time vector. (Takes value np.inf if never infected)
    def get_inf_time(self):
        return self.inf_time
    
    # Returns the recovery time vector. (Takes value np.inf if never recovered)
    def get_rec_time(self):
        return self.rec_time
    
    # Returns the number of new infections at time t
    def get_new_infected_at_t(self):
        return self.infected_at_t
    
    # Returns the number of new vaccinations at time t
    def get_new_vaccinated_at_t(self):
        return self.vaccinated_at_t
    
    # Returns the total number of infected at time t 
    def get_total_infected_at_t(self,time):
        return len(list(filter(lambda n: self.check_node_status2(n,time) == 1, range(self.number_nodes))))
    
    # Returns the total number of recovered at time t 
    def get_total_recovered_at_t(self,time):
        return len(list(filter(lambda n: self.check_node_status2(n,time) == 2, range(self.number_nodes))))
    
    # Returns the total number of vaccinated at time t 
    def get_total_vaccinated_at_t(self,time):
        return len(list(filter(lambda n: self.check_node_status(n,time) == 3, range(self.number_nodes))))
    
    # Returns the total number of susceptible at time t 
    def get_total_susceptible_at_t(self,time):
        return len(list(filter(lambda n: self.check_node_status(n,time) == 0, range(self.number_nodes))))

        
    # Get the current infected       
    def get_current_infected(self):
        a = lil_matrix(self.inf_time, dtype=np.int8).nonzero()[1]
        b = lil_matrix(self.rec_time, dtype=np.int8).nonzero()[1]
        c = lil_matrix(self.vac_time, dtype=np.int8).nonzero()[1]
        cols = np.setdiff1d(np.setdiff1d(a,b),c)
        rows=np.zeros((len(cols)),dtype=np.int8)
        data = np.ones((len(cols)),dtype=float)
        #print(self.number_nodes)
        return csr_matrix((data, (cols, rows)), shape=(self.number_nodes, 1))

    # Get the probability of infection for node_id
    def probability_of_infection(self,node_id):
        if self.vac_time[node_id] < np.inf:
            return 0 
        if self.rec_time[node_id] < np.inf:
            return 0
        elif self.inf_time[node_id] < np.inf:
            return 0
        else:
            return 1- np.power((1-self.beta),self.inf_neigh[node_id][0].toarray()[0][0])

    # Get the probability of recovery for node_id
    def probability_of_recovery(self,node_id):
        if self.rec_time[node_id] < np.inf:
            return 0
        elif self.inf_time[node_id] < np.inf:
            return self.rho
        else:
            return 0
    
    # Vaccination process
    def vaccination(self,current_time):
        number_of_vaccinations = int(self.V[current_time] * self.number_nodes) # Gets the NUMBER of people to be infected
        non_vaccinated = np.where(self.vac_time < np.inf)[0] # Checks which nodes are yet to be vaccinated
        # For the people not vaccinated yet, select n random nodes at with a uniform distribution and set these as vaccinated
        vaccination_candidates = [i for i in range(self.number_nodes) if i not in non_vaccinated]
        vaccinated_individuals = np.random.choice(vaccination_candidates,number_of_vaccinations)
        for i in vaccinated_individuals:
            self.vac_time[i] = current_time
        self.vaccinated_at_t.append(number_of_vaccinations) #Reccord the number of vaccianted at this time step 
        # (Note that the last step could of course be done when creating the object,
        # but if we want vaccination to a random process this is needed.
    
 
    # Set changes 
    def set_change(self, transitions, current_time, rec=True):
        if rec == True:
            for i in np.where(transitions)[0]:
                self.rec_time[i] = current_time
        else:
            for i in np.where(transitions)[0]:
                self.inf_time[i] = current_time
    
    # Status printer
    def printer(self,current_time,max_time, end=0):
        temp_inf = np.round(100 * (np.count_nonzero(self.inf_time < np.inf) / self.number_nodes),4)
        temp_rec = np.round(100 * (np.count_nonzero(self.rec_time < np.inf) / self.number_nodes),4)
        #temp_suc = np.round(100 - temp_inf  - temp_rec,4)
        temp_suc = np.round(100 * (self.get_total_susceptible_at_t(current_time) / self.number_nodes),4)
        infected = 'infected: ' + str(temp_inf) + ' %'
        recovered = 'recovered: ' + str(temp_rec) + ' %'
        susceptible = 'susceptible: ' + str(temp_suc) + ' %'
        if self.print_out is True:
 
            if end == 1:
                print('Epidemic terminated in week ' + str(current_time) + ' | ' + susceptible + ' | ' + ' | ' + infected + ' | ' + recovered, end="\r", flush=True)

            elif end == 2:
                print('Epidemic still progressing in week ' + str(current_time) + ' | ' + susceptible + ' | ' + ' | ' + infected + ' | ' + recovered, end="\r", flush=True)

            else:
                print('Week ' + str(current_time) + ' out of ' + str(max_time) + ' | ' + susceptible + ' | ' + ' | ' + infected + ' | ' + recovered,end="\r", flush=True)


    # Process of one time unit in the epidemic progression
    def _process(self,current_time):
        self.vaccination(current_time) #Vaccination process
        I = self.get_current_infected() # Gets the currently infected
        self.inf_neigh = np.dot(self.W,I) # Gets the number of infected neighbours for each node
        
        #Gets the probabilty of infection for every node and picks the nodes to infect given these probabilities
        infection_probability = np.fromiter(map(self.probability_of_infection,range(0,self.number_nodes)),dtype=np.float) 
        new_infected = np.fromiter(map(lambda x: x > rand.random(),infection_probability),dtype=np.bool)
        
        self.infected_at_t.append(np.sum(new_infected)) # Records how many people were infected in this time step. 
        
        #Gets the probabilty of recovery for every node and picks the nodes to recover given these probabilities 
        recovery_probability = np.fromiter(map(self.probability_of_recovery,range(0,self.number_nodes)),dtype=np.float)
        new_recovered = np.fromiter(map(lambda x: x > rand.random(),recovery_probability),dtype=np.bool)
        
        self.set_change(new_infected,current_time,rec=False) # Infects the selected nodes
        self.set_change(new_recovered,current_time) # "recovers" the selected nodes

    # Set the infection of the source nodes
    def set_children(self,source,current_time):
        for i in source:
            self.inf_time[i] = current_time
    
    # Checks if the epidemic has terminated
    def check_infection_stopped(self):
        return np.count_nonzero(self.inf_time < np.inf) == np.count_nonzero(self.rec_time < np.inf)
    
    # Starts the simulation        
    def simulate_epidemic(self, source, max_time = 100):
        current_time = 1
        self.set_children(source,current_time) # Sets the sorce of infection to infected in the network
        while current_time < max_time:
            self._process(current_time) #Run one process, i.e run one week.
            current_time = current_time + 1 
            
            # Check if the infection has reached a dead state, no need to continue the simulation -> BREAK
            if self.check_infection_stopped() is True:
                self.printer(current_time, max_time,end=1)
                break
            
            # Print the progression of the epidemic
            if self.print_out == True:
                self.printer(current_time, max_time)
        
        # Print the end information of the epidemic 
        if self.check_infection_stopped() is False:
            self.printer(current_time, max_time, end =2)
            
            
            
""" Simulates multiple epidemics several times and return the:
        · Average susceptible each week
        · Average infected each week
        · Average recovered each week
        · Average New infections each week
        · New Vaccinations each week (This could be returned in a more effective method)

If V = 0 it's a normal SIR without a vaccine (Default setting)
otherwise V is the list containing the number of vaccinated each week"""
def multiple_epidemics(G,beta,rho,source_nodes=-1,time_limit=15,N=10,epidemic='sir',V=0,norm=True,rand_source_nbr=1):
    if epidemic == 'sir':
        d_inf_time = {}
        d_rec_time = {}
        d_suc_time = {}
        d_new_infected = {}
        
        d_new_vaccinated = 0
        if V != 0:
            d_new_vaccinated = {}
        
        for i in range(1,time_limit+1):
            d_inf_time[i] = []
            d_rec_time[i] = []
            d_suc_time[i] = []
            d_new_infected[i] = []
            
            if V != 0:
                d_new_vaccinated[i] = []
        
        for i in range(N):
            
            # SIR or SIV?
            epidemic = 0
            if V == 0:
                epidemic = SIR(G,beta,rho,print_out=False)
            else:
                epidemic = SIRV(G,beta,rho,V,print_out=False)
            
            if source_nodes==-1:
                source_nodes = np.random.choice(range(len(G.nodes())),rand_source_nbr).tolist()
            
            epidemic.simulate_epidemic(source = source_nodes, max_time = time_limit)
            
                                                
            # Get the total number of infected for each week 
            infection_dates = [0]*(time_limit+1)
            for i in range(1,1+time_limit):
                infection_dates[i] = epidemic.get_total_infected_at_t(i)
            
            # Get the total number of susceptible for each week 
            susceptible_dates = [0]*(time_limit+1)
            for i in range(1,1+time_limit):
                susceptible_dates[i] = epidemic.get_total_susceptible_at_t(i)

            # Get total number of recovered for each week    
            recovery_dates = [0]*(time_limit+1)
            for i in range(1,1+time_limit):
                recovery_dates[i] = epidemic.get_total_recovered_at_t(i)
            
            # Get total number of vaccinated for each week (if SIRV)
            new_vaccinated = 0
            if V != 0:
                temp = epidemic.get_new_vaccinated_at_t()
                new_vaccinated = temp +  list(np.zeros(time_limit-len(temp)))
            
            # Get the number of newly infected each week.
            temp = epidemic.get_new_infected_at_t()
            new_infected = np.concatenate((temp, np.zeros(time_limit-len(temp))))
            for k in range(1,1+time_limit):
                d_inf_time[k].append(infection_dates[k])
                d_rec_time[k].append(recovery_dates[k])
                d_suc_time[k].append(susceptible_dates[k])
                
                d_new_infected[k].append(new_infected[k-1])
                if V != 0:
                    d_new_vaccinated[k].append(new_vaccinated[k-1])
        
        # Averaging over the n simulations for our returns.
        average_susceptible = dict((k,np.sum(v)/N) for k,v in d_suc_time.items()) 
        average_infected = dict((k,np.sum(v)/N) for k,v in d_inf_time.items()) 
        average_recovered = dict((k,np.sum(v)/N) for k,v in d_rec_time.items()) 
        average_new_infections = dict((k,np.sum(v)/N) for k,v in d_new_infected.items()) 
        
        new_vaccinations = 0
        if V != 0:
            new_vaccinations = dict((k,np.sum(v)/N) for k,v in d_new_vaccinated.items()) 
        
        # Normalize to get in % the average number of susceptible, recovered, infected each week. 
        if norm == True:
            factor = 100/len(G.nodes())
            average_susceptible = dict((k,v*factor) for k,v in average_susceptible.items()) 
            average_infected = dict((k,v*factor) for k,v in average_infected.items()) 
            average_recovered = dict((k,v*factor) for k,v in average_recovered.items())          
        
        
        return average_susceptible, average_infected, average_recovered, average_new_infections, new_vaccinations
            
    elif epidemic == 'si':
        print('Not implemented')
    
    else:
        print('must enter correct epidemic type')
        
# Creates a graph according to the preferential attachment method with average degree k and n nodes 
def preferential_attachment_graph(k,n = 100):
    G_loc = nx.complete_graph(k+1)
    node_count = k+1
    
    c = int(k/2)
    
    odd_k = True
    if k%2 == 0:
        odd_k = False
    
    t = k+2
    while(t <= n):
        new_links = []
        if odd_k is True:
            new_links = node_selection(G_loc,c + t%2) 
        else:
            new_links = node_selection(G_loc,c) 
        new_links = list(zip([node_count] * len(list(new_links)), list(new_links)))
        G_loc.add_edges_from(new_links)
        node_count = node_count + 1
        t = t+1
    return G_loc

# Helper function for the preferential function
def node_selection(G_loc,c):
    factor = np.sum(list(dict(G_loc.degree()).values()))
    prob_dict = dict((k,v/factor) for k,v in dict(G_loc.degree()).items())
    return list(np.random.choice(list(prob_dict.keys()), c, p=list(prob_dict.values()),replace=False))
    
    
    
    
    
    
    
    
    
## TRAINING PART    

# Optimizer object
class Optimizer(object):
    
    def __init__(self, k_step=1, beta_step=0.02, rho_step=0.03):
        self.k_step = k_step
        self.beta_step = beta_step
        self.rho_step = rho_step
    
    # Updates the learning rates by setting them to half their previous value.
    def update(self):
        self.beta_step = self.beta_step/2
        self.rho_step = self.rho_step/2
    
    # sets the learning rate to 0 for k -> thus fixing k with its current value. 
    def stat_k(self):
        self.k_step = 0
    
    # Returns trhe current learning rates
    def get_steps(self):
        return self.k_step, self.beta_step, self.rho_step




def train(k0, beta0,rho0,V,I,number_of_nodes=500, optim=Optimizer(), source_nodes = -1, batch_size=10, max_iter = 10, graph_type ='pref',update=False):
    learning_rates = optim.get_steps() #Get our initial learning rates
    i = 1
    best_params = (k0,beta0,rho0)
    rmses = []
    print('Initiating the training', end="\r",flush=True)
    while i < max_iter:
        # Running 10 simulations for each combination over the paramter space, returns the average number of newly infeced for each week
        d, param_space = process(best_params, learning_rates, number_of_nodes,V,I,source_nodes,batch_size,graph_type) 
        loc_rmse, param_index = evaluate(d,I) # Evalutaing how the simulated values compare to the ground truth
        loc_params = param_space[param_index] #Saving the best paramters
        rmses.append(loc_rmse) # Storing the best RMSE of this iteration
        
        print('Training in progress. | Current iteration: ' + str(i) + '| Current RMSE: '+ str(loc_rmse) + ' | Current params ' + str(loc_params), end="\r",flush=True) # Printing the learning progression
        if loc_params == best_params:
            if update is False:
                break
                print('Break')
            # If we want to update the learning rates, default is False
            if update is True:
                optim.update() 
                learning_rates = optim.get_steps()
        best_params = loc_params
        i += 1
    print('Training finished')
    
    # Plotting the loss function
    plt.figure()
    plt.plot(rmses)
    plt.title('Loss Function')
    plt.ylabel('RMSE')
    plt.xlabel('iteration')
    plt.show()
    return best_params, rmses[-1]



# Fininding the average newly infected each week using the parameters in available parameter space
def process(paramters,learning_rates,number_of_nodes,V,I,source_nodes,batch_size,graph_type):
    # These are the parameters their corresponing learning rates.
    k0 = paramters[0]
    beta0 = paramters[1]
    rho0 = paramters[2]
    k_step = learning_rates[0]
    beta_step = learning_rates[1]
    rho_step = learning_rates[2]
    
    k = [k0-k_step, k0, k0+k_step]
    beta = [np.round(beta0-beta_step,4), beta0, np.round(beta0+beta_step,4)]
    rho = [np.round(rho0 - rho_step,4), rho0, np.round(rho0 + rho_step,4)]
    param_space = list(itertools.product(*[k,beta,rho])) # Create all the different possible combinations of the paramters
    
    
    d = {} # The dictionary where we store the number of newly infected for every iteration for every paramter set-up
    for i in range(len(param_space)):
        d[i] = []
    
    t = 0
    for i in param_space:
        for k in range(batch_size):
            # Create the Graph, small world or preferential attachment
            G = preferential_attachment_graph(i[0],number_of_nodes)
            if graph_type == 'small_world':
                G = nx.watts_strogatz_graph(number_of_nodes, 6, 5)
            
            # Create the SIRV obejct and runt he simulations
            sirv = SIRV(G, i[1], i[2] ,V,print_out=False)
            if source_nodes == -1:
                temp = np.random.choice(range(number_of_nodes),1,replace=False)
                sirv.simulate_epidemic(source = temp, max_time = 15)
            else:
                sirv.simulate_epidemic(source = source_nodes, max_time = 15)
            
            
            
            #Get the number of newly infected each week for this simulation
            new_infected = sirv.get_new_infected_at_t()
            # Putting zeors in the end if we have the case that the simulation was terminated early
            new_infected = new_infected + list(np.zeros(len(I)-len(new_infected),dtype=int))
            # Storring the number of newly infected in the dictionary
            d[t].append(new_infected)
        t += 1
    return dict((k,list(np.average(np.array(v), axis=0))) for k,v in d.items()), param_space 
    #Averages all the simulations beofre returning them together with the paramater space


# Evaluating which one of the parameters is the best.
def evaluate(d,targets):
    best_rmse = np.inf
    best_params = -1
    for k,v in d.items():
        temp_rmse = rmse(v, targets)
        if temp_rmse < best_rmse:
            best_rmse = temp_rmse
            best_params = k
    return best_rmse, best_params

# Returns the RMSE of two lists. 
def rmse(predictions, targets):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())
  