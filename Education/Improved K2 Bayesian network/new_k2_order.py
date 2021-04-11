from queue import Queue
from Adjecency_Matrix import *
from math import log
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.estimators import  K2Score, BicScore
from graphviz import Digraph
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
#############################################################
import numpy as np
import pandas as pd
from numpy import genfromtxt
import collections
import csv
import matplotlib.pyplot as plt
import time
#Outputs graph edges to .gph file
from Adjecency_Matrix import AdjacencyMatrixGraph


def graph_out(dag,filename,mapping):
    with open(filename, 'w') as f:
        for i in range(np.size(dag[0])):
            for j in range(np.size(dag[0])):
                if(dag[i][j] == 1):
                    out_string = mapping[i] + ', ' + mapping[j] + '\n'
                    f.write(out_string)

def model_change(dag,data):
    bay_model=[]
    data = pd.DataFrame(data)
    print(data)
    for i in range(len(dag)):
        for j in range(np.size(data,1)):
            if dag[i][j] !=0:
                bay_model.append((str(i),str(j)))
    best_model = BayesianModel(bay_model)
    bic = BicScore(data)
    return bic.score(best_model)

    #Creates a dictionary of node index to category strings
def map_categories(categories):
    mapping = {}
    for i in range(len(categories)):
       # mapping[i] = categories[i][1:len(categories[i])-1]
        mapping[i]  = categories[i]
    return mapping

    #Given a target, find every instance index in array
def find(arr,target):
    array = np.array([],dtype='int64')
    for i in range(np.size(arr)):
        if(arr[i] == target):
            array = np.append(array,i)
    return array

#Ln gamma function ln((x-1)!) ->  ln(0) + ln(1) + ... + ln(x-1)
def ln_gamma(x):
    return sum(np.log(range(1,int(x))))

#Construct a data structure that stores the possible states (col) for each variable (row)
#Also returns a range vector that stores the number of states for each variable
def get_dim_range(_data, vec):
        count_n = 0
        d = np.size(vec[0,:])
        dim_length = np.zeros((1,d),dtype = 'int64')
        t = -1
        #Count number of states
        for q in range(d):
            temp_vec = np.unique(_data[:,vec[:,q]])
            x = temp_vec.reshape(1,np.size(temp_vec))
            temp_vec = x
            if(temp_vec[:,0] == -1):
                temp_vec = np.empty()
            range_n = np.size(temp_vec)
            dim_length[0,q] = range_n
            t += 1
            #Assign zeros to the end to create valid matrix dimensions.
            if(count_n == 0):
                count_n = range_n
                dim = np.zeros((d,count_n),dtype = 'int64')
                dim[t,:] = temp_vec
            elif(count_n >= range_n):
                #dim[t,:] = np.concatenate((temp_vec,np.zeros((1,count_n - range_n))),axis=1)
                dim = np.concatenate((dim,np.zeros((d,count_n - range_n))),axis=1)
                temp = temp_vec[0]
                for e in range(len(temp)):
                    dim[t, e] = temp[e]

            elif(count_n < range_n):
                dim = np.concatenate((dim,np.zeros((d,range_n - count_n))),axis =1)
                temp = temp_vec[0]
                for i in range(len(temp)):
                    dim[t,i] = temp[i]
        return dim,dim_length


def score(blob,var,var_parents):
    score = 0
    n = blob.n_samples
    dim_var = blob.var_range_length[0,var]
    range_var = blob.var_range[var,:]
    r_i = dim_var
    data_o = blob.data
    used = np.zeros(n,dtype='int64')
    d = 1
    #Get first unproccessed sample
    while(d <= n):
        freq = np.zeros(int(dim_var),dtype='int64')
        while(d <= n and used[d-1] == 1):
            d += 1;
        if(d > n):
            break
        for i in range(int(dim_var)):
            if(range_var[i] == data_o[d-1,var]):
                break
        freq[i] = 1
        used[d-1] = 1
        parent = data[d-1,var_parents]
        d += 1
        if(d > n):
            break
        #count frequencies of states while keeping track of used samples
        for j in range(d-1,n):
            if(used[j] == 0):
                if((parent==data[j,var_parents]).all()):
                    i = 0
                    while range_var[i] != data[j,var]:
                        i += 1
                    freq[i] += 1
                    used[j] = 1
        sum_m = np.sum(freq)
        r_i = int(r_i)
        #Finally, sum over frequencies to get log likelihood bayesian score
        #with uniform priors
        for j in range(1,r_i+1):
            if(freq[j-1] != 0):
                score += ln_gamma(freq[j-1]+1)
        score += ln_gamma(r_i) - ln_gamma(sum_m + r_i)-0.5*log(n)*(r_i-1)*2**len(var_parents)
        #score += ln_gamma(r_i) - ln_gamma(sum_m + r_i)
    return score

#Data structure to hold samples and dimension state info.
class data_blob:
    def __init__(self, _data):
        self.var_number = np.size(_data[0,:])
        self.n_samples = np.size(_data[:,0])
        self.data = _data
        (self.var_range, self.var_range_length) = get_dim_range(_data,np.arange(0,self.var_number).reshape(1,self.var_number))


#k2 uses scoring function to iteratively find best dag given a topological ordering
def k2(blob,order,constraint_u):
    dim = blob.var_number
    dag = np.zeros((dim,dim),dtype='int64')
    k2_score = np.zeros((1,dim),dtype='float')
    for i in range(1,dim):
        parent = np.zeros((dim,1))
        ok = 1
        p_old = -1e10
        while(ok == 1 and np.sum(parent) <= constraint_u):
            local_max = -10e10
            local_node = 0
            #iterate through possible parent connections to determine best action
            for j in range(i-1,-1,-1):
                if(parent[order[j]] == 0):
                    parent[order[j]] = 1
                    #score this node
                    local_score = score(blob,order[i],find(parent[:,0],1))
                    #determine local max
                    if(local_score > local_max):
                        local_max = local_score
                        local_node = order[j]
                    #mark parent processed
                    parent[order[j]] = 0
            #assign the highest parent
            p_new = local_max
            if(p_new > p_old):
                p_old = p_new
                parent[local_node] = 1
            else:
                ok = 0
        k2_score[0,order[i]] = p_old
        dag[:,order[i]] = parent.reshape(blob.var_number)
    return dag, k2_score




#####################################################################################

def topological_sort(graph):
    queue = Queue()
    indegreeMap = {}
    for i in range(graph.numVertices):
        indegreeMap[i] = graph.get_indegree(i)
        if indegreeMap[i]==0:
            queue.put(i)

    sortedList = []
    while not queue.empty():
        vertex = queue.get()
        sortedList.append(vertex)
        for v in graph.get_adjacent_vertices(vertex):
            indegreeMap[v] -=1
            if indegreeMap[v] == 0:
                queue.put(v)

    if len(sortedList) != graph.numVertices :
        raise ValueError("This graph has a cycle")

    print(sortedList)
    return sortedList



####################################################################################
###  baseline
'''
data_set = 'D:/paper/BayesianNetwork/R bayesian/newasia.csv'
categories = np.genfromtxt(data_set, delimiter=',', max_rows=1, dtype=str)
data = genfromtxt(data_set, dtype='int64', delimiter=',', skip_header=True)
sum_socre = 0
# initialize "the blob" and map its variable names to indicies
for a in range(10):
    start = a * 10000
    end = (a + 1) * 10000
    g = data_blob(data[start:end, ])
    data_id = data[start:end, ]
    data_id1 = data_id.transpose()
    mapping = map_categories(categories)
    # set the maximum number of parents any node can have
    iters = 1
    p_lim_max = 3
    # iterate from p_lim_floor to p_lim_max with random restart
    p_lim_floor = 2
    best_score = -10e10
    best_dag = np.zeros((1, 1))

    # order
    m = np.shape(data_id1)[1]
    print('len(data_id1)', len(data_id1))
    seq_array = np.zeros([len(data_id1), len(data_id1)])
    for i in range(len(data_id1)):
        for j in range(len(data_id1)):
            for k in range(m):
                if data_id1[i, k] == 1 and data_id1[j, k] == 1:
                    seq_array[i, j] += 1
    print('seq_array', seq_array)
    ori_array = data_id1.sum(axis=1)

    print('ori_array=', ori_array)
    print('m = ', np.shape(ori_array))
    print("seq_array_shape", np.shape(seq_array))

    seq_arr = seq_array / ori_array
    confidenceR = seq_arr.T

    # 错误的

    supportWQx=(data_id1==0).sum(axis=1)
    supportWQxQy=np.zeros([len(data_id1),len(data_id1)])
    for i in range (len(data_id1)):
        for j in range (len(data_id1)):
            for k in range (m):
                if data_id1[i,k]==0 and data_id1[j,k]==0:
                    supportWQxQy[i,j]+=1
    confidenceW=supportWQxQy/supportWQx
    confidenceW=confidenceW.T
    print('confidenceW',confidenceW)



    for i in range(len(data_id1)):
        confidenceR[i, i] = 0
        confidenceW[i, i] = 0
    confidenceR[np.isnan(confidenceR)] = 0
    confidenceW[np.isnan(confidenceW)]=0
    G1 = np.zeros([len(data_id1), len(data_id1)])

    for i in range(len(data_id1)):

        for j in range(len(data_id1)):
            for k in range(m):
                G1[i, j] = (data_id1[i] == data_id1[j]).sum()

    for i in range(len(data_id1)):
        for j in range(len(data_id1)):
            if G1[i, j] < m * 0.4:
                confidenceR[i, j] = 0
                confidenceW[i, j] = 0
    RelationshipGraph = np.zeros([len(data_id1), len(data_id1)])

    for i in range(len(data_id1)):
        for j in range(len(data_id1)):
            if confidenceR[i, j] != 0:
                if RelationshipGraph[i, j] < confidenceR[i, j]:
                    RelationshipGraph[i, j] = confidenceR[i, j]
    for i in range(len(data_id1)):
        for j in range(len(data_id1)):
            if confidenceW[i,j] !=0:
                if RelationshipGraph[j,i] <confidenceW[i,j]:
                   RelationshipGraph[j,i]=confidenceW[i,j]



    for i in range(len(RelationshipGraph)):
        for j in range(len(RelationshipGraph)):
            if RelationshipGraph[i, j] != 0 and RelationshipGraph[j, i] != 0:
                if RelationshipGraph[i, j] > RelationshipGraph[j, i]:
                    RelationshipGraph[j, i] = 0
                else:
                    RelationshipGraph[i, j] = 0

    RelationshipGraph1 = RelationshipGraph.copy()
    for i in range(len(RelationshipGraph)):
        for j in range(len(RelationshipGraph)):
            if RelationshipGraph1[i, j] < 0.9:
                RelationshipGraph1[i, j] = 0
    print('Relationship Gtaph', RelationshipGraph)
    print('Relationship Gtaph1', RelationshipGraph1)
    np.set_printoptions(linewidth=np.inf)


    
    
    
    
    gr = AdjacencyMatrixGraph(len(RelationshipGraph1), directed=True)
    for i in range(len(RelationshipGraph1)):
        for j in range(len(RelationshipGraph1)):
            if RelationshipGraph1[i, j] != 0:
                gr.add_edge(i, j)
    gr.display()
    topo = topological_sort(gr)
    print(topo)
    orders = []
    for re in range(1, len(topo) + 1):
        orders.append((topo[len(topo) - re]))
    print('orders:', orders)
    t1 = time.clock()
    maxscore = 0

    # score
    for i in range(iters):
        for u in range(p_lim_floor, p_lim_max):
            # generate random ordering
             #order = order_dfs
            order = topo
            #order = np.arange(g.var_number)
            (dags, k2_scores) = k2(g, order, u)
            scores = np.sum(k2_scores)
            if (scores > best_score):
                best_score = scores
                best_dag = dags
    t2 = time.clock()

    #best_bic = model_change(dags,data_id)
    sum_socre += scores
    #print("model_chage:",model_change(dags,data_id))
    # filename = '../graph/asias_orders.gph'
    # graph_out(dags,filename,mapping)
    print("bic", a + 1, ":", scores)
    print(dags)
    print('running time:', t2 - t1)
print('bic mean:',sum_socre/10.0)

##########################################################################
# concept map
'''

mean_time = 0

data_set = 'D:/paper/construct concept map/LPG-algorithm-datasets-master/files/new_qc_g5.csv'
categories = np.genfromtxt(data_set, delimiter=',', max_rows=1, dtype=str)
data = genfromtxt(data_set, dtype='int64', delimiter=',', skip_header=True)
g = data_blob(data)
data_id1 = data.transpose()
mapping = map_categories(categories)
#set the maximum number of parents any node can have
iters = 1
p_lim_max = 3
#iterate from p_lim_floor to p_lim_max with random restart
p_lim_floor = 2
best_score = -10e10
best_dag = np.zeros((1,1))
t1 = time.clock()
# order

m = np.shape(data_id1)[1]
print('len(data_id1)', len(data_id1))
seq_array = np.zeros([len(data_id1), len(data_id1)])
for i in range(len(data_id1)):
    for j in range(len(data_id1)):
        for k in range(m):
            if data_id1[i, k] == 1 and data_id1[j, k] == 1:
                seq_array[i, j] += 1
print('seq_array', seq_array)
ori_array = data_id1.sum(axis=1)

print('ori_array=', ori_array)
print('m = ', np.shape(ori_array))
print("seq_array_shape", np.shape(seq_array))

seq_arr = seq_array / ori_array
confidenceR = seq_arr.T

# 错误的

supportWQx=(data_id1==0).sum(axis=1)
supportWQxQy=np.zeros([len(data_id1),len(data_id1)])
for i in range (len(data_id1)):
    for j in range (len(data_id1)):
        for k in range (m):
            if data_id1[i,k]==0 and data_id1[j,k]==0:
                supportWQxQy[i,j]+=1
confidenceW=supportWQxQy/supportWQx
confidenceW=confidenceW.T
print('confidenceW',confidenceW)



for i in range(len(data_id1)):
    confidenceR[i, i] = 0
    confidenceW[i, i] = 0
confidenceR[np.isnan(confidenceR)] = 0
confidenceW[np.isnan(confidenceW)]=0
G1 = np.zeros([len(data_id1), len(data_id1)])

for i in range(len(data_id1)):

    for j in range(len(data_id1)):
        for k in range(m):
            G1[i, j] = (data_id1[i] == data_id1[j]).sum()

for i in range(len(data_id1)):
    for j in range(len(data_id1)):
        if G1[i, j] < m * 0.4:
            confidenceR[i, j] = 0
            confidenceW[i, j] = 0
RelationshipGraph = np.zeros([len(data_id1), len(data_id1)])

for i in range(len(data_id1)):
    for j in range(len(data_id1)):
        if confidenceR[i, j] != 0:
            if RelationshipGraph[i, j] < confidenceR[i, j]:
                RelationshipGraph[i, j] = confidenceR[i, j]
        if confidenceW[i,j] !=0:
            if RelationshipGraph[j,i] <confidenceW[i,j]:
               RelationshipGraph[j,i]=confidenceW[i,j]



for i in range(len(RelationshipGraph)):
    for j in range(len(RelationshipGraph)):
        if RelationshipGraph[i, j] != 0 and RelationshipGraph[j, i] != 0:
            if RelationshipGraph[j, i] > RelationshipGraph[i, j]:
                RelationshipGraph[i, j] = 0
            else:
                RelationshipGraph[j, i] = 0

RelationshipGraph1 = RelationshipGraph.copy()
for i in range(len(RelationshipGraph)):
    for j in range(len(RelationshipGraph)):
        if RelationshipGraph1[i, j] < 0.9:
            RelationshipGraph1[i, j] = 0

print('Relationship Gtaph1', RelationshipGraph1)
np.set_printoptions(linewidth=np.inf)

##############################################################################
gr = AdjacencyMatrixGraph(len(RelationshipGraph1), directed=True)
for i in range(len(RelationshipGraph1)):
    for j in range(len(RelationshipGraph1)):
        if RelationshipGraph1[i, j] != 0:
            gr.add_edge(i, j)
gr.display()
topo = topological_sort(gr)
print('topo',topo)



###############################################################################
t1 = time.clock()

# score
for i in range(iters):
    for u in range(p_lim_floor, p_lim_max):
        # generate random ordering
        # order = order_dfso
        order = topo
        #order = np.arange(g.var_number)
        (dags, k2_scores) = k2(g, order, u)
        scores = np.sum(k2_scores)
        if (scores > best_score):
            best_score = scores
            best_dag = dags
t2 = time.clock()
#filename = '../graph/new_qc_g5_mean.gph'

print("bic",  1, ":", scores)
print(dags)
t3=t2-t1
print('running time:', t2 - t1)
mean_time = mean_time+ t3

#print('100000 time:',mean_time/100.0)

########################################################################
