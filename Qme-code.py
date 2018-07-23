#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import networkx as nx
import heapq as hp
import time
import matplotlib.pyplot as plt
import random as rm

#1

#IMPORT FROM LOCALFILE
user_name = "federico"
file_path="/Users/"+user_name+"/Desktop/"
file_data=open(file_path+"full_dblp.json",'r')
data = json.loads(file_data.read())
file_data.close()


#Create GRAPH 1
conferences={}
publications={}
authors={}
G1=nx.Graph()

#Parse Data
for i in data:
    conf_int=i["id_conference_int"]
    pub_int=i["id_publication_int"]
    
    #Add conference to conferences
    #Add publication to conference in conferences
    if conf_int in conferences:
        conferences[conf_int]["pub_int"].append(pub_int)
    else:
        conferences[conf_int]={"conf":i["id_conference"],"pub_int":[pub_int]}
    
    #Add publication to publications
    publications[pub_int]={"pub":i["id_publication"],"title":i["title"],"authors":[]}
    
    #Add author to authors
    #Add publication to every authour
    #Add authors to publication in publications
    for j in i["authors"]:
        aut_id=j["author_id"]
        publications[pub_int]["authors"].append(aut_id)
        if aut_id not in authors:
            authors[aut_id]={"name":j["author"],"pub":[pub_int]}
            G1.add_node(aut_id) #add nodes to graph G
        else:
            authors[aut_id]["pub"].append(pub_int)




############################SHOW RESULTS##################################
conferences[34]
publications[2275]
authors[5823]
authors[5824]



#add edges to graph G1 (weight = Jaccard distance)
for p in publications.values():
    aut_list=p["authors"]
    l=len(aut_list)
    if l>1:
        for i in range(0,l-1):
            a1=aut_list[i]
            for j in range(i+1,l):
                a2=aut_list[j]
                if not G1.has_edge(a1,a2):
                    pub1=authors[a1]["pub"]
                    pub2=authors[a2]["pub"]
                    
                    union=set(pub1).union(pub2)
                    inters=set(pub1).intersection(pub2)
                    w=1-len(inters)/len(union)
                    G1.add_edge(a1,a2,weight=w)





#Centrality measures
conf_int = int(input("Insert conference int:"))

if conf_int in conferences:
    authors_list=set()
    for p_int in conferences[conf_int]["pub_int"]:
        for a in publications[p_int]["authors"]:
            authors_list.add(a)
    sub_G = G1.subgraph(authors_list)
    sizes = []
    for i in sub_G.nodes():
        sizes.append(15*sub_G.degree(i))

    nx.draw(sub_G, with_labels=False, node_size=sizes, node_color="orange", width=0.4, edge_color="blue")
    plt.show()


    # Centralities measures

    deg = nx.degree_centrality(sub_G)
    clos = nx.closeness_centrality(sub_G)
    bet = nx.betweenness_centrality(sub_G)

    plt.figure(1)
    plt.subplot(311)
    plt.hist(list(deg.values()))
    plt.subplot(312)
    plt.hist(list(clos.values()))
    plt.subplot(313)
    plt.hist(list(bet.values()))
    plt.show()
else:
    print("Publication not found")


#PATHS   
def shortest_path_alg(source,target):
    #Step 1    
    push=hp.heappush
    pop=hp.heappop
    G_neighbors=G1.adj
    reached=False
    
    temp_distances=dict([(node,float('inf')) for node in G1.nodes()])
    J=dict([(node,None) for node in G1.nodes()])
    
    final_distances={source:0}
    temp_distances[source]=0
    J[source]=0
    
    t_distances=[]
    hp.heapify(t_distances)
    
    for node,data in G_neighbors[source].items():
        push(t_distances,(data["weight"],node))
        temp_distances[node]=data["weight"]
        J[node]=source
    
    while True:
        #Step 2
        if not t_distances:
            break
        else:
            min_dist,ind=pop(t_distances)
            
            if ind in final_distances:
                continue
            
            final_distances[ind]=min_dist
            
            if ind==target:
                reached=True
                break
        
        #Step 3
        for node, data in G_neighbors[ind].items():
            if node not in final_distances:
                new_dist=final_distances[ind]+data["weight"]
                if temp_distances[node]>new_dist:
                    push(t_distances,(new_dist,node))
                    temp_distances[node]=new_dist
                    J[node]=ind
    
    if reached:
        tot=0
        path=[target]
        prev=J[target]
        while(True):
            tot+=G1[prev][path[-1]]["weight"]
            path.append(prev)
            if prev==source:
                break
            else:
                prev=J[prev]     
            
        return([path[::-1],tot])
    else:
        print("PATH NOT FOUND")
        return ([[],float("inf")])
    
def multi_source_shortest_path_alg(sources):
    #Step 1    
    push=hp.heappush
    pop=hp.heappop
    G_neighbors=G1.adj
    
    final_distances={}
    temp_distances=dict([(node,float('inf')) for node in G1.nodes()])
    J=dict([(node,None) for node in G1.nodes()])
    
    t_distances=[]
    hp.heapify(t_distances)
    
    for source in sources:
        final_distances[source]=0
        temp_distances[source]=0
        J[source]=0
        
        for node,data in G_neighbors[source].items():
            push(t_distances,(data["weight"],node))
            if temp_distances[node]>data["weight"]:
            	temp_distances[node]=data["weight"]
            	J[node]=source            	
        
    while True:
        #Step 2
        if not t_distances:
            break
        else:
            min_dist,ind=pop(t_distances)
            
            if ind in final_distances:
                continue
            
            final_distances[ind]=min_dist
        
        #Step 3
        for node, data in G_neighbors[ind].items():
            if node not in final_distances:
                new_dist=final_distances[ind]+data["weight"]
                if temp_distances[node]>new_dist:
                    push(t_distances,(new_dist,node))
                    temp_distances[node]=new_dist
                    J[node]=ind
    
    all_paths={}
    for target in G1.nodes():
        if target in sources:
            all_paths[target]=[target,0]
        else:
            tot=0
            path=[target]
            prev=J[target]
            
            if prev:
                while(True):
                    tot+=G1[prev][path[-1]]["weight"]
                    path.append(prev)
                    if prev in sources:
                        break
                    else:
                        prev=J[prev]
                all_paths[target]=[path[::-1],tot]
            else:
                all_paths[target]=[[],float('inf')]
                
    return(all_paths)

#SHORTEST PATH
print("Insert start id:")                    
aut_id=int(input())
print("Insert target id:")                    
target_id=int(input())

if aut_id in authors:  
    start_time = time.time()     
    
    result = shortest_path_alg(aut_id,target_id)
    
    print("Path: "+ str(result[0]))
    
    print("Erdös Number: "+ str(result[1]))
       
    print("Time: %s seconds" % (time.time() - start_time))
elif aut_id==target_id:                
    print("Source and target are the same node") 
else:
    print("Author not found")
    


#Multi-source
print("Insert author ids (space separated):")                    
aut_ids=list(set(map(int,input().split(" "))))

if all(a in authors for a in aut_ids):
    start_time = time.time()     
    
    all_paths=multi_source_shortest_path_alg(aut_ids)
    
    #for k in sorted(G.nodes):
        #print("Group number ("+str(k)+") = "+str(all_paths[k][1]))
       
    for k in sorted(G1.nodes())[:5]:
        print("Group number ("+str(k)+") = "+str(all_paths[k][1]))
    
    print("Time: %s seconds" % (time.time() - start_time))
else:
    print("Some authors not found")








#Create GRAPH 2
G2 = nx.DiGraph()

for pub_int in publications.keys():
    G2.add_node(pub_int,attr_dict={"type":"publication"}) #add publication node to graph G

for aut_id in authors.keys():
    G2.add_node(str(aut_id),attr_dict={"type":"author"})
    
    for pub_int in authors[aut_id]["pub"]:
        G2.add_edge(str(aut_id),pub_int) #add edge publ-author to graph G
        G2.add_edge(pub_int,str(aut_id)) #To compensate since didn't find a way to insert an undirected edge in a directed graph

##NB:
#To differentiate between publications_id and author_id, we changed the author's in string






#SUBGRAPH

#Example with author "73"
#Try with radius = 2,3,4, 5 is is not recommended
sub_G=nx.ego_graph(G2,"73",radius=3)

n_colors=[]
for i in sub_G.nodes(data=True):
    if i[1]["type"]=="author":
        n_colors.append("red")
    else:
        n_colors.append("blue")

nx.draw(sub_G,node_size=25,node_color=n_colors, arrows=False)

##Data seems to be clusterized in some way...





###ADD CITATION
## The probability is proportional to the similarity of the title
## A publication can cite only publications with publ_id < own_publ_id (is also proportional to the difference --> the older, the higher)
## Prob is also proportional to Jaccard similarity between the authors of the two publications


### Distance between strings
def dist_dyn(s,t):
    m=len(s)
    n=len(t)
    
    d=[[0] * (m+1) for i in range(n+1)]
    
    for i in range(1,n+1):
        d[i][0] = i
    
    for j in range(1,m+1):
        d[0][j] = j
        
    for i in range(1,n+1):
        for j in range(1,m+1):       
            if s[j-1]==t[i-1]:
                c = 0
            else:
                c = 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + c)
            
    return d[n][m]




#For all the graph is impossible, so we used a subgraph
publ_sub_G = sorted([n for n,d in sub_G.nodes(data=True) if d['attr_dict']['type'] == "publication"])
auth_sub_G = sorted([int(n) for n,d in sub_G.nodes(data=True) if d['attr_dict']['type'] == "author"])

#Get Jaccard:
jaccard={}

l=len(auth_sub_G)
for i in range(0,l-1):
    a1=int(auth_sub_G[i])
    for j in range(i+1,l):
        a2=int(auth_sub_G[j])
        
        pub1=authors[a1]["pub"]
        pub2=authors[a2]["pub"]
        
        union=set(pub1).union(pub2)
        inters=set(pub1).intersection(pub2)
        w=len(inters)/len(union)
        
        jaccard[(a1,a2)]=w
                    

citations=[]
for i in range(1,len(publ_sub_G)):
    print(i) #To 273
    for j in range(0,i):
        if rm.random()>0.6: #to ease computation
            s1=publications[publ_sub_G[i]]["title"]
            s2=publications[publ_sub_G[j]]["title"]
            dist=dist_dyn(s1,s2) #distance between the two titles
            prob_dist=max(0,min((100-dist)/100,1))
            
            #Numbers ad hoc
            a=-4/(publ_sub_G[i]**2)
            b=-a*publ_sub_G[i]
            prob_time=(a*(publ_sub_G[j]**2)+b*publ_sub_G[j])**4
            
            
            auth_1=publications[publ_sub_G[i]]["authors"]
            auth_2=publications[publ_sub_G[j]]["authors"]
            
            jacc_prob=0
            for a1 in auth_1:
                for a2 in auth_2:
                    if a1==a2:
                        jacc_prob+=1
                    else:
                        jacc_prob+=jaccard[(min(a1,a2),max(a1,a2))]
            jacc_prob/=len(set(auth_1).union(auth_2))
            
            tot_prob=(prob_dist+prob_time+jacc_prob)/3
            
            #print(str(prob_dist)+" + "+str(prob_time)+" + "+str(jacc_prob)+" = "+str(tot_prob))
            if rm.random()<tot_prob:
                citations.append([publ_sub_G[j],publ_sub_G[i]])
        
        
sub_G.add_edges_from(citations)

e_colors=[]
for i in sub_G.edges():
    if (type(i[0]) is int) and (type(i[1]) is int):
        e_colors.append("green")
    else:
        e_colors.append("black")

nx.draw(sub_G,node_size=25,node_color=n_colors,edge_color=e_colors)
