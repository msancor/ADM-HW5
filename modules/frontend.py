import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
from modules.backend import Backend
import matplotlib.pyplot as plt
from tabulate import tabulate
import networkx as nx
import numpy as np


#Here we set the style of the plots.
#First, we set as default that matplotlib plots text should be in LaTeX format.
plt.rcParams['text.usetex'] = True
#Here we set the font family to serif.
plt.rcParams['font.family'] = 'serif'
#Here we set the font size
plt.rcParams['font.size'] = 10
#Here we set the label size for axes
plt.rcParams['axes.labelsize'] = 10
#Here we set the label weight for axes
plt.rcParams['axes.labelweight'] = 'bold'
#Here we set the title size for axes
plt.rcParams['axes.titlesize'] = 10
#Here we set the ticks label size
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
#Here we set the legend font size
plt.rcParams['legend.fontsize'] = 10
#Here we set the figure title size
plt.rcParams['figure.titlesize'] = 20

class Frontend():
    TOP_HUBS = 10

    def __init__(self):
        self.backend = Backend()

    def print_graph_information(self, G: nx.Graph, graph_name: str, type: str) -> None:
        """
        Function that prints/plots the information of the graph G given the type of information and the graph name.

        Args:
            G (nx.Graph): Graph to print information from.
            graph_name (str): Name of the graph.
            type (str): Type of information to print.
        """
        #Here we call the functionality_1 method from the backend class.
        information_dict = self.backend.functionality_1(G, graph_name)

        #Here we check if the graph name is collaboration or citation.
        #If it is collaboration, we know that the graph is undirected and we can print the information.
        if graph_name == "collaboration":
            #If the type is general information, we print the general information of the graph.
            #This information is the number of nodes, the number of edges, the graph density, the average degree and if the graph is dense or not.
            if type == "general information":
                table = [["Number of Nodes", "Number of Edges", "Graph Density", "Average Degree", "Is Dense"]\
                    ,[information_dict["num_nodes"], information_dict["num_edges"], information_dict["graph_density"], information_dict["avg_degree"], information_dict["is_dense"]]]
                #Here we print the table using the tabulate library.
                print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
            #If the type is hubs, we print the top 10 hubs of the graph.
            elif type == "hubs":
                #Here we get the hubs from the information_dict.
                hubs = information_dict["hubs"]
                table = ["Top 10 Hubs"]
                table.extend(hubs[:self.TOP_HUBS])
                #Here we print the table using the tabulate library.
                print(tabulate([table], tablefmt='fancy_grid'))
            #If the type is degree distribution, we plot the degree distribution of the graph.
            elif type == "degree distribution":
                self.__plot_degree_distribution(information_dict["degree_distribution"], directed=False)
            #If the type is not valid, we raise a ValueError.
            else:
                raise ValueError("Invalid type: Valid types are 'general information', 'hubs' and 'degree distribution'")

        #If the graph name is citation, we know that the graph is directed and we can print the information.
        elif graph_name == "citation":
                #Here we check if the type is general information, hubs or degree distribution.
                if type == "general information":
                    #If the type is general information, we print the general information of the graph.
                    #This information is the number of nodes, the number of edges, the graph density, the average in degree, the average out degree and if the graph is dense or not.
                    avg_in_degree, avg_out_degree = information_dict["avg_degree"]
                    table = [["Number of Nodes", "Number of Edges", "Graph Density", "Average In Degree", "Average Out Degree", "Is Dense"]\
                        ,[information_dict["num_nodes"], information_dict["num_edges"], information_dict["graph_density"], avg_in_degree, avg_out_degree, information_dict["is_dense"]]]
                    #Here we print the table using the tabulate library.
                    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
                #If the type is hubs, we print the top 10 in hubs and the top 10 out hubs of the graph.
                elif type == "hubs":
                    #Here we get the in hubs and the out hubs from the information_dict.
                    in_hubs, out_hubs = information_dict["hubs"]
                    in_table = ["Top 10 In Hubs"]
                    in_table.extend(in_hubs[:self.TOP_HUBS])
                    out_table = [f"Top 10 Out Hubs"]
                    out_table.extend(out_hubs[:self.TOP_HUBS])
                    #Here we combine the in hubs and the out hubs in a single table.
                    table = [in_table, out_table]
                    #Here we print the table using the tabulate library.
                    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
                #If the type is degree distribution, we plot the in degree distribution and the out degree distribution of the graph.
                elif type == "degree distribution":
                    self.__plot_degree_distribution(information_dict["degree_distribution"], directed=True)
                #If the type is not valid, we raise a ValueError.
                else:
                    raise ValueError("Invalid type: Valid types are 'general information', 'hubs' and 'degree distribution'")
        #If the graph name is not valid, we raise a ValueError.         
        else:
            raise ValueError("Invalid graph name: Valid graph names are 'collaboration' and 'citation'")
        
    def print_node_centrality(self, G: nx.Graph, node_name: str, graph_name: str) -> None:
        """
        Function that prints the centrality of a node in a graph.

        Args:
            G (nx.Graph): Graph to print the centrality of a node.
            node_name (str): Name of the node.
            graph_name (str): Name of the graph.
        """
        #Here we call the functionality_2 method from the backend class to get the centrality of the node.
        centralities = self.backend.functionality_2(G, node_name, graph_name)
        
        #Here we create a table with the centrality of the node.
        table = [["Betweenness Centrality", "Page Rank", "Closeness Centrality", "Degree Centrality"]\
            ,[centralities["betweenness_centrality"], centralities["pagerank_centrality"], centralities["closeness_centrality"], centralities["degree_centrality"]]]
        #Here we print the table using the tabulate library.
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    def plot_shortest_walk(self, G: nx.Graph, source_node: str, target_node: str, nodes_list: List[str] = "random", N: int =100) -> None:
        """
        Function that plots the shortest ordered walk between two nodes in a graph.

        Args:
            G (nx.Graph): Graph to plot the shortest walk.
            source_node (str): Name of the source node.
            target_node (str): Name of the target node.
            nodes_list (List[str]): List of nodes the walk must go through in order.
            N (int, optional): Number of nodes to take into account when computing the shortest walk. Defaults to 100.
        """
        #Here we call the functionality_3 method from the backend class to get the shortest walk between two nodes.
        shortest_walk = self.backend.functionality_3(G, source_node, target_node, nodes_list, N)

        #Here we plot the shortest walk.
        plt.figure(figsize = [10,10])

        #First we create a subgraph with the nodes and the edges of the shortest walk.
        H = self.__create_subgraph(G, graph_name="collaboration", N=N)

        #First, we get the attributes of the nodes and the edges.
        weights, sizes, colors, labels1, alpha, edge_labels, edge_colors, edge_alpha = self.__get_plot_attributes(H, shortest_walk)

        #Now we plot the nodes with a Kamada-Kawai layout, where the size and the color of the nodes depend on the betweenness centrality and the pagerank.
        pathcollection = nx.draw_networkx_nodes(H,
                pos=nx.kamada_kawai_layout(H),
                node_size=5000*sizes,
                node_color=colors,
                alpha=alpha)

        #Now we plot the labels of the nodes.
        labels=nx.draw_networkx_labels(H,
                                pos=nx.kamada_kawai_layout(H),
                                labels=labels1,
                                font_size=10,
                                font_family="sans-serif",
                                font_weight="bold")

        #Now we plot the edges with a Kamada-Kawai layout, where the width and the color of the edges depend on the weight of the edges.
        nx.draw_networkx_edges(H,
                pos=nx.kamada_kawai_layout(H),
                width=0.5*weights,
                edge_color=edge_colors,
                alpha=edge_alpha)

        #Now we plot the labels of the edges. The labels are the order of the edges in the path.
        nx.draw_networkx_edge_labels(H,
                pos=nx.kamada_kawai_layout(H),
                edge_labels=edge_labels,
                font_color="red",
                bbox=dict(facecolor="white", alpha=0.1, edgecolor="none"),
                font_size=12,
                font_family="sans-serif")

        #Here we plot the colorbar.
        plt.axis("off")
        plt.colorbar(pathcollection, shrink=0.5, label = "PageRank")
        #Here we plot the length of the shortest path between two nodes.
        print(f"The length of the shortest path between {source_node} and {target_node} is {shortest_walk['shortest_path_length']}")

    def plot_communities(self, G: nx.Graph, node1_name: str, node2_name: str, N: int = 500) -> None:

        #First we run functionality_5 to get the communities of the graph and other information of interest.
        components_dict = self.backend.functionality_5(G, node1_name, node2_name, N)

        #Now we can define subgraphs from the component dictionary resulting from functionality_5.
        #First, the original subgraph without removing any edges.
        H = components_dict["original_graph"]
        #Then, the subgraph with the edges removed.
        I = components_dict["cut_graph"]

        #Now we can get the attributes of the nodes and the edges of the graph.
        colors, sizes = self.__get_community_attributes(H, components_dict)

        #Now we can plot the graph before and after performing the Girvan-Newman algorithm.
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

        #First we plot the original graph. We use the spring layout to highlight the communities.
        nx.draw_networkx_nodes(H,
                pos=nx.spring_layout(H, k=0.01, iterations=50, seed=42),
                node_color="dodgerblue",
                node_size=50,
                ax=ax1)
        #Here we plot the edges of the original graph.
        nx.draw_networkx_edges(H,
                pos=nx.spring_layout(H, k=0.01, iterations=50, seed=42),
                alpha=0.4,
                ax=ax1)
        #Here we label the graph.
        ax1.axis('off')
        ax1.set_title("Original Graph", fontsize=20)

        #Here we plot the graph after performing the Girvan-Newman algorithm.
        nx.draw_networkx_nodes(I,
                pos=nx.spring_layout(I, k=0.01, iterations=50, seed=42),
                node_size=sizes,
                node_color=colors,
                ax=ax2)
        #Here we plot the edges of the graph after performing the Girvan-Newman algorithm.
        nx.draw_networkx_edges(I,
                pos=nx.spring_layout(I, k=0.01, iterations=50, seed=42),
                alpha=0.4,
                ax=ax2)

        #Here we eliminate the axis of the plot.
        ax2.axis('off')
        #Here we create the legend of the plot.
        red_patch = mpatches.Patch(color='lightcoral', label=f'Community 1')
        blue_patch = mpatches.Patch(color='dodgerblue', label=f'Community 2')
        ax2.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=12)
        ax2.set_title("Graph Communities", fontsize=20)

        #Now we must print some information about the functionality
        print("The number of removed links to obtain the two communities is", components_dict["num_edges_removed"])

        #Here we get the node ids of the nodes of interest for each community.
        #We also print the communities to which the nodes of interest belong.
        node1_id, _ = components_dict["node_ids"]
        if components_dict["same_community"]:
            if node1_id in components_dict["component_1"]:
                print(f"The two nodes: {node1_name} and {node2_name} belong to the same community: Community 1")
            else:
                print(f"The two nodes: {node1_name} and {node2_name} belong to the same community: Community 2")
        else:
            if node1_id in components_dict["component_1"]:
                print(f"{node1_name} belongs to Community 1")
                print(f"{node2_name} belongs to Community 2")
            else:
                print(f"{node1_name} belongs to Community 2")
                print(f"{node2_name} belongs to Community 1")


    def __get_community_attributes(self, G: nx.Graph, components_dict: Dict[str, List[str]]) -> Tuple[List[str], List[float]]:
        """
        Function that gets the attributes of the nodes and the edges of the graph for the Visualization 5.

        Args:
            G (nx.Graph): Graph to get the attributes of the nodes and the edges.
            components_dict (Dict[str, List[str]]): Dictionary with the nodes of the communities.

        Returns:
            Tuple[List[str], List[float]]: Tuple with the attributes of the nodes and the edges.
        """
        
        #Here we get the attributes of the nodes.
        #First we initialize two lists, one for the colors and one for the sizes.
        colors, sizes = [], []

        #Then we obtain the node ids of the nodes of interest for each community.
        node1_id, node2_id = components_dict["node_ids"]

        #Then we iterate through the nodes of the graph.
        for node in G.nodes():
            #If the node is in the first community, we color it red.
            #Specifically, if the node is one of the nodes of interest, we color it red and we make it bigger.
            #The other nodes of the first community are colored lightcoral and they are smaller.
            if node in components_dict["component_1"]:
                if node == node1_id or node == node2_id:
                    colors.append("red")
                    sizes.append(500)
                else:
                    colors.append("lightcoral")
                    sizes.append(50)
            #If the node is in the second community, we color it blue.
            #Specifically, if the node is one of the nodes of interest, we color it blue and we make it bigger.
            #The other nodes of the second community are colored dodgerblue and they are smaller.
            else:
                if node == node1_id or node == node2_id:
                    colors.append("blue")
                    sizes.append(500)
                else:
                    colors.append("dodgerblue")
                    sizes.append(50)
        
        #Here we return the colors and the sizes.
        return colors, sizes
    
    def __create_subgraph(self, G: nx.Graph, graph_name: str, N: int = 100) -> nx.Graph:
        """
        This function returns a subgraph of a given graph with the top N nodes by n_papers (number of published papers).

        Args:
            G (nx.Graph): NetworkX Graph
            graph_name (str): Name of the graph
            N (int, optional): Number of nodes. Defaults to 500.

        Returns:
            nx.Graph: Subgraph
        """
        if graph_name == "collaboration":
            key_ = "n_papers"
        elif graph_name == "citation":
            key_ = "n_citations"
        else:
            raise ValueError("Invalid graph name: Valid graph names are 'collaboration' and 'citation'")
        
        #Here we sort the nodes by n_papers in descending order
        nodes = sorted(G.nodes(data=True), key=lambda x: x[1][key_], reverse=True)
        #Then we select the top N nodes
        nodes = nodes[:N]
        #Then we create a list of the node ids
        nodes = [node[0] for node in nodes]
        #Then we create the subgraph
        H = G.subgraph(nodes)
        #Here we return the subgraph
        return H
        

    def __get_plot_attributes(self, G: nx.Graph, shortest_walk: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str], List[float], Dict[Tuple[str, str], str], List[str], List[float]]:
        """
        Function that gets the attributes of the nodes and the edges of the graph.

        Args:
            G (nx.Graph): Graph to get the attributes of the nodes and the edges.
            shortest_walk (Dict[str]): Dictionary with the shortest walk between two nodes.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str], List[float], Dict[Tuple[str, str], str], List[str], List[float]]: Tuple with the attributes of the nodes and the edges.
        """

        #Here we get the attributes of the nodes.
        weights, sizes, colors, labels1, alpha = self.__get_node_attributes(G, shortest_walk)
        #Here we get the attributes of the edges.
        edge_labels, edge_colors, edge_alpha = self.__get_edge_attributes(G, shortest_walk)

        return weights, sizes, colors, labels1, alpha, edge_labels, edge_colors, edge_alpha



    def __get_node_attributes(self, G: nx.Graph, shortest_walk: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str], List[float]]:
        """
        Function that gets the attributes of the nodes of the graph.

        Args:
            G (nx.Graph): Graph to get the attributes of the nodes.
            shortest_walk (Dict[str]): Dictionary with the shortest walk between two nodes.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str], List[float]]: Tuple with the attributes of the nodes.
        """
        #We will use the betweenness centrality and the pagerank as the size and color of the nodes.
        tam=nx.betweenness_centrality(G)
        col=nx.pagerank(G)

        #Here we get the weights of the edges.
        weights = np.array([i['weight'] for i in dict(G.edges).values()])
        #Here we get the sizes and the colors of the nodes.
        sizes = np.array([tam[i] for i in G])
        colors = np.array([col[i] for i in G])

        #We will label only the nodes that pass through the shortest walk.
        #At the same time, we will show only the nodes that pass through the shortest walk.
        labels1 = {}
        alpha = []
        #Here we create a dictionary with the nodes and their names.
        nodes_dict = dict(G.nodes(data=True))
        #Here we iterate through the nodes and we check if the node is in the shortest walk.
        for node in nodes_dict.keys():
            if nodes_dict[node]["name"] in shortest_walk["shortest_path"]:
                #If the node is in the shortest walk, we add the name of the node to the labels1 dictionary.
                labels1[node] = nodes_dict[node]["name"]
                #If the node is in the shortest walk, we add 1 to the alpha list since we want to show the node.
                alpha.append(1)
            else:
                #If the node is not in the shortest walk, we add an empty string to the labels1 dictionary.
                labels1[node] = ''
                #If the node is not in the shortest walk, we add 0.05 to the alpha list since we do not want to show the node.
                alpha.append(0.05)

        return weights, sizes, colors, labels1, alpha


    def __get_edge_attributes(self, G: nx.Graph, shortest_walk: Dict[str, int]) -> Tuple[Dict[Tuple[str, str], str], List[str], List[float]]:
        """
        Function that gets the attributes of the edges of the graph.

        Args:
            G (nx.Graph): Graph to get the attributes of the edges.
            shortest_walk (Dict[str]): Dictionary with the shortest walk between two nodes.

        Returns:
            Tuple[Dict[Tuple[str, str], str], List[str], List[float]]: Tuple with the attributes of the edges.
        """
        #First we create pairs of nodes that are connected by an edge from the path
        path_edges = [(shortest_walk["shortest_path"][i], shortest_walk["shortest_path"][i+1]) for i in range(len(shortest_walk["shortest_path"])-1)]
        #Now we convert the pairs of nodes into pairs of node ids
        node_name_to_id = {node_name: node_id for node_id, node_name in nx.get_node_attributes(G, "name").items()}
        path_edges = [(node_name_to_id[edge[0]], node_name_to_id[edge[1]]) for edge in path_edges]

        #Now we create a dictionary with the labels of the edges
        edge_labels = {}
        #Here we iterate through the edges and we check if the edge is in the path.
        for node_1, node_2 in G.edges():
            if (node_1, node_2) in path_edges or (node_2, node_1) in path_edges:
                try:
                    #If the edge is in the path, we add the order of the edge in the path to the edge_labels dictionary.
                    edge_labels[(node_1, node_2)] = path_edges.index((node_1, node_2)) + 1
                except:
                    edge_labels[(node_1, node_2)] = path_edges.index((node_2, node_1)) + 1
            else:
                #If the edge is not in the path, we add an empty string to the edge_labels dictionary.
                edge_labels[(node_1, node_2)] = ''

        #Now, we will color red the edges that are in the path and black the ones that are not
        #We will also create a list with the transparency of the edges depending on if they are in the path or not
        #Here we create a list with the colors of the edges.
        edge_colors, edge_alpha = [], []
        #Here we iterate through the edges and we check if the edge is in the path.
        for edge in G.edges():
            #If the edge is in the path, we add red to the edge_colors list.
            if (edge[0], edge[1]) in path_edges or (edge[1], edge[0]) in path_edges:
                edge_colors.append("red")
                edge_alpha.append(1)
            else:
                #If the edge is not in the path, we add black to the edge_colors list.
                edge_colors.append("black")
                edge_alpha.append(0.05)

        return edge_labels, edge_colors, edge_alpha


    def __plot_degree_distribution(self, degree_distribution: Dict[int, int], directed: bool) -> None:
        """
        Function that plots the degree distribution of a graph.

        Args:
            degree_distribution (Dict[int, int]): Degree distribution of the graph.
            directed (bool): Boolean that indicates if the graph is directed or not.
        """
        #If the graph is directed, we plot the in degree distribution and the out degree distribution.
        if directed:
            #Here we get the in degree distribution and the out degree distribution from the degree_distribution dictionary.
            in_deg_dist, out_deg_dist = degree_distribution
            #Here we plot the in degree distribution and the out degree distribution.
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), sharey=True, tight_layout=True)
            ax1.hist(in_deg_dist.keys(), weights=in_deg_dist.values(), bins=80, density=True, color = "dodgerblue", edgecolor="black")
            ax1.set_xlabel("In Degree")
            ax1.set_ylabel("Probability")
            ax1.set_title("In Degree Distribution (Citation Network)")
            ax2.hist(out_deg_dist.keys(), weights=out_deg_dist.values(), bins=80, density=True, color = "yellow", edgecolor="black")
            ax2.set_xlabel("Out Degree")
            ax2.set_title("Out Degree Distribution (Citation Network)")
            pass
        #If the graph is not directed, we plot the degree distribution.
        else:
            #Here we plot the degree distribution.
            plt.figure(figsize=(8, 5))
            plt.hist(degree_distribution.keys(), weights=degree_distribution.values(), bins=80, density=True, color = "dodgerblue", edgecolor="black")
            plt.xlabel("Degree")
            plt.ylabel("Probability")
            plt.title("Degree Distribution (Collaboration Network)")
            pass
