from typing import List, Tuple, Dict
from collections import Counter
from fibheap import Fheap, Node
from collections import deque
import networkx as nx
import numpy as np

class Backend():
    DENSITY_THRESHOLD = 0.8 

    def __init__(self):
        pass

    def functionality_1(self, G: nx.Graph, graph_name: str) -> Dict:
        """
        This function returns the following features of a given Graph:
        - Number of nodes
        - Number of edges
        - Graph Density
        - Graph Degree Distribution
        - Average Degree
        - Graph hubs (nodes with degrees higher than the 95th percentile of the degree distribution)
        - Whether the graph is dense or sparse

        When the graph is directed, the degree distribution, average degree and hubs are calculated for both in-degree and out-degree.

        Args:
            G (nx.Graph): NetworkX Graph
            graph_name (str): Name of the graph

        Returns:
            Dict: Dictionary of the features
        """

        #Here we assert that the name of the graph is either "collaboration" or "citation"
        assert graph_name in ["collaboration", "citation"], "The graph name must be either 'collaboration' or 'citation'"

        #Here we obtain the number of nodes, number of edges and graph density of the graph
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        graph_density = nx.density(G)

        #Here we check whether the graph is dense or sparse. We set the threshold to 0.8 meaning that if the graph density is higher than 0.8, the graph is dense
        is_dense = True if graph_density >= self.DENSITY_THRESHOLD else False

        #If the graph is the collaboration graph (undirected and weighted) we obtain the degree distribution, average degree and hubs in the traditional way
        if graph_name == "collaboration":
            degree_distribution = self.__get_degree_distribution(G)
            avg_degree = self.__get_avg_degree(G)
            hubs = self.__get_hubs(G)
        
        #If the graph is the citation graph (directed and unweighted) we obtain the degree distribution, average degree and hubs for both in-degree and out-degree
        elif graph_name == "citation":
            #Here we obtain the in/out degree distribution and save them in a tuple called degree_distribution
            in_degree_distribution = self.__get_degree_distribution(G, degree="in")
            out_degree_distribution = self.__get_degree_distribution(G, degree="out")
            degree_distribution = (in_degree_distribution, out_degree_distribution)

            #Here we obtain the average in/out degree and save them in a tuple called avg_degree
            avg_in_degree = self.__get_avg_degree(G, degree="in")
            avg_out_degree = self.__get_avg_degree(G, degree="out")
            avg_degree = (avg_in_degree, avg_out_degree)

            #Here we obtain the in/out hubs and save them in a tuple called hubs
            in_hubs = self.__get_hubs(G, degree="in")
            out_hubs = self.__get_hubs(G, degree="out")
            hubs = (in_hubs, out_hubs)
        
        #Here we save all the results in a dictionary called results
        results = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "graph_density": graph_density,
            "degree_distribution": degree_distribution,
            "avg_degree": avg_degree,
            "hubs": hubs,
            "is_dense": is_dense
        }
        #Here we return the results
        return results
    
    def functionality_2(self, G: nx.Graph, node_name: str, graph_name: str) -> Dict[str, float]:
        """
        This function returns the following centrality measures for a given node of a given Graph:
        - Betweenness Centrality
        - PageRank Centrality
        - Closeness Centrality
        - Degree Centrality

        Args:
            G (nx.Graph): NetworkX Graph
            node_name (str): Node name
            graph_name (str): Name of the graph

        Returns:
            Dict[str, float]: Dictionary of the centrality measures
        """
        #Here we assert that the name of the graph is either "collaboration" or "citation"
        assert graph_name in ["collaboration", "citation"], "The graph name must be either 'collaboration' or 'citation'"

        #Here we build a dictionary with the id and name of each node in the graph
        node_name_to_id = self.__get_node_names(G)
        #And we assert that the node name is in the graph
        assert node_name in node_name_to_id.keys(), "The node name must be in the graph"

        #Here we obtain the id of the node
        node_id = node_name_to_id[node_name]

        if graph_name == "collaboration":
            #Since the collaboration graph is undirected and weighted, we can obtain the centralities for this type of graph:
            betweenness_centrality = nx.betweenness_centrality(G, weight="weight", seed=42)[node_id]
            pagerank_centrality = nx.pagerank(G, weight="weight")[node_id]
            closeness_centrality = nx.closeness_centrality(G, distance="weight")[node_id]
        
        elif graph_name == "citation":
            #Since the citation graph is directed and unweighted, we can obtain the centralities for this type of graph:
            betweenness_centrality = nx.betweenness_centrality(G, seed=42)[node_id]
            pagerank_centrality = nx.pagerank(G)[node_id]
            closeness_centrality = nx.closeness_centrality(G)[node_id]

        #Here we obtain the degree centrality
        degree_centrality = nx.degree_centrality(G)[node_id]

        #Here we save all the results in a dictionary called results
        results = {
            "betweenness_centrality": betweenness_centrality,
            "pagerank_centrality": pagerank_centrality,
            "closeness_centrality": closeness_centrality,
            "degree_centrality": degree_centrality
        }

        #Here we return the results
        return results
    
    def functionality_3(self, G: nx.Graph, source: str, target: str, nodes_list: List[int] = "random", N: int = 100) -> Dict:
        """
        This function returns the shortest ordered walk between two nodes of a given weighted and undirected Graph.
        It does this by using the Dijkstra algorithm. The walk must pass through all the nodes in the nodes_list in the given order.

        Args:
            G (nx.Graph): NetworkX Graph
            source (str): Source node name
            target (str): Target node name
            nodes_list (List[int]): Ordered list of nodes to visit
        """
        #First we assert that the graph is undirected and weighted
        assert nx.is_weighted(G), "The graph must be weighted"
        assert not nx.is_directed(G), "The graph must be undirected"

        #If the nodes_list is "random", we create the list by taking only the source and target nodes
        #We create an empty list since after we will insert the source and target nodes
        if nodes_list == "random":
            nodes_list = []

        #Here we have create pairs of nodes from the nodes_list in order to obtain the shortest path between each pair of nodes
        #First, we have to add the source and target nodes to the list
        nodes_list.insert(0, source)
        nodes_list.append(target)

        #Now we convert the names of the nodes to their ids
        node_name_to_id = self.__get_node_names(G)
        nodes_list = [node_name_to_id[node_name] for node_name in nodes_list]
        source = node_name_to_id[source]
        target = node_name_to_id[target]

        #Here we create a subgraph of the original graph with the top N nodes by n_papers (number of published papers)
        G = self.__create_subgraph(G, graph_name="collaboration", N=N)

        #Here we assert that the source and target nodes are in the graph
        assert source in G.nodes(), "The source node must be in the graph"
        assert target in G.nodes(), "The target node must be in the graph"
        #Here we assert that all the nodes in the nodes_list are in the graph
        assert all(node in G.nodes() for node in nodes_list), "All the nodes in the nodes_list must be in the graph"

        #Then we create the pairs of nodes
        pairs = [(nodes_list[i], nodes_list[i+1]) for i in range(len(nodes_list)-1)]
        #Here we initialize the shortest path length and the shortest path
        shortest_path_length = 0
        shortest_path = [source]

        #Here we iterate over the pairs of nodes
        for pair in pairs:
            #Here we obtain the shortest path length and the shortest path between the pair of nodes
            path_length, path = self.__dijkstra_algorithm(G, pair[0], pair[1])
            #We check if the shortest path is empty. If it is, it means that the target node is not reachable from the source node
            #In this case the whole walk is not possible and we return an exception
            if len(path) == 0:
                raise Exception("The target node is not reachable from the source node passing through the given nodes")

            #Here we update the shortest path length and the shortest path
            shortest_path_length += path_length
            shortest_path.extend(path[1:])

        #Now we convert again the ids to the names of the nodes
        shortest_path = [G.nodes[node]["name"] for node in shortest_path]

        #At the end, we return the shortest path length and the shortest path
        return {
            "shortest_path_length": shortest_path_length,
            "shortest_path": shortest_path
        }
    
    def functionality_4(self, G: nx.Graph, node_1: str, node_2: str, N: int = 500) -> Dict:
        #First we create a subgraph of the original graph with the top N nodes by n_papers (number of published papers)
        H = self.__create_subgraph(G, graph_name="collaboration", N=N)

        #Now, we find the connected components of the graph
        components = self.__get_connected_components(H)
        #If the graph is not connected we choose as our graph the largest connected component
        if len(components) > 1:
            #Here we print a warning message
            print("The graph is not connected. We will use the largest connected component")
            #Here we obtain the largest connected component
            largest_component = max(components, key=len)
            #Here we create a subgraph of the largest connected component
            H = H.subgraph(largest_component)

        #Here we make a copy of the graph to modify it
        H = nx.Graph(H)

        #Here we obtain the node ids of the two nodes
        node_name_to_id = self.__get_node_names(G)
        node_1_id = node_name_to_id[node_1]
        node_2_id = node_name_to_id[node_2]

        #Here we assert that the two nodes are in the subgraph
        assert node_1_id in H.nodes(), "The first node must be in the subgraph"
        assert node_2_id in H.nodes(), "The second node must be in the subgraph"

        #Here we obtain the min cut of the graph
        min_cut_value, partition = nx.minimum_cut(H, node_1_id, node_2_id, capacity='weight')

        #Here we create a new graph that contains the two subgraphs that are the result of the minimum cut and their edges
        subgraph1 = H.subgraph(partition[0])
        subgraph2 = H.subgraph(partition[1])
        #Here we add the edges between the two subgraphs
        subgraph = nx.compose(subgraph1, subgraph2)

        #Here we return the results
        return {
            "original_graph": H,
            "cut_graph": subgraph,
            "min_cut_value": min_cut_value,
            "node_ids": (node_1_id, node_2_id)
        }

    
    def functionality_5(self, G: nx.Graph, node_1: str, node_2: str, N: int = 500) -> Dict:
        """
        This function performs the Girvan-Newman algorithm on a given graph to return the minimum number of edges to remove in order to form two connected components.
        The function also returns the two connected components (communities) formed by removing the minimum number of edges and whether the two nodes belong to the same community.

        Args:
            G (nx.Graph): NetworkX Graph
            node_1 (str): Node name
            node_2 (str): Node name
            N (int, optional): Number of nodes. Defaults to 100.

        Returns:
            Dict: Dictionary of the results
        """

        #First we create a subgraph of the original graph with the top N nodes by n_citations (number of citations)
        H = self.__create_subgraph(G, graph_name="citation", N=N)

        #Now, we find the connected components of the graph
        components = self.__get_connected_components(H)
        #If the graph is not connected we choose as our graph the largest connected component
        if len(components) > 1:
            #Here we print a warning message
            print("The graph is not connected. We will use the largest connected component")
            #Here we obtain the largest connected component
            largest_component = max(components, key=len)
            #Here we create a subgraph of the largest connected component
            H = H.subgraph(largest_component)

        #Here we convert the graph to an undirected graph since it is easier to work with undirected graphs
        H = H.to_undirected()

        #Now, we find the connected components of the graph
        components = self.__get_connected_components(H)
        #If the graph is not connected we choose as our graph the largest connected component
        if len(components) > 1:
            #Here we print a warning message
            print("The graph is not connected. We will use the largest connected component")
            #Here we obtain the largest connected component
            largest_component = max(components, key=len)
            #Here we create a subgraph of the largest connected component
            H = H.subgraph(largest_component)
            #Here we obtain the connected components of the graph
            components = self.__get_connected_components(H)

        #Here we make a copy of the graph to modify it
        I = nx.Graph(H)

        #Then we perform the Girvan-Newman algorithm. First we make a while loop that stops when the number of connected components is 2
        #Here we initialize a counter to count the number of iterations i.e. the number of edges removed
        count = 0
        while len(components) == 1:
            #Here we obtain the edge betweenness of the graph
            edge_betweenness = nx.edge_betweenness_centrality(I)
            #Here we obtain the edge with the highest betweenness
            edge_to_remove = sorted(edge_betweenness.items(), key=lambda item: item[1], reverse = True)[0][0]
            #Here we remove the edge with the highest betweenness
            I.remove_edge(*edge_to_remove)
            count += 1

            #Here we obtain the connected components of the graph
            components = self.__get_connected_components(I)

        #Here we obtain the two connected components
        component_1, component_2 = components

        #Here we obtain the node ids of the two nodes
        node_name_to_id = self.__get_node_names(G)
        
        #node_name_to_id = self.__get_node_names(G)
        node_1_id = node_name_to_id[node_1]
        node_2_id = node_name_to_id[node_2]

        #Here we check whether the two nodes belong to the same community
        same_community = True if node_1_id in component_1 and node_2_id in component_1 or node_1_id in component_2 and node_2_id in component_2 else False

        #Here we return the results
        return {
            "original_graph": H,
            "cut_graph": I,
            "num_edges_removed": count,
            "component_1": component_1,
            "component_2": component_2,
            "same_community": same_community,
            "node_ids": (node_1_id, node_2_id)
        }
    
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
            
    def __dijkstra_algorithm(self, G: nx.Graph, source: int, target: int) -> Tuple[float, List[int]]:
        """
        This function returns the shortest path between two nodes of a given graph using the Dijkstra algorithm.

        Args:
            G (nx.Graph): NetworkX Graph
            source (int): Source node id
            target (int): Target node id

        Returns:
            Tuple[float, List[int]]: Tuple of the shortest path length and the shortest path
        """
        #Here we assert that the source and target nodes are in the graph
        assert source in G.nodes(), "The source node must be in the graph"
        assert target in G.nodes(), "The target node must be in the graph"

        #Here we initialize the distance dictionary which will contain the distance from the source node to each node in the graph
        distances = {node: np.inf for node in G.nodes()}

        #Here we initialize the previous dictionary which will contain the previous node of each node in the graph
        #Thus, we can reconstruct the shortest path by starting from the target node and going backwards until we reach the source node
        previous = {node: None for node in G.nodes()}

        #Here we initialize the distance of the source node to 0 since the distance from the source node to itself is 0
        distances[source] = 0

        #To implement the Dijkstra algorithm, we will use a Fibonacci heap since it is the most efficient data structure for this algorithm
        #First, we initialize the nodes dictionary which will contain the nodes of the Fibonacci heap and their keys
        nodes = {node: Node((distances[node], node)) for node in G.nodes()}
        #Then, we initialize the Fibonacci heap and insert all the nodes
        fheap = Fheap()
        #Here we insert all the nodes in the Fibonacci heap
        for node in nodes.values():
            fheap.insert(node)

        #Here we implement the Dijkstra algorithm
        #While the Fibonacci heap is not empty, we perform the following steps:
        while fheap.num_nodes > 0:
            #We extract the minimum node i.e. the node with the lowest key (distance)
            node = fheap.extract_min()
            #We obtain the id of the node
            node_id = node.key[1]

            #If the node is the target, we stop
            #We do this to avoid unnecessary iterations
            if node_id == target:
                break

            #Now we iterate over the neighbors of the node
            for neighbor in G.neighbors(node_id):
                #For each neighbor, we obtain the distance from the source node to the neighbor
                distance = distances[node_id] + G[node_id][neighbor]["weight"]

                #If the distance is lower than the current distance, we update it
                if distance < distances[neighbor]:
                    #We update the distance
                    distances[neighbor] = distance
                    #We update the previous node
                    previous[neighbor] = node_id

                    #Finally, we decrease the key of the neighbor in the Fibonacci heap
                    #This is, we update the key of the neighbor to the new distance
                    fheap.decrease_key(nodes[neighbor], (distance, neighbor))

        #Here we initialize the shortest path list
        shortest_path = []
        #If the target node is reachable from the source node, we reconstruct the shortest path
        if previous.get(target) is not None or target == source:
            shortest_path = self.__reconstruct_path(previous, target)

        #Here we return the shortest path length and the shortest path
        return distances[target], shortest_path
    
    def __breadth_first_search(self, G: nx.Graph, source: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        This function returns the previous and distance dictionaries of a given graph using the Breadth First Search algorithm.

        Args:
            G (nx.Graph): NetworkX Graph
            source (int): Source node id

        Returns:
            Tuple[Dict[int, int], Dict[int, int]]: Tuple of the previous and distance dictionaries
        """
        #Here we assert that the source node is in the graph
        assert source in G.nodes(), "The source node must be in the graph"

        #Here we initialize the distance dictionary which will contain the distance from the source node to each node in the graph
        distances = {node: np.inf for node in G.nodes()}

        #Here we initialize the previous dictionary which will contain the previous node of each node in the graph
        #Thus, we can reconstruct the shortest path by starting from the target node and going backwards until we reach the source node
        previous = {node: None for node in G.nodes()}

        #Here we initialize the distance of the source node to 0 since the distance from the source node to itself is 0
        distances[source] = 0

        #Here we initialize the queue which will contain the nodes to visit
        queue = deque()
        #Here we append the source node to the queue
        queue.append(source)

        #Here we implement the Breadth First Search algorithm
        #While the queue is not empty, we perform the following steps:
        while len(queue) > 0:
            #We pop the first node of the queue
            node = queue.popleft()
            #Now we iterate over the neighbors of the node
            for neighbor in G.neighbors(node):
                #If the distance from the source node to the neighbor is infinite, it means that the neighbor has not been visited yet
                #In this case, we update the distance and the previous node
                if distances[neighbor] == np.inf:
                    #We update the distance
                    distances[neighbor] = distances[node] + 1
                    #We update the previous node
                    previous[neighbor] = node
                    #Finally, we append the neighbor to the queue
                    queue.append(neighbor)

        #Here we return the previous and distance dictionaries
        return previous, distances
    
    def __depth_first_search(self, G: nx.Graph, source_node: int, visited: set) -> List[int]:
        """
        This function returns the component of a given node in a graph using the Depth First Search algorithm.

        Args:
            G (nx.Graph): NetworkX Graph
            source_node (int): Source node id
            visited (set): Set of visited nodes

        Returns:
            List[int]: Component
        """
        #Here we initialize the set of visited nodes and add the source node to it
        visited.add(source_node)
        #Here we initialize the component list with the source node
        component = [source_node]
        #Now we iterate over the neighbors of the source node
        for neighbor in G.neighbors(source_node):
            #If the neighbor has not been visited yet, we append the neighbor to the component list
            if neighbor not in visited:
                #We perform a recursive call to the depth_first_search function and extend the component list with the result
                component.extend(self.__depth_first_search(G, neighbor, visited))
        
        #Here we return the component list
        return component

    def __get_connected_components(self, G: nx.Graph) -> List[List[int]]:
        """
        This function returns the connected components of a given graph.

        Args:
            G (nx.Graph): NetworkX Graph

        Returns:
            List[List[int]]: List of connected components
        """
        # Here we initialize the list of connected components
        components = []
        # Here we initialize the set of visited nodes
        visited = set()

        #Here we iterate over the nodes of the graph
        for node in G.nodes():
            #If the node has not been visited yet, we append the component to the components list
            if node not in visited:
                #We perform a recursive call to the depth_first_search function and append the result to the components list
                components.append(self.__depth_first_search(G, node, visited))

        #Here we return the components list
        return components
                    
    def __reconstruct_path(self, previous: Dict[int, int], node_id:int) -> List[int]:
        """
        This function returns the shortest path between two nodes of a given graph using the previous dictionary.

        Args:
            previous (Dict[int, int]): Previous dictionary
            node_id (int): Target node id

        Returns:
            List[int]: Shortest path
        """
        #Here we initialize the shortest path list with the target node
        path =[node_id]

        #While the previous node of the target node is not None, we append the previous node to the shortest path
        #This reconstructs the shortest path by going backwards from the target node to the source node
        while previous.get(node_id) is not None:
            node_id = previous[node_id]
            path.append(node_id)
        
        #Finally, we reverse the shortest path since we want the source node to be the first element of the list
        return path[::-1]

    def __get_node_names(self, G: nx.Graph) -> Dict[str, int]:
        """
        This function returns a dictionary with the id and name of each node in the graph.

        Args:
            G (nx.Graph): NetworkX Graph

        Returns:
            Dict[str, int]: Dictionary of the node names
        """
        #Here we build a dictionary with the id and name of each node in the graph
        node_name_to_id = {node_name: node_id for node_id, node_name in nx.get_node_attributes(G, "name").items()}
        #Here we return the dictionary
        return node_name_to_id

        
    def __get_degree_distribution(self, G: nx.Graph, degree: str = "total") -> Dict[int, int]:
        """
        This function returns the degree distribution of a given graph.
        This is, a dictionary where the keys are the degree values and the values are the frequency of each degree value.

        Args:
            G (nx.Graph): NetworkX Graph
            degree (str, optional): Degree type. Defaults to "total".

        Returns:
            Dict[int, int]: Degree distribution
        """
        #Here we obtain the degree distribution of the graph
        #If the graph is undirected, we obtain the total degree distribution or degree distribution for short
        if degree == "total":
            #Here we obtain the degree distribution by calling the get_degree_histogram function and converting the list to a dictionary
            degree_distribution = {k: v for k,v in enumerate(self.get_degree_histogram(G))}
        #If the graph is directed, we obtain the in/out degree distribution
        elif degree == "in":
            #Here we obtain the in-degree distribution by calling the get_degree_histogram function and converting the list to a dictionary
            degree_distribution = {k: v for k,v in enumerate(self.get_degree_histogram(G, degree="in"))}
        elif degree == "out":
            #Here we obtain the out-degree distribution by calling the get_degree_histogram function and converting the list to a dictionary
            degree_distribution = {k: v for k,v in enumerate(self.get_degree_histogram(G, degree="out"))}
        
        #Here we return the degree distribution
        return degree_distribution
    
    def get_degree_histogram(self, G: nx.Graph, degree: str = "total") -> List[int]:
        """
        This function returns a list of the frequency of each degree value for a given graph.
        This code was partially taken from the NetworkX documentation: 
        https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.degree_histogram.html

        We used the code to adapt the nx.degree_histogram function to work with directed graph and thus obtain the in/out degree distribution.
        We don't take credit for all parts of the code.

        Args:
            G (nx.Graph): NetworkX Graph
            degree (str, optional): Degree type. Defaults to "total".

        Returns:
            List[int]: Degree histogram
        """
        #Here we obtain the degree histogram of the graph
        #If the graph is undirected, we obtain the total degree histogram or degree histogram for short using the nx.degree_histogram function
        if degree == "total":
            return nx.degree_histogram(G)
        #If the graph is directed, we obtain the in/out degree histogram
        elif degree == "in":
            #Here we obtain the in-degree histogram by using the same code as the nx.degree_histogram function but using G.in_degree() instead of G.degree()
            #Here we count the number of nodes with each degree
            counts = Counter(d for n, d in G.in_degree())
            #Here we return a list of the frequency of each degree value
            return [counts.get(i, 0) for i in range(max(counts) + 1)]
        elif degree == "out":
            #Here we obtain the out-degree histogram by using the same code as the nx.degree_histogram function but using G.out_degree() instead of G.degree()
            #Here we count the number of nodes with each degree
            counts = Counter(d for n, d in G.out_degree())
            #Here we return a list of the frequency of each degree value
            return [counts.get(i, 0) for i in range(max(counts) + 1)]
    
    def __get_avg_degree(self, G: nx.Graph, degree: str = "total") -> float:
        """
        This function returns the average degree of a given graph.

        Args:
            G (nx.Graph): NetworkX Graph
            degree (str, optional): Degree type. Defaults to "total".

        Returns:
            float: Average Degree
        """
        #Here we obtain the average degree of the graph
        #If the graph is undirected, we obtain the total average degree or average degree for short
        if degree == "total":
            #Here we obtain the average degree by dividing the sum of all degrees by the number of nodes
            return sum(dict(G.degree()).values())/G.number_of_nodes()
        #If the graph is directed, we obtain the in/out average degree
        elif degree == "in":
            #Here we obtain the average in-degree by dividing the sum of all in-degrees by the number of nodes
            return sum(dict(G.in_degree()).values())/G.number_of_nodes()
        elif degree == "out":
            #Here we obtain the average out-degree by dividing the sum of all out-degrees by the number of nodes
            return sum(dict(G.out_degree()).values())/G.number_of_nodes()
        
    def __get_hubs(self, G: nx.Graph, degree: str = "total") -> List[str]:
        """
        This function returns the hubs of a given graph. Hubs are nodes with degrees higher than the 95th percentile of the degree distribution.

        Args:
            G (nx.Graph): NetworkX Graph
            degree (str, optional): Degree type. Defaults to "total".

        Returns:
            List[str]: List of hubs
        """
        #Here we obtain the hubs of the graph
        #If the graph is undirected we obtain the dictionary of total degrees
        if degree == "total":
            degrees = dict(G.degree())
        #If the graph is directed we obtain the dictionary of in/out degrees
        elif degree == "in":
            degrees = dict(G.in_degree())
        elif degree == "out":
            degrees = dict(G.out_degree())

        #Then we obtain the 95th percentile of the degree distribution using the np.percentile function
        percentile_95 = np.percentile(list(degrees.values()), 95)
        #Here we obtain the hubs by filtering the nodes with degrees higher than the 95th percentile of the degree distribution
        high_degree_nodes = [(node,degree) for node, degree in degrees.items() if degree > percentile_95]
        #Here we sort the hubs by degree in descending order since we want the nodes with the highest degrees first
        high_degree_nodes = [node[0] for node in sorted(high_degree_nodes, key=lambda x: x[1], reverse=True)]
        #Finally we obtain the names of the hubs by using the "name" attribute of each node in order to return a list of strings instead of a list of nodes
        hubs = [G.nodes[node]["name"] for node in high_degree_nodes]

        #Here we return the hubs
        return hubs






    