from itertools import combinations
from collections import Counter
from typing import List, Tuple
import networkx as nx
import pickle
import ijson
import heapq
import os

class DataHandler():
    RAW_DATA_PATH = "data/raw_data.json"
    CITATION_NETWORK_PATH = "data/citation_network.gpickle"
    COLLABORATION_NETWORK_PATH = "data/collaboration_network.gpickle"
    TOP_PAPERS = 10000

    def __init__(self):
        """
        Constructor for the DataHandler class.
        """
        self.citation_network, self.collaboration_network = self.__get_networks()

    def __get_networks(self) -> Tuple[nx.Graph, nx.Graph]:
        """
        This method returns the citation and collaboration networks.
        If the networks have already been built and saved it loads them from the saved files.

        Returns:
            citation_network (nx.Graph): The citation network.
            collaboration_network (nx.Graph): The collaboration network.
        """
        #First we check if the networks have already been built and saved
        if os.path.isfile(self.CITATION_NETWORK_PATH) and os.path.isfile(self.COLLABORATION_NETWORK_PATH):
            #If they have we load them from the saved files
            with open(self.CITATION_NETWORK_PATH, "rb") as file:
                citation_network = pickle.load(file)

            with open(self.COLLABORATION_NETWORK_PATH, "rb") as file:   
                collaboration_network = pickle.load(file)

        else:
            #If they have not been built and saved we build them and save them
            citation_network, collaboration_network = self.__build_networks()

            with open(self.CITATION_NETWORK_PATH, "wb") as file:
                pickle.dump(citation_network, file)

            with open(self.COLLABORATION_NETWORK_PATH, "wb") as file:
                pickle.dump(collaboration_network, file)

        #Here we return the citation and collaboration networks
        return citation_network, collaboration_network

    def __build_networks(self) -> Tuple[nx.Graph, nx.Graph]:
        """
        This method builds the citation and collaboration networks from the raw data file.

        Returns:
            citation_network (nx.Graph): The citation network.
            collaboration_network (nx.Graph): The collaboration network.
        """
        #First we get the most connected component of the citation network
        most_connected_component = self.__get_most_connected_component()

        #Here we build the citation graph
        citation_graph = self.__build_citation_graph(most_connected_component)

        #Here we build the collaboration graph
        collaboration_graph = self.__build_collaboration_graph(most_connected_component)
        
        #Here we return the citation and collaboration graphs
        return citation_graph, collaboration_graph
    
    def __build_collaboration_graph(self, most_connected_component: List[Tuple[int, int, str, List[str], List[int], List[int]]]) -> nx.Graph:
        """
        This method builds the collaboration graph from the most connected component of the citation network.

        Args:
            most_connected_component (list): The most connected component of the citation network.

        Returns:
            G (nx.Graph): The collaboration graph.
        """
        #Here we define some constants of the tuple i.e. the indexes of the different attributes
        AUTHORS_NAME_ID = 3
        AUTHORS_ID_ID = 4
        
        #First, we initialize a list to store the edges of the collaboration graph
        collaboration_graph = []
        #We also initialize a dictionary to store the attributes of the nodes
        node_attributes = {}

        #Here we iterate over each paper in the most connected component represented by a tuple
        for tuple_ in most_connected_component:
            #For each author in the paper we save its name and the number of papers in which it appears in the node_attributes dictionary
            for i, author in enumerate(tuple_[AUTHORS_ID_ID]):
                #If the author is already in the node_attributes dictionary we increment the number of papers in which it appears
                if author in node_attributes.keys():
                    node_attributes[author]["n_papers"] += 1
                else:
                    #If the author is not in the node_attributes dictionary we add it
                    node_attributes[author] = {"name": tuple_[AUTHORS_NAME_ID][i], "n_papers": 1}
            #At the same time we build the collaboration graph
            #Each undirected edge is represented by a tuple (x,y) where x and y are the authors of the paper
            #We use the combinations function to get all the possible pairs of authors in the paper
            collaboration_graph.extend(combinations(tuple_[AUTHORS_ID_ID], 2))

        #Now we count the number of times each pair of edges appears in the collaboration graph since we want to weight the edges
        #We sort the tuples in the collaboration graph to make sure that we count the same pair of authors regardless of the order
        count = Counter([tuple(sorted(t)) for t in collaboration_graph])

        #We can use the count dictionary to build the collaboration graph as an undirected weighted graph as follows:
        G = nx.Graph((x, y, {'weight': v}) for (x, y), v in count.items())
        #Here we set the node attributes: name and number of papers
        nx.set_node_attributes(G, node_attributes)
        #Here we return the collaboration graph
        return G

    def __build_citation_graph(self, most_connected_component: List[Tuple[int, int, str, List[str], List[int], List[int]]]) -> nx.Graph:
        """
        This method builds the citation graph from the most connected component of the citation network.

        Args:
            most_connected_component (list): The most connected component of the citation network.

        Returns:
            G (nx.Graph): The citation graph.
        """
        #Here we define some constants of the tuple i.e. the indexes of the different attributes
        N_CITATIONS_ID = 0
        PAPER_TITLE_ID = 2
        REFERENCES_ID = -1
        PAPER_ID = 1

        #First, we initialize a list to store the edges of the citation graph
        citation_graph = []
        #We also initialize a dictionary to store the attributes of the nodes
        node_attributes = {}
        
        #Here we iterate over each paper in the most connected component represented by a tuple
        for tuple_ in most_connected_component:
            #First we save the attributes of the paper in the node_attributes dictionary: the name and the number of citations
            node_attributes[tuple_[PAPER_ID]] = {"name": tuple_[PAPER_TITLE_ID], "n_citations": tuple_[N_CITATIONS_ID]}
            #Then, if the paper has references we use them to build the citation graph
            #Each directed edge is represented by a tuple (x,y) where x is the paper and y is the reference
            if tuple_[REFERENCES_ID] is not None:
                citation_graph.extend([(tuple_[PAPER_ID], reference) for reference in tuple_[REFERENCES_ID]])
            else:
                continue
        
        #Now, we need to clean the citation graph from edges that do not have both nodes in the node_attributes dictionary
        #This is since we only care about the top 10000 most cited papers and the interactions between them
        #We do this by making sure the receiver of the citation is in the node_attributes dictionary
        cleaned_citation_graph = [(x,y) for x,y in citation_graph if y in node_attributes.keys()]
        #Finally, we build the citation graph from the cleaned citation graph and the node_attributes dictionary
        #Here we use a directed graph since the citation graph is directed and unweighted
        G = nx.from_edgelist(cleaned_citation_graph, create_using=nx.DiGraph)
        #Here we set the node attributes: name and number of citations
        nx.set_node_attributes(G, node_attributes)
        #Here we return the citation graph
        return G

    def __get_most_connected_component(self) -> List[Tuple[int, int, str, List[str], List[int], List[int]]]:
        """
        This method returns the approximated most connected component of the citation network
        It does this by identifying the top 10000 most cited papers and then extracting information about them from the raw data file.

        Returns:
            heap (list): A list containing the information of the 10000 most cited papers.
        """
        #Here we initialize a min-heap to store the information of the 10000 most cited papers
        heap = []
        heapq.heapify(heap)

        #Here we open the raw data file and parse it using ijson
        with open(self.RAW_DATA_PATH, "r") as file:
            json_items = ijson.items(file, "item")
            #Here we iterate over the items in the file
            for item in json_items:
                #For each item we extract the information we need
                #Here we extract the number of citations, the id and the title for a given paper
                n_citation = item.get("n_citation")
                id = item.get("id")
                title = item.get("title")
                
                #Here we extract the authors and their ids for a given paper if they exist and store them in a list
                if item.get("authors") is not None:
                    authors = [author.get("name") for author in item.get("authors")]
                    authors_id = [author.get("id") for author in item.get("authors")]
                #If there are no authors we set the authors and authors_id to None
                else:
                    authors = None
                    authors_id = None

                #Finally we extract the references for a given paper if they exist and store them in a list
                references = item.get("references")
                
                #Now, if the heap is not full (has less than 10000 elements) we push the information of the paper to the heap
                if len(heap) < self.TOP_PAPERS:
                    heapq.heappush(heap, (n_citation, id, title, authors, authors_id, references))
                #If the heap is full we push the information of the paper to the heap and pop the smallest element
                #This way we keep the 10000 most cited papers in the heap
                else:
                    heapq.heappushpop(heap, (n_citation, id, title, authors, authors_id, references))

        #Here we return the heap as a list
        return list(heap)
