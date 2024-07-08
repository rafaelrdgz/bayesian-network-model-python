import collections
import graphlib
import itertools
import typing

import pandas as pd

from Util import pointwise_mul


class BayesianNetwork:
    """
    Represents a Bayesian Network.

    A Bayesian Network is a probabilistic graphical model that represents a set of random variables and their
    conditional dependencies via a directed acyclic graph (DAG).

    Args:
        *structure: The structure of the Bayesian Network. It can contain nodes and edges. Nodes and edges are represented as tuples of parent node and child node.

    Attributes:
        parents (dict): A dictionary that maps each node to its parent nodes.
        children (dict): A dictionary that maps each node to its child nodes.
        nodes (list): A list of nodes in the Bayesian Network.
        P (dict): A dictionary that stores the probability distributions for each node.

    Methods:
        prepare(): Prepares the Bayesian Network for querying by setting the appropriate names and order of the
            probability distributions.
        ancestors(node): Returns a set of ancestor nodes for a given node.
        variable_elimination(*query, evidence): Performs variable elimination algorithm to compute the probability
            distribution given the query variables and evidence.
        query(*query, evidence): Computes the probability distribution given the query variables and evidence.

    """
    def __init__(self, *structure):
        # Convert single nodes to lists
        def convert_list(obj):
            if isinstance(obj, list):
                return obj
            return [obj]

        # The structure is made up of nodes (scalars) and edges (tuples)
        edges = (e for e in structure if isinstance(e, tuple))
        nodes = set(e for e in structure if not isinstance(e, tuple))

        # Initialize the parents and children dictionaries
        self.parents = collections.defaultdict(set)
        self.children = collections.defaultdict(set)

        # Populate the parents and children
        for parents, children in edges:
            for parent, child in itertools.product(
                convert_list(parents), convert_list(children)
            ):
                self.parents[child].add(parent)
                self.children[parent].add(child)

        # Convert the parents and children to sorted lists
        self.parents = {node: sorted(parents) for node, parents in self.parents.items()}
        self.children = {node: sorted(children) for node, children in self.children.items()}

        # Get the topological order of the nodes
        ts = graphlib.TopologicalSorter()
        for node in sorted({*self.parents.keys(), *self.children.keys(), *nodes}):
            ts.add(node, *self.parents.get(node, []))
        self.nodes = list(ts.static_order())

        self.P = {}


    def prepare(self) -> "BayesianNetwork":
            """
            Prepares the Bayesian network by adjusting the probability tables and names of the nodes.

            Returns:
                BayesianNetwork: The prepared Bayesian network.
            """
            for node, P in self.P.items():
                if node not in self.parents:
                    P.index.name = node   
                elif set(P.index.names) == set([*self.parents[node], node]):
                    P = P.reorder_levels([*self.parents[node], node]) 
                else:
                    P.index.names = [*self.parents[node], node]
                P.sort_index(inplace=True)
                P.name = (
                    f'P({node} | {", ".join(map(str, self.parents[node]))})'
                    if node in self.parents
                    else f"P({node})"
                )


    def ancestors(self, node):
        """
        Returns a set of all ancestors of the given node in the Bayesian network.

        Parameters:
        - node: The node for which to find the ancestors.

        Returns:
        - A set containing all ancestors of the given node.

        """
        parents = self.parents.get(node, ())
        # If the node has parents, return the union of the parents and the ancestors of the parents.
        if parents:
            return set(parents) | set.union(*[self.ancestors(p) for p in parents])
        return set()


    def variable_elimination(self, *query, evidence):
        """
        Performs variable elimination algorithm to compute the query variables distribution.

        Args:
            query: Variable(s) of interest for which the posterior distribution is computed.
            evidence: Dictionary of observed evidence variables and their corresponding values.

        Returns:
            The query variables distribution as a pandas DataFrame.

        """
        # Get the relevant variables
        relevant = {*query, *evidence}
        # Get the ancestors of the relevant variables and add them to the relevant variables
        for node in list(relevant):
            relevant |= self.ancestors(node)
        # Get the hidden variables, we can ignore each node that isn't an ancestor of the query variables or the evidence variables.
        hidden = relevant - {*query, *evidence}

        # Get the factors for the relevant variables
        factors = []
        for node in relevant:
            # Get the conditional probability table for the node
            factor = self.P[node].copy()
            for var, val in evidence.items():
                # Filter the factor based on the evidence values for the node
                if var in factor.index.names:
                    factor = factor[factor.index.get_level_values(var) == val]

            factors.append(factor)

        # Eliminate the hidden variables
        for node in hidden:
            # Find the factors that contain the hidden variable and perform pointwise multiplication 
            prod = pointwise_mul(
                factors.pop(i)
                for i in reversed(range(len(factors)))
                if node in factors[i].index.names
            )
            # Sum out the hidden variable 
            prod = prod.cdt.sum_out(node)
            factors.append(prod)

        # Pointwise multiply the rest of the factors and normalize the result    
        posterior = pointwise_mul(factors)
        posterior = posterior / posterior.sum()
        # Drop the irrelevant variables
        posterior.index = posterior.index.droplevel(
            list(set(posterior.index.names) - set(query))
        )
        return posterior
    

    def query(
        self,
        *query: typing.Tuple[str],
        evidence: dict
    ) -> pd.Series:
        
        if not query:
            raise ValueError("At least one query variable has to be specified")

        for q in query:
            if q in evidence:
                raise ValueError("A query variable cannot be part of the evidence")

        answer = self.variable_elimination(*query, evidence=evidence)
        
        # Rename the answer to reflect the query variables
        answer = answer.rename(f'P({", ".join(query)})')

        # If the index is a MultiIndex(+1 query), reorder the levels and sort the index
        if isinstance(answer.index, pd.MultiIndex):
            answer = answer.reorder_levels(sorted(answer.index.names))

        return answer.sort_index()
