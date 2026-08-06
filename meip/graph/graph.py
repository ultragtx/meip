from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from meip.utils import write_json


class Node:
    """A node in a graph."""

    def __init__(self, index: int, info: dict | None = None) -> None:
        """Initialize a Node.

        Args:
            index (int): the index of this node
            info (dict, optional): any additional information about this node.
        """
        self.index = index
        self.info = info
        self.neighbors: dict[int, list[DirectedEdge | UndirectedEdge]] = {}

    def add_neighbor(self, index, edge) -> None:
        """Draw an directed edge between self and the node specified by index.

        Args:
            index (int): the index of neighboring node
            edge (DirectedEdge): an DirectedEdge object pointing from self to the node.
        """
        if index not in self.neighbors:
            self.neighbors[index] = [edge]
        else:
            self.neighbors[index].append(edge)


class Edge(ABC):
    """Abstract base class for edges in a graph."""

    def __init__(
        self, nodes: list, index: int | None = None, info: dict | None = None
    ) -> None:
        """Initialize an Edge."""
        self.nodes = nodes
        self.index = index
        self.info = info

    def __repr__(self) -> str:
        """String representation of this edge."""
        nodes, index, info = self.nodes, self.index, self.info
        return f"{type(self).__name__}({nodes=}, {index=}, {info=})"

    def __hash__(self) -> int:
        """Hash this edge."""
        img = (self.info or {}).get("image")
        img_str = "" if img is None else img.tobytes()
        return hash((self.nodes[0], self.nodes[1], img_str))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if two edges are equal."""
        raise NotImplementedError


class UndirectedEdge(Edge):
    """An undirected/bi-directed edge in a graph."""

    __hash__ = Edge.__hash__

    def __eq__(self, other: object) -> bool:
        """Check if two undirected edges are equal."""
        return set(self.nodes) == set(other.nodes) and self.info == other.info


class DirectedEdge(Edge):
    """A directed edge in a graph."""

    __hash__ = Edge.__hash__

    def make_undirected(self, index: int, info: dict | None = None) -> UndirectedEdge:
        """Make a directed edge undirected."""
        info = info or {}
        info["distance"] = self.info["distance"]
        return UndirectedEdge(self.nodes, index, info)

    def is_crossing_cell(self) -> bool:
        """Check if the directed edge crosses the periodic cell."""
        img = (self.info or {}).get("image")
        if img is None:
            return False
        return any(img != 0)

    def __eq__(self, other: object) -> bool:
        """Check if the two directed edges are equal.

        Args:
            other (DirectedEdge): another DirectedEdge to compare to

        Returns:
            bool: True if other is the same directed edge, or if other is the directed
                edge with reverse direction of self, else False.
        """
        if not isinstance(other, DirectedEdge):
            return False
        self_img = (self.info or {}).get("image")
        other_img = (other.info or {}).get("image")
        none_img = self_img is other_img is None
        if self.nodes == other.nodes and (none_img or all(self_img == other_img)):
            # the image key here is provided by Pymatgen, which refers to the periodic
            # cell the neighbor node comes from

            # In this case the two directed edges are exactly the same, but this is not
            # supposed tp happen unless there's a bug in Pymatgen. (we will never add
            # the same edge twice in creating a crystal graph.)
            print(
                "!!!!!! the two directed edges are equal but this operation is "
                "not supposed to happen",
                file=sys.stderr,
            )
            return True

        # In this case the first edge is from node i to j and the second edge is
        # from node j to i
        if self.nodes == other.nodes[::-1]:
            pass
        return self.nodes == other.nodes[::-1] and (
            none_img or all(self_img == -1 * other_img)
        )


class Graph:
    """A graph for storing the neighbor information of atoms."""

    def __init__(self, nodes: list[Node]) -> None:
        """Initialize a Graph from a list of nodes."""
        self.nodes = nodes
        self.directed_edges: dict[frozenset[int], list[DirectedEdge]] = {}
        self.directed_edges_list: list[DirectedEdge] = []
        self.undirected_edges: dict[frozenset[int], list[UndirectedEdge]] = {}
        self.undirected_edges_list: list[UndirectedEdge] = []

    def add_edge(
        self, center_index, neighbor_index, image, distance, dist_tol: float = 1e-6
    ) -> None:
        """Add an directed edge to the graph.

        Args:
            center_index (int): center node index
            neighbor_index (int): neighbor node index
            image (np.array): the periodic cell image the neighbor is from
            distance (float): distance between center and neighbor.
            dist_tol (float): tolerance for distance comparison between edges.
                Default = 1e-6
        """
        # # For debugging
        # if (center_index, neighbor_index) in [(0, 27), (27, 0)]: # and (image == [0, 0, 0]).all():
        #     print(f"add_edge: {center_index}, {neighbor_index}, {image}, {distance}")

        # Create directed_edge (DE) index using the length of added DEs
        directed_edge_index = len(self.directed_edges_list)

        # Create the DE with info
        this_directed_edge = DirectedEdge(
            [center_index, neighbor_index],
            index=directed_edge_index,
            info={"image": image, "distance": distance},
        )

        # Record the two ends of the DE and see if they have been added
        # Here the sequence of the ends doesn't matter since we want to
        #  search for associated undirected_edge (UDE)
        tmp = frozenset([center_index, neighbor_index])

        # This combination of two ends hasn't been added
        # A cell-crossing directed edge only exists in one direction (no return edge), so add it as well
        if (this_directed_edge.is_crossing_cell()) or (tmp not in self.undirected_edges):
            # Create UDE index
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            # Create UDE
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            if tmp not in self.undirected_edges:
                self.undirected_edges[tmp] = [this_undirected_edge]
            else:
                self.undirected_edges[tmp].append(this_undirected_edge)
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)
        else:
            # This pair of nodes has been added before, we need to see if this time,
            # 1. it's the other directed edge of the same undirected edge or,
            # 2. it's another totally different undirected edge that has
            # different image and distance (this is possible consider periodicity)
            for undirected_edge in self.undirected_edges[tmp]:
                if (
                    abs(undirected_edge.info["distance"] - distance) < dist_tol
                    and len(undirected_edge.info["directed_edge_index"]) == 1
                ):
                    # There is an undirected edge with similar length and only one of
                    # the directed edges associated has been added
                    added_dir_edge = self.directed_edges_list[
                        undirected_edge.info["directed_edge_index"][0]
                    ]

                    if added_dir_edge.is_crossing_cell():
                        # The cell-crossing directed edge was added first;
                        # this one should be counted as a separate undirected edge
                        continue

                    # See if the DE that's associated to this UDE
                    # is the reverse of our DE
                    if added_dir_edge == this_directed_edge:
                        # Add UDE index to this DE
                        this_directed_edge.info["undirected_edge_index"] = (
                            added_dir_edge.info["undirected_edge_index"]
                        )

                        # At the center node, draw edge with this DE
                        self.nodes[center_index].add_neighbor(
                            neighbor_index, this_directed_edge
                        )

                        # Add this DE to DE list
                        self.directed_edges_list.append(this_directed_edge)

                        # Add this DE to the UDE
                        undirected_edge.info["directed_edge_index"].append(
                            directed_edge_index
                        )
                        return # Note: return here, skipping the code below

            # This acts as the else branch, since the loop above returns on a match
            # no undirected_edge matches to this directed edge
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp].append(this_undirected_edge)
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)

    def adjacency_list(self) -> tuple[list[list[int]], list[int]]:
        """Get the adjacency list
        Return:
            graph: the adjacency list
                [[0, 1],
                 [0, 2],
                 ...
                 [5, 2]
                 ...  ]]
                the fist column specifies center/source node,
                the second column specifies neighbor/destination node
            directed2undirected:
                [0, 1, ...]
                a list of length = num_directed_edge that specifies
                the undirected edge index corresponding to the directed edges
                represented in each row in the graph adjacency list.
        """
        graph = [edge.nodes for edge in self.directed_edges_list]
        directed2undirected = [
            edge.info["undirected_edge_index"] for edge in self.directed_edges_list
        ]

        # Sort graph by first column and adjust directed2undirected accordingly
        sorted_indices = sorted(range(len(graph)), key=lambda i: graph[i][0])
        # Verify that sorted_indices did not change the order, i.e. it equals range(len(graph))
        if sorted_indices != list(range(len(graph))):
            raise ValueError("sorted_indices changed the order")
        # graph = [graph[i] for i in sorted_indices]
        # directed2undirected = [directed2undirected[i] for i in sorted_indices]

        return graph, directed2undirected

    def line_graph_adjacency_list(self, cutoff) -> tuple[list[list[int]], list[int]]:
        """Get the line graph adjacency list.

        Args:
            cutoff (float): a float to indicate the maximum edge length to be included
                in constructing the line graph, this is used to decrease computation
                complexity

        Return:
            line_graph:
                [[0, 1, 1, 2, 2],
                [0, 1, 1, 4, 23],
                [1, 4, 23, 5, 66],
                ... ...  ]
                the fist column specifies node(atom) index at this angle,
                the second column specifies 1st undirected edge(left bond) index,
                the third column specifies 1st directed edge(left bond) index,
                the fourth column specifies 2nd undirected edge(right bond) index,
                the fifth column specifies 2nd directed edge(right bond) index,.
            undirected2directed:
                [32, 45, ...]
                a list of length = num_undirected_edge that
                maps the undirected edge index to one of its directed edges indices
        """

        line_graph = []
        undirected2directed = []
        visited = set()

        # Loop through all the undirected edges(UDE)
        for left_bond_ude in self.undirected_edges_list:
            # Create the mapping from the UDE index to one of its DE index
            undirected2directed.append(left_bond_ude.info["directed_edge_index"][0])

            # Ignore the UDE if its length is larger than cutoff (3A for default)
            if left_bond_ude.info["distance"] > cutoff:
                continue

            # This UDE is valid to be considered as a node in Bond_Graph

            # Iterate over the associated directed edges
            for left_bond_de_idx in left_bond_ude.info["directed_edge_index"]:
                # Get the two endpoints
                atom_center, atom_left  = self.directed_edges_list[left_bond_de_idx].nodes
                for right_bond_des in self.nodes[atom_center].neighbors.values():
                    for right_bond_de in right_bond_des:
                        # Skip the left bond itself
                        if right_bond_de.index == left_bond_de_idx:
                            continue
                        # Get the index of the corresponding undirected bond
                        right_bond_ude_idx = right_bond_de.info["undirected_edge_index"]

                        # This should not happen
                        if right_bond_ude_idx == left_bond_ude.index:
                            raise ValueError(
                                "This should not happen after the left bond was excluded above"
                            )

                        # Not implemented yet: further deduplication
                        # key = frozenset([left_bond_ude.index, right_bond_ude_idx])
                        # Skip duplicate (left bond, right bond) combinations
                        key = (left_bond_ude.index, right_bond_ude_idx)
                        if key in visited:
                            continue

                        visited.add(key)

                        if right_bond_de.info["distance"] < cutoff:
                            line_graph.append(
                                [
                                    atom_center, # 0, one atom of this UDE
                                    left_bond_ude.index, # 1, index of this UDE | left bond, fixed to the current UDE index
                                    left_bond_de_idx, # 2, index of the DE (starting from center) of this UDE | DE of the left bond
                                    right_bond_ude_idx, # 3, UDE index of one of the DEs leaving center; the right bond paired with the current UDE as left bond
                                    right_bond_de.index, # 4, index of one of the DEs leaving center | DE of the right bond
                                ]
                            )

        # Add the reversed line_graph entries
        for line in line_graph:
            key = (line[3], line[1])
            if key in visited:
                continue
            visited.add(key)
            line_graph.append([line[0], line[3], line[4], line[1], line[2]])

        return line_graph, undirected2directed

    def directed_line_graph_adjacency_list(self, cutoff):
        directed_line_graph = []
        visited = set()

        for left_de_bond in self.directed_edges_list:
            # Skip bonds beyond the cutoff
            if left_de_bond.info["distance"] > cutoff:
                continue

            # Get the two endpoints
            atom_center, atom_left  = left_de_bond.nodes
            for right_bond_des in self.nodes[atom_center].neighbors.values():
                for right_bond_de in right_bond_des:
                    # Skip the left bond itself
                    if right_bond_de.index == left_de_bond.index:
                        continue

                    # Skip DEs belonging to the same UDE
                    right_bond_ude_idx = right_bond_de.info["undirected_edge_index"]
                    left_bond_ude_idx = left_de_bond.info["undirected_edge_index"]
                    if right_bond_ude_idx == left_bond_ude_idx:
                        continue

                    # Skip bonds beyond the cutoff
                    if right_bond_de.info["distance"] > cutoff:
                        continue

                    # Only vectors from the original cell to other cells exist, so the
                    # center atoms of the two bonds are guaranteed to be consistent

                    key = (left_de_bond.index, right_bond_de.index)
                    # key = frozenset([left_de_bond.index, right_bond_de.index]) # cannot deduplicate order-swapped pairs, since the tensor product is not commutative
                    if key in visited:
                        continue

                    visited.add(key)

                    directed_line_graph.append([
                        atom_center,
                        left_de_bond.index,
                        right_bond_de.index,
                    ])

        return directed_line_graph


    def dihedral_graph_adjacency_list(self, cutoff):
        dihedral_graph = []
        visited = set()

        for bond2_ude in self.undirected_edges_list:
            if bond2_ude.info["distance"] > cutoff:
                continue

            bond2_ude_idx = bond2_ude.index
            bond2_de_idx = bond2_ude.info["directed_edge_index"][0]
            bond2_de = self.directed_edges_list[bond2_de_idx]

            # A cell-crossing bond can never be the center bond
            if bond2_de.is_crossing_cell():
                continue

            atomB, atomC = bond2_de.nodes  # direction: B → C

            # --- bond1: A—B
            for bond1_list in self.nodes[atomB].neighbors.values():
                for bond1_de in bond1_list:
                    bond1_ude_idx = bond1_de.info["undirected_edge_index"]
                    if bond1_ude_idx == bond2_ude_idx:
                        continue
                    if bond1_de.info["distance"] > cutoff:
                        continue

                    bond1_de_idx = bond1_de.index
                    atomA = bond1_de.nodes[0] if bond1_de.nodes[1] == atomB else bond1_de.nodes[1]

                    # --- bond3: C—D
                    for bond3_list in self.nodes[atomC].neighbors.values():
                        for bond3_de in bond3_list:
                            bond3_ude_idx = bond3_de.info["undirected_edge_index"]
                            if bond3_ude_idx in {bond1_ude_idx, bond2_ude_idx}:
                                continue
                            if bond3_de.info["distance"] > cutoff:
                                continue

                            bond3_de_idx = bond3_de.index
                            atomD = bond3_de.nodes[1] if bond3_de.nodes[0] == atomC else bond3_de.nodes[0]

                            # --- Deduplicate: key formed by the UDE indices of the three bonds
                            # key = (frozenset([bond1_ude_idx, bond3_ude_idx]), bond2_ude_idx)
                            key = (bond1_ude_idx, bond2_ude_idx, bond3_ude_idx)
                            if key in visited:
                                continue

                            if atomA == atomD:
                                # Prevent three atoms forming a triangle instead of four distinct points
                                # This requires bond1_de and bond2_de to point to different nodes
                                # Use the image to determine whether it is the same node
                                if all(bond1_de.info["image"] == bond2_de.info["image"]):
                                    continue

                            visited.add(key)

                            dihedral_graph.append([
                                atomA, atomB, atomC, atomD,
                                bond1_ude_idx, bond1_de_idx, # 4, 5
                                bond2_ude_idx, bond2_de_idx, # 6, 7
                                bond3_ude_idx, bond3_de_idx, # 8, 9
                            ])

                            # Simplified version for debugging
                            # dihedral_graph.append([
                            #     bond1_ude_idx,
                            #     bond2_ude_idx,
                            #     bond3_ude_idx,
                            #     atomA, atomB, atomC, atomD,
                            # ])

        # Add the reversed dihedral_graph entries
        for dihedral in dihedral_graph:
            key = (dihedral[8], dihedral[6], dihedral[4])
            if key in visited:
                continue
            visited.add(key)

            # Get the reverse of center_de
            center_de_idx = dihedral[7]
            center_de = self.directed_edges_list[center_de_idx]
            center_ude_idx = center_de.info["undirected_edge_index"]
            center_ude = self.undirected_edges_list[center_ude_idx]
            two_directed = center_ude.info["directed_edge_index"]
            assert len(two_directed) == 2
            reversed_de_idx = two_directed[1] if two_directed[0] == center_de_idx else two_directed[0]

            dihedral_graph.append([
                dihedral[3], dihedral[2], dihedral[1], dihedral[0],
                dihedral[8], dihedral[9],
                dihedral[6], reversed_de_idx,
                dihedral[4], dihedral[5],
            ])

        return dihedral_graph


    def undirected2directed(self) -> list[int]:
        """The index map from undirected_edge index to one of its directed_edge
        index.
        """
        return [
            undirected_edge.info["directed_edge_index"][0]
            for undirected_edge in self.undirected_edges_list
        ]

    def as_dict(self) -> dict:
        """Return dictionary serialization of a Graph."""
        return {
            "nodes": self.nodes,
            "directed_edges": self.directed_edges,
            "directed_edges_list": self.directed_edges_list,
            "undirected_edges": self.undirected_edges,
            "undirected_edges_list": self.undirected_edges_list,
        }

    def to(self, filename="graph.json") -> None:
        """Save graph dictionary to file."""
        write_json(self.as_dict(), filename)

    def __repr__(self) -> str:
        """Return string representation of the Graph."""
        num_nodes = len(self.nodes)
        num_directed_edges = len(self.directed_edges_list)
        num_undirected_edges = len(self.undirected_edges_list)
        return f"Graph({num_nodes=}, {num_directed_edges=}, {num_undirected_edges=})"


    def line_graph_adjacency_list_orig(self, cutoff) -> tuple[list[list[int]], list[int]]:
        """Get the line graph adjacency list.

        Args:
            cutoff (float): a float to indicate the maximum edge length to be included
                in constructing the line graph, this is used to decrease computation
                complexity

        Return:
            line_graph:
                [[0, 1, 1, 2, 2],
                [0, 1, 1, 4, 23],
                [1, 4, 23, 5, 66],
                ... ...  ]
                the fist column specifies node(atom) index at this angle,
                the second column specifies 1st undirected edge(left bond) index,
                the third column specifies 1st directed edge(left bond) index,
                the fourth column specifies 2nd undirected edge(right bond) index,
                the fifth column specifies 2nd directed edge(right bond) index,.
            undirected2directed:
                [32, 45, ...]
                a list of length = num_undirected_edge that
                maps the undirected edge index to one of its directed edges indices
        """
        if len(self.directed_edges_list) != 2 * len(self.undirected_edges_list):
            raise ValueError(
                f"Error: number of directed edges={len(self.directed_edges_list)} != 2 "
                f"* number of undirected edges={len(self.undirected_edges_list)}!"
                f"This indicates directed edges are not complete"
            )
        line_graph = []
        undirected2directed = []

        # Loop through all the undirected edges(UDE)
        for u_edge in self.undirected_edges_list:
            # Create the mapping from the UDE index to one of its DE index
            undirected2directed.append(u_edge.info["directed_edge_index"][0])

            # Ignore the UDE if its length is larger than cutoff (3A for default)
            if u_edge.info["distance"] > cutoff:
                continue

            # Assert if the UDE is valid
            # if encountered exception,
            # it means after Atom_Graph creation, the UDE has only 1 DE associated
            # This exception is not encountered from the develop team's experience
            if len(u_edge.info["directed_edge_index"]) != 2:  # noqa: PLR2004
                raise ValueError(
                    "Did not find 2 Directed_edges !!!"
                    f"undirected edge {u_edge} has:"
                    f"edge.info['directed_edge_index'] = "
                    f"{u_edge.info['directed_edge_index']}"
                    f"len directed_edges_list = {len(self.directed_edges_list)}"
                    f"len undirected_edges_list = {len(self.undirected_edges_list)}"
                )

            # This UDE is valid to be considered as a node in Bond_Graph

            # Get the two ends (centers) and the two DE associated with this UDE
            # DE1 should have center=center1 and DE2 should have center=center2
            # We will need to find directed edges with center = center1
            # and create angles with DE1, then do the same for center2 and DE2
            for center, dir_edge in zip(
                u_edge.nodes, u_edge.info["directed_edge_index"], strict=True
            ):
                for directed_edges in self.nodes[center].neighbors.values():
                    for directed_edge in directed_edges:
                        if directed_edge.index == dir_edge:
                            continue
                        if directed_edge.info["distance"] < cutoff:
                            line_graph.append(
                                [
                                    center,
                                    u_edge.index,
                                    dir_edge,
                                    directed_edge.info["undirected_edge_index"],
                                    directed_edge.index,
                                ]
                            )
        return line_graph, undirected2directed