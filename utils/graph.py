import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - full : 127 body face and hand keypoints
        - body : 15 body keypointss
        - hands : 42 hand keypoints
        - head : 70 facial keypoints
        - bodyhands : body + hand keypoints
        - headbody : face + body keypoints
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'openpose_orig':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'full':
            self.num_node = 127
            self_link = [(i, i) for i in range(self.num_node)]
            print('using full body')
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26), (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (33, 34), (34, 35), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60), (70, 71), (70, 81), (70, 82), (71, 72), (71, 75), (71, 78), (72, 73), (73, 74), (75, 76), (76, 77), (78, 79), (78, 80), (81 ,83), (82, 84), (85, 86), (86, 87), (87, 88), (88, 89), (90, 91), (91, 92), (92, 93), (94, 95), (95, 96), (96, 97), (98, 99), (99, 100), (100, 101), (102, 103), (103, 104), (104, 105), (85, 90), (85, 94), (85, 98), (85, 102), (106, 107), (107, 108), (108, 109), (109, 110), (111, 112), (112, 113), (113, 114), (115, 116), (116, 117), (117, 118), (119, 120), (120, 121), (121, 122), (123, 124), (124, 125), (125, 126), (106, 111), (106, 115), (106, 119), (106, 123)]
            self.edge = self_link + neighbor_link
            self.center = 71
        elif layout == 'headbody':
            self.num_node = 85
            self_link = [(i, i) for i in range(self.num_node)]
            print('using head body')
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26), (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (33, 34), (34, 35), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60), (70, 71), (70, 81), (70, 82), (71, 72), (71, 75), (71, 78), (72, 73), (73, 74), (75, 76), (76, 77), (78, 79), (78, 80), (81 ,83), (82, 84)]
            self.edge = self_link + neighbor_link
            self.center = 71
        elif layout == 'body':
            self.num_node = 15
            self_link = [(i, i) for i in range(self.num_node)]
            print('using body')
            neighbor_link = [(0, 1), (0, 11), (0, 12), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 10), (11 ,13), (12, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'head':
            self.num_node = 70
            self_link = [(i, i) for i in range(self.num_node)]
            print('using head')
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26), (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (33, 34), (34, 35), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)]
            self.edge = self_link + neighbor_link
            self.center = 30
        elif layout == 'hands':
            self.num_node = 42
            self_link = [(i, i) for i in range(self.num_node)]
            print('using hands')
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (0, 9), (0, 13), (0, 17), (21, 22), (22, 23), (23, 24), (24, 25), (26, 27), (27, 28), (28, 29), (30, 31), (31, 32), (32, 33), (34, 35), (35, 36), (36, 37), (38, 39), (39, 40), (40, 41), (21, 26), (21, 30), (21, 34), (21, 38)]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'bodyhands':
            self.num_node = 57
            self_link = [(i, i) for i in range(self.num_node)]
            print('using body hands')
            neighbor_link = [(0, 1), (0, 11), (0, 12), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 10), (11 ,13), (12, 14), (15, 16), (16, 17), (17, 18), (18, 19), (20, 21), (21, 22), (22, 23), (24, 25), (25, 26), (26, 27), (28, 29), (29, 30), (30, 31), (32, 33), (33, 34), (34, 35), (15, 20), (15, 24), (15, 28), (15, 32), (36, 37), (37, 38), (38, 39), (39, 40), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47), (47, 48), (49, 50), (50, 51), (51, 52), (53, 54), (54, 55), (55, 56), (36, 41), (36, 45), (36, 49), (36, 53)]

            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                              [6, 12], [7, 13], [6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

