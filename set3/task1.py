import copy
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class HopfieldNetwork:
    def __init__(self, n: int, p: int, ts: float = 1):
        self.memory_states = self.construct_memory_states(p, n)
        self.memory_matrix = self.construct_mem_matrix(self.memory_states, p, n)
        self.ts = ts
        self.n = n
        self.p = p
        # print(self.memory_matrix)

    @staticmethod
    def construct_memory_states(p: int, n: int):
        memory_vecs = []
        for i in range(p):
            temp = np.random.rand(n).round()
            temp[temp == 0] = -1
            memory_vecs.append(temp)
        return memory_vecs

    @staticmethod
    def construct_mem_matrix(memory_states: List[np.ndarray], p: int, n: int):
        matrix = []
        for i in range(n):
            temp = []
            for j in range(n):
                if i == j:
                    temp.append(0)
                else:
                    value = 0
                    for k in range(p):
                        value += memory_states[k][i] * memory_states[k][j]
                    temp.append(value)
            matrix.append(np.array(temp))
        return np.stack(matrix)

    @staticmethod
    def overlap(v1: np.ndarray, v2: np.ndarray):
        return (v1@v2) / len(v1)

    def step(self, v: np.ndarray):
        temp = self.memory_matrix @ v
        # temp = np.transpose(self.memory_matrix) @ v
        for i in range(self.n):
            if temp[i] >= 0:
                temp[i] = 1
            else:
                temp[i] = -1
        return temp

    def run(self, v: np.ndarray, t: int, plot: bool = False):
        t = int(t / self.ts)
        overlap = []
        for i in range(t):
            overlap_scores = []
            for j in range(self.p):
                overlap_scores.append(self.overlap(v, self.memory_states[j]))
            overlap.append(overlap_scores)
            v = self.step(v)
        if plot:
            for ov in overlap:
                print(f"Overlap-Scores: {ov}")

            plt.plot(overlap)
            plt.show()
        return overlap[-1]


def p_exp(p_range: int, idx: int = 0, t: int = 5, n: int = 100):
    hop = HopfieldNetwork(n, 1)
    inp = hop.memory_states[0]
    ps = [i for i in range(p_range)]
    y = []
    for p in tqdm(ps, total=len(ps)):
        if p == 0:
            y.append(0)
        elif p == 1:
            y.append(hop.run(inp, t, False)[0])
        else:
            hop_temp = copy.deepcopy(hop)
            hop_temp.p = p
            hop_temp.memory_states = hop_temp.construct_memory_states(p, n)
            hop_temp.memory_states[0] = inp
            hop_temp.memory_matrix = hop_temp.construct_mem_matrix(hop_temp.memory_states, p, n)
            y.append(hop_temp.run(inp, t, False)[0])
    plt.plot(ps, y)
    plt.show()


if __name__ == "__main__":
    # obj = HopfieldNetwork(100, 1)
    # obj.run(HopfieldNetwork.construct_memory_states(1, 100)[0], 5, True)
    p_exp(40, t=30)
