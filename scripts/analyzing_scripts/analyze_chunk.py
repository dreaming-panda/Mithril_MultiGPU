import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statistics import mean

if __name__ == "__main__":
    edges = []
    runtimes = []
    vertices = []
    with open("./analyze_chunk_data.txt", "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            line = line.strip()
            line = line.split()
            edges.append([float(line[-1][:-1])])
            runtimes.append(float(line[2][:-2]))
            vertices.append(float(line[-2][:-1]))
    #plt.plot(edges, runtimes, "*")
    #plt.savefig("analyze_chunk.pdf")
    edges = np.asarray(edges)
    runtimes = np.asarray(runtimes)
    reg = LinearRegression().fit(edges, runtimes)
    print("Score = ", reg.score(edges, runtimes))
    print(reg.coef_)
    print(reg.intercept_)
    avg_vertices = mean(vertices)

    v_coef = reg.intercept_ / avg_vertices
    e_coef = reg.coef_

    print("Prediction: Runtime (unit: ms) = %.6f Vertices (unit: K) + %.6f Edges (unit: M)" % (
        v_coef, e_coef
        ))

    #for i in range(len(runtimes)):
    #    pred = v_coef * vertices[i] + e_coef * edges[i][0]
    #    print("Pred %.6f, GroudTruth %.6f, Ratio %.6f" % (
    #        pred, runtimes[i], pred / runtimes[i]
    #        ))

# graph sage: Prediction: Runtime (unit: ms) = 0.036809 Vertices (unit: K) + 0.529691 Edges (unit: M)
# GCN: Runtime (unit: ms) = 0.011158 Vertices (unit: K) + 0.535258 Edges (unit: M)
# GCNII: Runtime (unit: ms) = 0.031549 Vertices (unit: K) + 0.529915 Edges (unit: M)
# Ratio: ~50



