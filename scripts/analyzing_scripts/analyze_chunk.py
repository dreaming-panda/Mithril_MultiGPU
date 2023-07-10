import matplotlib.pyplot as plt

if __name__ == "__main__":
    edges = []
    runtimes = []
    with open("./analyze_chunk_data.txt", "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            line = line.strip()
            line = line.split()
            edges.append(float(line[-1][:-1]))
            runtimes.append(float(line[2][:-2]))
    plt.plot(edges, runtimes, "*")
    plt.savefig("analyze_chunk.pdf")

