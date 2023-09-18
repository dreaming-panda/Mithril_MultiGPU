import os
import sys

def load_node_lists(num_nodes):
    nodes = []
    with open("./hosts", "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            line = line.strip()
            nodes.append(line)
    assert(len(nodes) == num_nodes)
    nodes.sort()
    return nodes

if __name__ == "__main__":
    assert(len(sys.argv) == 3)
    hostname = sys.argv[1]
    num_nodes = int(sys.argv[2])
    #print(hostname, num_nodes)

    nodes = load_node_lists(num_nodes)
    gpus_per_node = 1
    node_id = None
    for i in range(num_nodes):
        if nodes[i] == hostname:
            node_id = i
            break
    assert(node_id != None)

    master_node = nodes[0]

    command = "python ./main.py %s %s %s" % (
            node_id, num_nodes, master_node
            )
    print("Executing: ", "'%s'" % (command), "on", hostname)
    sys.stdout.flush()
    os.system(command)
