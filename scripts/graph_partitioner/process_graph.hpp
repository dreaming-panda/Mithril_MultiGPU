#ifndef PROCESS_GRAPH_HPP
#define PROCESS_GRAPH_HPP

#include <algorithm>
#include <string>

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

typedef int VertexId;
typedef uint64_t EdgeId;
typedef float DataType;

struct Edge {
    VertexId src;
    VertexId dst;
} __attribute__((packed));

class GraphProcessor {
    private:
        static void read_file(int f, uint8_t * ptr, long size) {
        	long total_read_bytes = 0;
        	long read_bytes;
        	while (total_read_bytes < size) {
        		read_bytes = read(f, ptr + total_read_bytes, size - total_read_bytes);
        		assert(read_bytes >= 0);
        		total_read_bytes += read_bytes;
        	}
        	assert(total_read_bytes == size);
        }

        static void write_file(int f, uint8_t * ptr, long size) {
        	long total_write_bytes = 0;
        	long write_bytes;
        	while (total_write_bytes < size) {
        		write_bytes = write(f, ptr + total_write_bytes, size - total_write_bytes);
        		assert(write_bytes >= 0);
        		total_write_bytes += write_bytes;
        	}
        	assert(total_write_bytes == size);
        }

        // removing self-loops and duplicated edges
        void prepreocess_edges(
                EdgeId &num_edges, 
                Edge * edges,
                Edge * &processed_edges
                ) {
            printf("Removing self-loops...\n");
            for (EdgeId i = 0; i < num_edges; ++ i) {
                VertexId src = edges[i].src;
                VertexId dst = edges[i].dst;
                edges[i].src = std::min(src, dst);
                edges[i].dst = std::max(src, dst);
            }
            for (EdgeId i = 0; i < num_edges; ++ i) {
                if (edges[i].src == edges[i].dst) {
                    std::swap(edges[i], edges[num_edges - 1]);
                    num_edges --;
                }
            }
            printf("Removing the duplicated edges...\n");
            auto cmp = [](const Edge &a, const Edge &b) {
                if (a.src != b.src) {
                    return a.src < b.src;
                }
                return a.dst < b.dst;
            };
            std::sort(
                    edges, edges + num_edges, cmp
                    );
            bool * removed = new bool [num_edges];
            assert(removed);
            removed[0] = false;
            for (EdgeId i = 1; i < num_edges; ++ i) {
                removed[i] = (edges[i].src == edges[i - 1].src &&
                        edges[i].dst == edges[i - 1].dst);
            }
            for (EdgeId i = 0; i < num_edges; ++ i) {
                if (removed[i]) {
                    std::swap(edges[i], edges[num_edges - 1]);
                    num_edges --;
                }
            }
            printf("Adding reversed edges...\n");
            processed_edges = new Edge [num_edges * 2];
            assert(processed_edges);
            for (EdgeId i = 0; i < num_edges; ++ i) {
                VertexId src = edges[i].src;
                VertexId dst = edges[i].dst;
                processed_edges[i * 2].src = src;
                processed_edges[i * 2].dst = dst;
                processed_edges[i * 2 + 1].src = dst;
                processed_edges[i * 2 + 1].dst = src;
            }
            num_edges *= 2;
            printf("Done added the reversed edges.\n");
        }

        // invoke the python wrapper to partition the graoh
        void perform_partitions(
                VertexId num_vertices, 
                EdgeId num_edges,
                Edge * edges,
                const std::vector<int>& num_partitions
                ) {
            printf("Sorting the edges to construct the adjacent list...\n");
            auto cmp = [](const Edge &a, const Edge &b) {
                if (a.src != b.src) {
                    return a.src < b.src;
                }
                return a.dst < b.dst;
            };
            std::sort(
                    edges, edges + num_edges, cmp
                    );

            printf("Dumping the graph data to the file...\n");
            FILE * graph_file = fopen("./tmp/graph.txt", "w");
            assert(graph_file);
            fprintf(graph_file, "%u \# number of vertices", 
                    num_vertices);
            EdgeId edge_idx = 0;
            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                while (edge_idx < num_edges && edges[edge_idx].src == v_i) {
                    fprintf(graph_file, "%u ", edges[edge_idx].dst);
                    edge_idx ++;
                }
                fprintf(graph_file, "\n");
            }
            assert(fclose(graph_file) == 0);

            FILE * partition_file = fopen("./tmp/num_partitions.txt", "w");
            assert(partition_file);
            for (int i: num_partitions) {
                fprintf(partition_file, "%d\n", i);
            }
            assert(fclose(partition_file) == 0);

            // invoke the python script
            os.system("python ./partiton_graph.py");
        }

        void load_membership(
                int * membership, 
                int num_parts, 
                VertexId num_vertices
                ) {
            std::string membership_file = "./tmp/" + 
                std::to_string(num_parts) + "_parts.txt";
            FILE * f = fopen(membership_file.c_str(), "r");
            assert(f);
            for (VertexId i = 0; i < num_vertices; ++ i) {
                assert(fscanf(f, "%d", &membership[i]) == 1);
            }
            assert(fclose(f) == 0);
        }

        void obtain_id_mappings(
                int num_parts, 
                VertexId num_vertices, 
                int * membership,
                VertexId * id_mapping_old2new,
                VertexId * id_mapping_new2old,
                VertexId * partition_offsets
                ) {
            VertexId part_sizes[num_parts];
            memset(part_sizes, 0, sizeof(part_sizes));
            for (VertexId i = 0; i < num_vertices; ++ i) {
                part_sizes[membership[i]] ++;
            }
            partition_offsets[0] = 0;
            for (int i = 1; i <= num_parts; ++ i) {
                partition_offsets[i] = partition_offsets[i - 1] + part_sizes[i - 1];
            }
            VertexId discovered_vertices[num_parts];
            memset(discovered_vertices, 0, sizeof(discovered_vertices));
            // establish the id mapping
            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                int part = membership[v_i];
                VertexId new_id = partition_offsets[part] + discovered_vertices[part];
                ++ discovered_vertices[part];
                id_mapping_old2new[v_i] = new_id;
                id_mapping_new2old[new_id] = v_i;
            }
            for (int i = 0; i < num_parts; ++ i) {
                assert(discovered_vertices[i] == part_sizes[i]);
            }
        }

        void dump_partition_offsets(
                VertexId num_vertices, 
                VertexId * partition_offsets,
                std::string data_dir,
                int num_parts
                ) {
            std::string partition_file = data_dir + "/partitions.txt";
            FILE * f = fopen(partition_file.c_str(), "w");
            assert(f);
            for (int i = 0; i < num_parts; ++ i) {
                fprintf(f, "%u %u\n", 
                        partition_offsets[i], partition_offsets[i + 1]);
            }
            assert(fclose(f) == 0);
        }

        void dump_graph_topology(
                EdgeId num_edges,
                VertexId num_vertices,
                int feature_size,
                int num_labels,
                Edge * edges,
                int * id_mapping_old2new,
                int * id_mapping_new2old,
                std::string data_dir
                ) {
            // write to the meta data file
            std::string meta_data_file = data_dir + "/meta_data.txt"; 
            FILE * meta_f = fopen(meta_data_file.c_str(), "w");
            assert(meta_f);
            fprintf("%u %lu %d %d\n",
                    num_vertices, num_edges, feature_size, num_labels);
            assert(fclose(meta_f) == 0);

            for (EdgeId i = 0; i < num_edges; ++ i) {
                VertexId src = edges[i].src;
                VertexId dst = edges[i].dst;
                edges[i].src = id_mapping_old2new[src];
                edges[i].dst = id_mapping_old2new[dst];
            }

            // write to the edge list file
            std::string edge_list_file = data_dir + "/edge_list.bin";
            int f = open(
                    edge_list_file.c_str(),
                    O_CREAT | O_WRONLY | O_TRUNC,
                    0644
                    );
            assert(f != -1);
            // CSR
            std::sort(
                    edges, edges + num_edges, [](const Edge &a, const Edge &b) {
                        if (a.src != b.src) {
                            return a.src < b.src;
                        }
                        return a.dst < b.dst;
                    }
                    );
            write_file(f, (uint8_t*) edges, sizeof(Edge) * num_edges);
            // CSC
            std::sort(
                    edges, edges + num_edges, [](const Edge &a, const Edge &b) {
                        if (a.dst != b.dst) {
                            return a.dst < b.dst;
                        }
                        return a.src < b.src;
                    }
                    );
            write_file(f, (uint8_t*) edges, sizeof(Edge) * num_edges);
            assert(fclose(f) == 0);

            for (EdgeId i = 0; i < num_edges; ++ i) {
                VertexId src = edges[i].src;
                VertexId dst = edges[i].dst;
                edges[i].src = id_mapping_new2old[src];
                edges[i].dst = id_mapping_new2old[dst];
            }
        }

        void dump_features(
                VertexId num_vertices,
                int feature_size,
                std::string data_dir,
                VertexId * id_mapping_new2old,
                DataType * features
                ) {
            const int buff_size = 1024;
            DataType * buff = new DataType [buff_size * feature_size];
            assert(buff);
            int buffered_vertices = 0;

            std::string file_name = data_dir + "/feature.bin";
            int f = open(
                    file_name.c_str(),
                    O_CREAT | O_WRONLY | O_TRUNC,
                    0644
                    );
            assert(f != -1);
            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                VertexId old_id = id_mapping_new2old[v_i];
                memcpy(
                        buff + buffered_vertices * feature_size,
                        features + old_id * feature_size,
                        sizeof(DataType) * feature_size
                      );
                buffered_vertices ++;
                if (buffered_vertices >= buff_size || 
                        v_i == num_vertices - 1) {
                    write_file(
                            f, (uint8_t*) buff,
                            sizeof(DataType) * buffered_vertices * feature_size
                            );
                    buffered_vertices = 0;
                }
            }
            assert(close(f) == 0);

            delete [] buff;
        }

        void dump_labels(
                VertexId num_vertices,
                int num_labels,
                std::string data_dir,
                VertexId * id_mapping_new2old,
                DataType * labels
                ) {
            const int buff_size = 1024;
            DataType * buff = new DataType [buff_size * num_labels];
            assert(buff);
            int buffered_vertices = 0;

            std::string file_name = data_dir + "/label.bin";
            int f = open(
                    file_name.c_str(),
                    O_CREAT | O_WRONLY | O_TRUNC,
                    0644
                    );
            assert(f != -1);
            for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
                VertexId old_id = id_mapping_new2old[v_i];
                memcpy(
                        buff + buffered_vertices * num_labels,
                        labels + old_id * num_labels,
                        sizeof(DataType) * num_labels
                      );
                buffered_vertices ++;
                if (buffered_vertices >= buff_size ||
                        v_i == num_vertices - 1) {
                    write_file(
                            f, (uint8_t*) buff,
                            sizeof(DataType) * buffered_vertices * num_labels
                            );
                    buffered_vertices = 0;
                }
            }
            assert(close(f) == 0);

            delete [] buff;
        }

        void dump_data_set_split(
                VertexId num_vertices,
                int * data_set_split,
                std::string data_dir,
                VertexId * id_mapping_new2old
                ) {
            std::string file_name = data_dir + "/split.txt";
            FILE * f = fopen(file_name.c_str(), "w");
            assert(f);
            for (VertexId i = 0; i < num_vertices; ++ i) {
                VertexId old_id = id_mapping_new2old[i];
                fprintf("%u %d\n", i, data_set_split[old_id]);
            }
            assert(fclose(f) == 0);
        }

    public:
        GraphProcessor() {}
        ~GraphProcessor() {}

        void partition_graphs(
                VertexId num_vertices, 
                EdgeId num_edges,
                int feature_size,
                int num_labels,
                Edge * edges, // assumed undirected graphs 
                DataType * features, // very large => probably backed up by a file with mmap
                DataType * labels, // one hot very large => probably backed by a file with mmap
                int * data_set_split,
                const std::vector<int>& num_partitions,
                std::string target_graph_dir
                ) {
            os.system(("mkdir -p " + target_graph_dir).c_str());

            Edge * processed_edges = NULL;
            prepreocess_edges(
                    num_edges, edges, processed_edges
                    );
            assert(processed_edges);
            edges = processed_edges;

            perform_partitions(
                    num_vertices, num_edges, edges,
                    num_partitions
                    );

            int * membership = new int [num_vertices];
            VertexId * id_mapping_old2new = new VertexId [num_vertices];
            VertexId * id_mapping_new2old = new VertexId [num_vertices];
            assert(membership && id_mapping_old2new && id_mapping_new2old);

            for (int num_parts: num_partitions) {
                std::string data_dir = target_graph_dir + "/" + std::to_string(num_parts) 
                    + "_parts";
                std::string create_dir = "mkdir -p " + data_dir;
                os.system(create_dir.c_str());

                load_membership(membership, num_parts, num_vertices);
                VertexId partition_offsets[num_parts + 1];
                obtain_id_mappings(
                        num_parts, num_vertices, membership,
                        id_mapping_old2new, id_mapping_new2old,
                        partition_offsets
                        );
                dump_partition_offsets(
                        num_vertices, partition_offsets, 
                        data_dir, num_parts
                        );
                dump_graph_topology(
                        num_edges, num_vertices, 
                        feature_size, num_labels,
                        edges, id_mapping_old2new,
                        id_mapping_new2old, data_dir
                        );
                dump_features(
                        num_vertices, feature_size, 
                        data_dir, id_mapping_new2old,
                        features
                        );
                dump_labels(
                        num_vertices, num_labels, 
                        data_dir, id_mapping_new2old,
                        labels
                        );
                dump_data_set_split(
                        num_vertices, data_set_split,
                        data_dir, id_mapping_new2old
                        );
            }

            delete [] membership;
            delete [] id_mapping_old2new;
            delete [] id_mapping_new2old;
            delete [] processed_edges;
        }

};

#endif


