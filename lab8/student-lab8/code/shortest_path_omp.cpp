#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

struct EdgeInput {
    int source;
    int target;
    double weight;
};

struct Query {
    int source_id;
    int target_id;
    int source_index;
    int target_index;
};

struct Graph {
    std::vector<int> ids;
    std::unordered_map<int, int> id_to_index;
    std::vector<std::vector<std::pair<int, double>>> adj;
    std::size_t csv_edges = 0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        while (!item.empty() && (item.back() == '\r' || item.back() == '\n' || item.back() == ' ' || item.back() == '\t')) {
            item.pop_back();
        }
        std::size_t start = 0;
        while (start < item.size() && (item[start] == ' ' || item[start] == '\t')) {
            ++start;
        }
        if (start > 0) {
            item.erase(0, start);
        }
        cols.push_back(item);
    }
    return cols;
}

bool parse_int(const std::string& text, int& value) {
    char* end = nullptr;
    long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_double(const std::string& text, double& value) {
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return !(end == text.c_str() || *end != '\0');
}

std::vector<EdgeInput> read_edges(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open graph file: " + path);
    }

    std::vector<EdgeInput> edges;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        auto cols = split_csv_line(line);
        if (cols.size() < 3) {
            continue;
        }

        int u = 0;
        int v = 0;
        double w = 0.0;
        if (!parse_int(cols[0], u) || !parse_int(cols[1], v) || !parse_double(cols[2], w)) {
            continue;  // Header row.
        }
        if (w < 0.0) {
            throw std::runtime_error("negative edge weight is not supported by Dijkstra");
        }
        edges.push_back({u, v, w});
    }
    return edges;
}

Graph build_graph(const std::vector<EdgeInput>& edges) {
    Graph graph;
    graph.csv_edges = edges.size();
    graph.ids.reserve(edges.size() * 2);

    for (const auto& edge : edges) {
        graph.ids.push_back(edge.source);
        graph.ids.push_back(edge.target);
    }
    std::sort(graph.ids.begin(), graph.ids.end());
    graph.ids.erase(std::unique(graph.ids.begin(), graph.ids.end()), graph.ids.end());

    graph.id_to_index.reserve(graph.ids.size() * 2);
    for (std::size_t i = 0; i < graph.ids.size(); ++i) {
        graph.id_to_index[graph.ids[i]] = static_cast<int>(i);
    }

    graph.adj.assign(graph.ids.size(), {});
    for (const auto& edge : edges) {
        int u = graph.id_to_index.at(edge.source);
        int v = graph.id_to_index.at(edge.target);
        graph.adj[u].push_back({v, edge.weight});
        graph.adj[v].push_back({u, edge.weight});
    }
    return graph;
}

std::vector<Query> read_queries(const std::string& path, const Graph& graph) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open query file: " + path);
    }

    std::vector<Query> queries;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        auto cols = split_csv_line(line);
        if (cols.size() < 2) {
            continue;
        }

        int s = 0;
        int t = 0;
        if (!parse_int(cols[0], s) || !parse_int(cols[1], t)) {
            continue;  // Header row.
        }

        auto sit = graph.id_to_index.find(s);
        auto tit = graph.id_to_index.find(t);
        queries.push_back({
            s,
            t,
            sit == graph.id_to_index.end() ? -1 : sit->second,
            tit == graph.id_to_index.end() ? -1 : tit->second,
        });
    }
    return queries;
}

void dijkstra(
    int source,
    const std::vector<std::vector<std::pair<int, double>>>& adj,
    std::vector<double>& dist
) {
    using State = std::pair<double, int>;
    std::fill(dist.begin(), dist.end(), INF);
    std::priority_queue<State, std::vector<State>, std::greater<State>> heap;

    dist[source] = 0.0;
    heap.push({0.0, source});
    while (!heap.empty()) {
        auto [du, u] = heap.top();
        heap.pop();
        if (du != dist[u]) {
            continue;
        }
        for (const auto& [v, w] : adj[u]) {
            double nd = du + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                heap.push({nd, v});
            }
        }
    }
}

void write_answers(
    const std::string& path,
    const std::vector<Query>& queries,
    const std::vector<double>& answers
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open output file: " + path);
    }

    out << "source,target,distance\n";
    out << std::setprecision(10);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        out << queries[i].source_id << ',' << queries[i].target_id << ',';
        if (std::isinf(answers[i])) {
            out << "INF";
        } else {
            out << answers[i];
        }
        out << '\n';
    }
}

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " <graph.csv> <query.csv> <output.csv> [repeat=1] [--no-output]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string graph_path = argv[1];
    const std::string query_path = argv[2];
    const std::string output_path = argv[3];
    int repeat = 1;
    bool no_output = false;

    if (argc >= 5) {
        repeat = std::max(1, std::atoi(argv[4]));
    }
    for (int i = 5; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-output") {
            no_output = true;
        }
    }

    try {
        auto edges = read_edges(graph_path);
        Graph graph = build_graph(edges);
        auto queries = read_queries(query_path, graph);

        std::vector<std::vector<int>> queries_by_source(graph.ids.size());
        std::size_t valid_query_count = 0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            if (queries[i].source_index >= 0) {
                queries_by_source[queries[i].source_index].push_back(static_cast<int>(i));
                if (queries[i].target_index >= 0) {
                    ++valid_query_count;
                }
            }
        }

        std::size_t active_sources = 0;
        for (const auto& bucket : queries_by_source) {
            active_sources += bucket.empty() ? 0 : 1;
        }

        std::vector<double> answers(queries.size(), INF);
        double checksum = 0.0;

        const auto compute_start = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; ++r) {
            double round_checksum = 0.0;
            #pragma omp parallel reduction(+:round_checksum)
            {
                std::vector<double> dist(graph.ids.size(), INF);
                #pragma omp for schedule(dynamic, 1)
                for (int source = 0; source < static_cast<int>(queries_by_source.size()); ++source) {
                    const auto& bucket = queries_by_source[source];
                    if (bucket.empty()) {
                        continue;
                    }
                    dijkstra(source, graph.adj, dist);
                    for (int query_index : bucket) {
                        double value = INF;
                        int target = queries[query_index].target_index;
                        if (target >= 0) {
                            value = dist[target];
                        }
                        if (r == repeat - 1) {
                            answers[query_index] = value;
                        }
                        if (!std::isinf(value)) {
                            round_checksum += value;
                        }
                    }
                }
            }
            checksum += round_checksum;
        }
        const auto compute_end = std::chrono::steady_clock::now();
        double compute_seconds = std::chrono::duration<double>(compute_end - compute_start).count();

        double output_seconds = 0.0;
        if (!no_output) {
            const auto output_start = std::chrono::steady_clock::now();
            write_answers(output_path, queries, answers);
            const auto output_end = std::chrono::steady_clock::now();
            output_seconds = std::chrono::duration<double>(output_end - output_start).count();
        }

        long long undirected_arcs = 0;
        for (const auto& edges_of_vertex : graph.adj) {
            undirected_arcs += static_cast<long long>(edges_of_vertex.size());
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "graph=" << graph_path
                  << " vertices=" << graph.ids.size()
                  << " csv_edges=" << graph.csv_edges
                  << " undirected_arcs=" << undirected_arcs
                  << " queries=" << queries.size()
                  << " valid_queries=" << valid_query_count
                  << " active_sources=" << active_sources
                  << " threads=" << omp_get_max_threads()
                  << " repeat=" << repeat
                  << " compute_seconds_total=" << compute_seconds
                  << " compute_seconds_avg=" << compute_seconds / repeat
                  << " output_seconds=" << output_seconds
                  << " checksum=" << checksum
                  << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
