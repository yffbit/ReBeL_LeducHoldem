#include <iostream>
#include <functional>
#include <thread>
#include <mutex>
#include "recursive_solver.h"
#include "net.h"
using namespace std;

class Worker {
public:
    Worker(shared_ptr<Game> game, CFRParam& cfr_param, SolverParam& param, int n_thread, int seed, int n_game, shared_ptr<ValueNet> net) {
        for(int i = 0; i < n_game; i++) game_id.push_back(i);
        strategies.resize(n_game);
        auto thread_func = [&, this, seed]() {
            FullStrategySolver solver(game, cfr_param, param, net, true, true);
            while(run) {
                int id;
                {
                    lock_guard<mutex> lg(mlock);
                    if(game_id.empty()) break;
                    id = game_id.front();
                    game_id.pop_front();
                    if(id % 10 == 0) std::cout << id << std::endl;
                }
                solver.reset(id + seed);
                strategies[id] = solver.get_full_strategy();
            }
        };
        for(int i = 0; i < n_thread; i++) threads.emplace_back(thread_func);
    }
    ~Worker() {
        join();
    }
    void join() {
        for(thread& t : threads) t.join();
        threads.clear();
    }
    void stop() {
        run = false;
    }
    const vector<Strategy>& get_strategies() {
        return strategies;
    }
private:
    deque<int> game_id;
    vector<thread> threads;
    mutex mlock;
    atomic_bool run = true;
    vector<Strategy> strategies;
};

Strategy weight_avg_strategy(const vector<Strategy>& strategies, const BFSTree& bfs_tree, HandVector2& init_prob) {
    int n = strategies.size(), size = bfs_tree.size();
    vector<HandVector2> reach_probs(size);
    Strategy strategy_sum(size);
    for(int i = 0; i < size; i++) {
        if(bfs_tree[i].is_leaf() || bfs_tree[i].node->type != ACTION_NODE) continue;
        int n_act = bfs_tree[i].node->children.size();
        strategy_sum[i] = HandVectorX::Constant(N_CARD, n_act, 0);
    }
    reach_probs[0] = init_prob;
    for(int i = 0; i < n; i++) {// 每个策略组
        const Strategy& strategy = strategies[i];
        if(strategy.size() != size) continue;
        for(int j = 0; j < size; j++) {// 每个节点
            if(bfs_tree[j].is_leaf()) continue;
            bool is_chance = bfs_tree[j].node->type == CHANCE_NODE;
            int player = bfs_tree[j].node->player;
            int child_begin = bfs_tree[j].child_begin, n_act = bfs_tree[j].node->children.size();
            for(int act = 0; act < n_act; act++) {
                reach_probs[child_begin+act] = reach_probs[j];
                if(is_chance) reach_probs[child_begin+act].row(act) = 0;
                else reach_probs[child_begin+act].col(player) *= strategy[j].col(act);
            }
            if(!is_chance) strategy_sum[j] += strategy[j].colwise() * reach_probs[j].col(player);
        }
    }
    for(int i = 0; i < size; i++) {
        if(bfs_tree[i].is_leaf() || bfs_tree[i].node->type != ACTION_NODE) continue;
        int n_act = bfs_tree[i].node->children.size();
        double uniform = (double)1 / n_act;
        HandVector norm = strategy_sum[i].rowwise().sum();
        strategy_sum[i].colwise() /= norm;
        for(int j = 0; j < N_CARD; j++) {
            if(norm(j) == 0) strategy_sum[i].row(j).fill(uniform);
        }
    }
    return strategy_sum;
}

void copy_strategy(const Strategy& src, Strategy& dst, int player, const BFSTree& bfs_tree) {
    int size = bfs_tree.size();
    if(src.size() != size) throw runtime_error("strategy size error");
    if(dst.size() != size) dst.resize(size);
    for(int i = 0; i < size; i++) {
        if(bfs_tree[i].is_leaf() || bfs_tree[i].node->type != ACTION_NODE) continue;
        if(bfs_tree[i].node->player != player) continue;
        dst[i] = src[i];
    }
}

int main(int argc, char* argv[]) {
    if(argc < 3) {
        printf("%s n_game n_thread [net_path]\n", argv[0]);
        return 0;
    }
    shared_ptr<Game> game = make_shared<Game>();
    HandVector2& init_prob = game->init_prob;
    CFRParam cfr_param;
    cfr_param.discount = true;
    cfr_param.alpha = 1.5;
    cfr_param.beta = 0;
    cfr_param.gamma = 2;
    SolverParam param;
    //param.n_thread = 1;
    param.max_iter = 2000;
    param.print_interval = 0;
    LimitSolver full_solver(game, cfr_param, param);
    full_solver.set_subtree_data(game->root, init_prob);
    full_solver.train();
    const Strategy& full_strategy = full_solver.get_avg_strategy();
    int n_game = atoi(argv[1]), n_thread = atoi(argv[2]);
    shared_ptr<ValueNet> net = nullptr;
    if(argc > 3) {
        string path = argv[3];
        printf("%s\n", argv[3]);
        net = make_shared<TorchScriptNet>(path, "cpu");
    }
    else {
        net = make_shared<OracleNet>(game, cfr_param, param);
        n_thread = 1;
    }
    Worker worker(game, cfr_param, param, n_thread, 2022, n_game, net);
    worker.join();
    const vector<Strategy>& strategies = worker.get_strategies();
    const BFSTree& bfs_tree = full_solver.get_tree();
    Strategy avg_strategy = weight_avg_strategy(strategies, bfs_tree, init_prob);
    vector<double> e = full_solver.exploitability(avg_strategy);
    printf("exploitability:%f,%f,%f\n", (e[0]+e[1])/2, e[0], e[1]);
    Strategy comb_st;
    for(int player = 0; player < N_PLAYER; player++) {
        int opp = 1 - player;
        copy_strategy(full_strategy, comb_st, player, bfs_tree);
        copy_strategy(avg_strategy, comb_st, opp, bfs_tree);
        e = full_solver.exploitability(comb_st);
        printf("full_strategy (player %d) vs rebel (player %d)\n", player, opp);
        printf("exploitability:%f,%f,%f\n", (e[0]+e[1])/2, e[0], e[1]);
    }
    return 0;
}
