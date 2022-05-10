#if !defined(_RECURSIVE_SOLVER_H_)
#define _RECURSIVE_SOLVER_H_

#include <random>
#include <tuple>
#include <utility>
#include "limit_solver.h"
using std::deque;
using std::tuple;
using std::pair;
using std::mt19937;
using std::discrete_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

// 生成训练数据
class RecursiveSolver {
public:
    RecursiveSolver(shared_ptr<Game> game, CFRParam& cfr_param, SolverParam& param, shared_ptr<ValueNet> net, int seed)
    :game(game), solver(game, cfr_param, param, net), curr(nullptr), max_iter(param.max_iter), engine(seed), generator(0, max_iter) {
        init_prob = HandVector2::Constant(N_CARD, N_PLAYER, 0);
    }
    void step() {
        init_prob = game->init_prob;
        curr = game->root;
        int iter = 0;
        bool skip_sample = false;
        while(!curr->is_leaf()) {
            // 归一化
            Array<double, 1, N_PLAYER> prob_sum = init_prob.colwise().sum();
            if(prob_sum(0) == 0 || prob_sum(1) == 0) break;
            init_prob.rowwise() /= prob_sum;
            solver.set_subtree_data(curr, init_prob, 1);
            int sample_iter = generator(engine);
            // std::cout << "init_prob:\n" << init_prob << std::endl << sample_iter << std::endl;
            for(iter = 0; iter < sample_iter; iter++) {
                solver.step();
            }
            sample_path.clear();
            if(curr->round == N_ROUND-1) skip_sample = true;// 最后一轮不需要采样
            else sample_leaf();
            for(; iter < max_iter; iter++) {
                solver.step();
            }
            solver.add_training_data();
            for(auto& [node, prob] : sample_path) {
                solver.add_training_data(node, prob);
            }
            if(skip_sample) break;
        }
    }
private:
    int max_iter;
    shared_ptr<Game> game;
    LimitSolver solver;
    shared_ptr<Node> curr;
    vector<pair<shared_ptr<Node>,HandVector2>> sample_path;
    HandVector2 init_prob;
    std::mt19937 engine;
    uniform_int_distribution<int> generator;
    float epsilon = 0.25;

    int sample_from_root() {
        HandSquare& mask = game->hand_mask[curr->board];
        // at::Tensor prob = init_prob[0].reshape({N_CARD,1}) * init_prob[1].reshape({1,N_CARD}) * mask;
        // // std::cout << "prob:\n" << prob << std::endl;
        // auto acc = prob.accessor<double, 2>();
        const int n = N_CARD * N_CARD;
        double weight[n] {0};
        for(int i = 0; i < N_CARD; i++) {
            for(int j = 0; j < N_CARD; j++) {
                weight[i*N_CARD+j] = init_prob(i, 0) * init_prob(j, 1) * mask(i, j);
            }
        }
        return discrete_distribution<int>(weight, weight+n)(engine);
    }
    void sample_leaf() {
        // 两种实现方式
        // 1.根节点root中采样某个h,之后一直以h为条件采样动作
        // 2.根节点root中采样某个h,之后采样某个动作a,求出root->a之后的h概率分布,再以该分布采样h,接着采样动作,以此类推
        // 两种方式采样出某个动作序列a1,a2,...,an的概率相等,所以等价
        // 区别:方式1只需要采样一次h,方式2每次动作前都需要采样h
        // 每次动作的概率分布只跟信息集有关,所以只需要根据信息集的分布采样信息集,然后采样动作
        const BFSTree& bfs_tree = solver.get_tree();
        const Strategy& sampling_strategy = solver.get_sampling_strategy();
        const Strategy& propogation_strategy = solver.get_belief_propogation_strategy();
        int idx = 0, player = uniform_int_distribution(0, 1)(engine);
        uniform_real_distribution<float> distribution(0, 1);
        HandVector2 sample_belief = init_prob;
        int h = sample_from_root();
        int hand[N_PLAYER] = {h/N_CARD, h%N_CARD};
        // std::cout << "hand:" << hand[0] << ',' << hand[1] << std::endl;
        sample_path.clear();
        bool is_chance = curr->type == CHANCE_NODE;
        if(is_chance) sample_path.emplace_back(curr, init_prob);
        while(!bfs_tree[idx].is_leaf() || is_chance) {
            float c = distribution(engine);
            int act = 0, n_act = curr->children.size();
            int node_p = curr->player;
            // 采样动作
            if(is_chance) {
                vector<double> weight(N_CARD, 1);
                weight[hand[0]] = weight[hand[1]] = 0;
                act = discrete_distribution(weight.begin(), weight.end())(engine);
            }
            else if((node_p == player && c < epsilon) /*|| type == CHANCE_NODE*/) {// 均匀采样
                act = uniform_int_distribution(0, n_act-1)(engine);
            }
            else {
                // 先采样hand
                // double* hand_prob = sample_belief.col(node_p).data();// 这里的分布不一定等于信息集的分布
                // int hand = discrete_distribution(hand_prob, hand_prob+N_CARD)(engine);
                ActVector act_prob = sampling_strategy[idx].row(hand[node_p]);// 复制
                double* ptr = act_prob.data();
                act = discrete_distribution(ptr, ptr+n_act)(engine);
                // at::Tensor prob = sampling_strategy[idx].select(1, hand[node_p]);
                // // std::cout << "sampling_strategy:\n" << prob << std::endl;
                // double* ptr = prob.data_ptr<double>();
                // vector<double> act_prob(n_act);
                // for(int i = 0; i < n_act; i++, ptr += N_CARD) act_prob[i] = *ptr;
                // act = discrete_distribution(act_prob.begin(), act_prob.end())(engine);
            }
            // 更新belief
            if(is_chance) {
                sample_belief.row(act) = 0;
                init_prob.row(act) = 0;
            }
            else {
                sample_belief.col(node_p) *= sampling_strategy[idx].col(act);
                init_prob.col(node_p) *= propogation_strategy[idx].col(act);
            }
            idx = bfs_tree[idx].child_begin + act;
            curr = curr->children[act];
            bool pre_is_chance = is_chance;
            is_chance = curr->type == CHANCE_NODE;
            if(is_chance) sample_path.emplace_back(curr, init_prob);
            if(pre_is_chance) break;
        }
    }
};

// 采样得到叶子节点,递归地求解,最终得到整个博弈树的策略
class FullStrategySolver {
public:
    FullStrategySolver(shared_ptr<Game> game, CFRParam& cfr_param, SolverParam& param, shared_ptr<ValueNet> net, bool sampling_iter, bool use_sampling_strategy)
    :game(game), solver(game, cfr_param, param, net), max_iter(param.max_iter), sampling_iter(sampling_iter), use_sampling_strategy(use_sampling_strategy) {
        game->builder.dfs_to_bfs(game->root, bfs_tree);
        reset(1);
        if(sampling_iter) {
            vector<double> weight(max_iter);
            for(int i = 0; i < max_iter; i++) weight[i] = i + 1;
            generator = discrete_distribution<int>(weight.begin(), weight.end());
        }
    }
    void reset(int seed) {
        engine.seed(seed);
        int size = bfs_tree.size();
        strategy.resize(size);// 初始化为均匀分布
        for(int i = 0; i < size; i++) {
            if(bfs_tree[i].is_leaf() || bfs_tree[i].node->type != ACTION_NODE) continue;
            int n_act = bfs_tree[i].node->children.size();
            strategy[i] = HandVectorX::Constant(N_CARD, n_act, (double)1/n_act);
        }
    }
    const Strategy& get_full_strategy() {
        recursive_solving(game->init_prob, 0);
        return strategy;
    }
    
private:
    bool use_sampling_strategy, sampling_iter;
    int max_iter;
    shared_ptr<Game> game;
    LimitSolver solver;
    BFSTree bfs_tree;
    mt19937 engine;
    discrete_distribution<int> generator;
    Strategy strategy;

    void recursive_solving(HandVector2& reach_prob, int idx = 0) {
        solver.set_subtree_data(bfs_tree[idx].node, reach_prob, 1);// 博弈树展开至随机节点
        int sample_iter = sampling_iter ? generator(engine) + 1 : max_iter;
        solver.multi_step(sample_iter);
        const Strategy& part_strategy = 
            use_sampling_strategy ? solver.get_sampling_strategy() : solver.get_avg_strategy();
        const Strategy& belief_strategy = 
            use_sampling_strategy ? solver.get_belief_propogation_strategy() : solver.get_avg_strategy();
        const BFSTree& part_bfs_tree = solver.get_tree();
        int size = part_bfs_tree.size();
        // 复制策略,采样belief
        deque<tuple<int, int, HandVector2>> dq, leaf_dq;// full_id, part_id, reach_prob
        dq.emplace_back(idx, 0, reach_prob);
        while(!dq.empty()) {// bfs
            auto [full_id, part_id, prob] = std::move(dq.front());
            dq.pop_front();
            BFSNode& bfs_node = bfs_tree[full_id];
            if(bfs_node.node.get() != part_bfs_tree[part_id].node.get()) throw runtime_error("sub tree error");
            if(part_bfs_tree[part_id].is_leaf()) {
                if(!bfs_node.node->is_leaf()) leaf_dq.emplace_back(full_id, part_id, prob);
                continue;
            }
            strategy[full_id] = part_strategy[part_id];
            int child_begin = bfs_node.child_begin, part_child_begin = part_bfs_tree[part_id].child_begin;
            int n_act = bfs_node.child_end - child_begin, player = bfs_node.node->player;
            bool is_chance = bfs_node.node->type == CHANCE_NODE;
            for(int i = 0; i < n_act; i++) {
                HandVector2 new_prob = prob;
                if(is_chance) new_prob.row(i) = 0;
                else new_prob.col(player) *= part_strategy[part_id].col(i);
                dq.emplace_back(child_begin+i, part_child_begin+i, std::move(new_prob));
            }
        }
        for(auto& [full_id, _, prob] : leaf_dq) {// 求解subgame的每个叶子节点
            Array<double, 1, N_PLAYER> prob_sum = prob.colwise().sum();
            if(prob_sum(0) == 0 || prob_sum(1) == 0) continue;
            prob.rowwise() /= prob_sum;// 归一化
            recursive_solving(prob, full_id);
        }
    }
};

#endif // _RECURSIVE_SOLVER_H_
