#if !defined(_LIMIT_SOLVER_H_)
#define _LIMIT_SOLVER_H_

#include <thread>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <algorithm>
#include "util.h"
#include "game.h"
#include "cfr.h"
#include "net_interface.h"

#define AVG_STRATEGY 1
#define BEST_CFV 2

using Strategy = vector<HandVectorX>;

struct SolverParam {
    int n_thread = N_CARD, max_iter = 1000, print_interval = 10;
    double accuracy = 0.001;
};

inline HandVector other_sum(const HandSquare& mat, const HandVector2& reach_prob, int player) {
    return mat.matrix() * reach_prob.col(1-player).matrix();// N_PLAYER == 2
}

class LimitSolver {
public:
    LimitSolver(shared_ptr<Game> game, CFRParam& cfr_param, SolverParam& param, shared_ptr<ValueNet> net = nullptr)
    :builder(game->builder), hand_mask(game->hand_mask), call_value(game->call_value), cfr_param(cfr_param), net(net) {
        if(hand_mask.size() != N_CARD + 1) throw runtime_error("hand mask matrix size error");
        if(call_value.size() != N_CARD) throw runtime_error("call value matrix size error");
        n_thread = param.n_thread;
        if(n_thread <= 0) n_thread = omp_get_num_procs();
        if(n_thread > N_CARD) n_thread = N_CARD;
        // if(n_thread > 1) start_thread_pool();
        omp_set_num_threads(n_thread);
        max_iter = param.max_iter;
        print_interval = param.print_interval;
        accuracy = param.accuracy;
        root_cfv_mean = HandVector2::Constant(N_CARD, N_PLAYER, 0);
        chance_prob = game->chance_prob;
        int size = hand_mask.size();
        tensor_mask.resize(size);
        int n = N_CARD * N_CARD;
        for(int i = 0; i < size; i++) {
            tensor_mask[i] = torch::zeros({N_CARD,N_CARD}, torch::kF64);
            std::copy_n(hand_mask[i].data(), n, tensor_mask[i].data_ptr<double>());
            // std::cout << hand_mask[i] << std::endl;
            // std::cout << tensor_mask[i] << std::endl;
        }
    }
    /*~LimitSolver() {
        stop = true;
        task_cv.notify_all();
        for(std::thread& t : threads) t.join();
    }*/
    void set_subtree_data(shared_ptr<Node> root, const HandVector2& init_prob, int round_limit = INT_MAX, int depth_limit = INT_MAX) {
        if(!root) throw runtime_error("root node is null");
        this->root = root;
        this->init_prob = init_prob;
        norm = 0;
        HandSquare& mask = hand_mask[root->board];
        for(int i = 0; i < N_CARD; i++) {
            for(int j = 0; j < N_CARD; j++) {
                norm += init_prob(i, 0) * init_prob(j, 1) * mask(i, j);
            }
        }
        // 根节点,如果对方到达概率为0,那么己方cfv均为0,无法更新regret
        // 如果己方到达概率为0,那么累计策略不会变化,从而均值策略不变
        if(norm == 0) throw runtime_error("root node is unreachable (reach probability is 0)");
        // this->round_limit = round_limit;
        // this->depth_limit = depth_limit;
        builder.dfs_to_bfs(this->root, bfs_tree, round_limit, depth_limit);
        int size = bfs_tree.size();
        cfr_data.assign(size, nullptr);
        curr_strategy.clear();
        curr_strategy.resize(size);// 初始化为均匀分布
        shared_ptr<CFRData> data = nullptr;
        shared_ptr<Node> node = nullptr;
        for(int i = 0; i < size; i++) {
            if(bfs_tree[i].is_leaf() || bfs_tree[i].node->type != ACTION_NODE) continue;
            node = bfs_tree[i].node;
            int n_act = node->children.size();
            curr_strategy[i] = HandVectorX::Constant(N_CARD, n_act, (double)1/n_act);
            if(cfr_param.discount) data = make_shared<DCFRData>(n_act, node->board, cfr_param);
            else data = make_shared<CFRData>(n_act, node->board, cfr_param);
            cfr_data[i] = data;
        }
        avg_strategy = curr_strategy;// 初始化为均匀分布
        root_cfv_mean.fill(0);
        iter = 0;
    }
    void step() {
        double cfv_coef = cfr_param.linear ? (double)2/(iter+2) : (double)1/(iter+1);
        for(int player = 0; player < N_PLAYER; player++) {
            HandVector root_cfv = cfr(player, root, init_prob, 0);
            root_cfv_mean.col(player) += cfv_coef * (root_cfv - root_cfv_mean.col(player));
        }
        iter++;
    }
    void multi_step(int iters) {
        if(iters == -1) iters = max_iter;
        for(iter = 0; iter < iters; ) step();
    }
    virtual double train() {
        Timer timer;
        double e_threshold = (root->pots[0] + root->pots[1]) * accuracy / 100;
        vector<double> e = exploitability();
        double res = (e[0] + e[1]) / 2;
        printf("% 4d\t%.6f\t%.6f\t%.6f\t%gs\n", 0, res, e[0], e[1], timer.duration()/1e6);
        if(res < e_threshold) return res;
        for(iter = 0; iter < max_iter; ) {
            step();
            if(print_interval != 0 && iter % print_interval == 0) {
                e = exploitability();res = (e[0] + e[1]) / 2;
                printf("% 4d\t%.6f\t%.6f\t%.6f\t%gs\n", iter, res, e[0], e[1], timer.duration()/1e6);
                if(res < e_threshold) return res;
            }
        }
        e = exploitability();res = (e[0] + e[1]) / 2;
        printf("exploitability:%.6f\t%.6f\t%.6f\t%gs\n", res, e[0], e[1], timer.duration()/1e6);
        return res;
    }
    vector<double> exploitability(Strategy& outer_strategy) {
        if(outer_strategy.size() != avg_strategy.size()) throw runtime_error("strategy size error");
        avg_strategy.swap(outer_strategy);// 多线程不安全
        vector<double> e = exploitability(false);
        avg_strategy.swap(outer_strategy);
        return e;
    }
    vector<double> exploitability(bool update = true) {
        task = AVG_STRATEGY | BEST_CFV;
        if(update) get_avg_strategy();
        HandVector2 root_cfv = cfv(root, init_prob, 0);
        vector<double> value(N_PLAYER);
        for(int player = 0; player < N_PLAYER; player++) {
            value[player] = (root_cfv.col(player) * init_prob.col(player)).sum() / norm;
        }
        return value;
    }
    HandVector get_root_value(int player, bool cfv = true) {
        if(cfv) return root_cfv_mean.col(player);
        HandVector opp_prob = other_sum(hand_mask[root->board], init_prob, player);
        HandVector ev = root_cfv_mean.col(player) / opp_prob;
        for(int i = 0; i < N_CARD; i++) if(opp_prob(i) == 0) ev(i) = 0;
        return ev;
    }
    at::Tensor get_feature(shared_ptr<Node> node, const HandVector2& reach_prob) {
        if(node->pots[0] != node->pots[1]) throw runtime_error("the bets of both player must be equal");
        int act_player = node->player, query_size = net->query_size();
        at::Tensor feature = torch::zeros({query_size}, torch::kF32);
        feature[1] = node->type != CHANCE_NODE ? act_player : 0;
        feature[2] = node->pots[0];// 双方累计投注额相等
        if(node->board != N_CARD) feature[3+node->board] = 1;// 公共牌特征onehot
        Array<double, 1, N_PLAYER> prob_sum = reach_prob.colwise().sum();
        HandVector2 prob_feature = reach_prob.rowwise() / prob_sum;
        double* src_ptr = prob_feature.data();
        float* dst_ptr = feature.data_ptr<float>() + (3 + N_CARD);
        for(int player = 0; player < N_PLAYER; player++) {
            if(prob_sum(player) != 0) {
                std::copy_n(src_ptr, N_CARD, dst_ptr);
            }
            src_ptr += N_CARD;
            dst_ptr += N_CARD;
        }
        // std::cout << "reach_prob:\n" << reach_prob << std::endl;
        // std::cout << "prob_feature:\n" << prob_feature << std::endl;
        // std::cout << "feature:\n" << feature << std::endl;
        return feature;
    }
    void add_training_data() {
        if(root->type != ACTION_NODE || root->player != 0 || root->pots[0] != root->pots[1]) {
            throw runtime_error("only support node at the start of betting round");
        }
        at::Tensor query = get_feature(root, init_prob);
        for(int player = 0; player < N_PLAYER; player++) {
            HandVector ev = get_root_value(player, false);
            at::Tensor node_ev = torch::zeros({N_CARD}, torch::kF32);
            std::copy_n(ev.data(), N_CARD, node_ev.data_ptr<float>());
            query[0] = player;
            net->add_training_data(query.clone(), node_ev);
        }
    }
    void add_training_data(shared_ptr<Node> node, const HandVector2& reach_prob) {
        if(node->type != CHANCE_NODE) throw runtime_error("only support node at the end of betting round");
        int next_round_player = 0, query_size = net->query_size();
        at::Tensor query = get_feature(node, reach_prob);
        at::Tensor child_feature = torch::zeros({query_size}, torch::kF32);
        // query[1] = next_round_player;
        child_feature[1] = next_round_player;
        child_feature[2] = node->pots[0];// 双方累计投注额相等
        at::Tensor child_query = child_feature.repeat({N_CARD,1});
        auto acc = child_query.accessor<float, 2>();
        int offset = 3;
        for(int i = 0; i < N_CARD; i++) acc[i][offset+i] = 1;// 公共牌特征onehot
        offset += N_CARD;
        Array<double, 1, N_PLAYER> prob_sum = reach_prob.colwise().sum();
        for(int player = 0; player < N_PLAYER; player++) {// 每个玩家的归一化到达概率
            double sum = prob_sum(player);
            if(sum == 0) continue;
            for(int i = 0; i < N_CARD; i++) {
                HandVector prob = reach_prob.col(player);
                double temp_sum = sum - prob(i);
                prob(i) = 0;
                if(temp_sum == 0) continue;
                prob /= temp_sum;
                double* ptr = prob.data();
                for(int j = 0; j < N_CARD; j++, ptr++) acc[i][offset+j] = *ptr;
            }
            offset += N_CARD;
        }
        at::Tensor opp_prob = torch::zeros({N_CARD}, torch::kF64);
        double* dst_ptr = opp_prob.data_ptr<double>();
        for(int player = 0; player < N_PLAYER; player++) {// 双方ev都加入训练数据
            std::copy_n(reach_prob.col(1-player).data(), N_CARD, dst_ptr);
            // std::cout << opp_prob << std::endl;
            at::Tensor hand_prob = tensor_mask[N_CARD].matmul(opp_prob);// [n_hand]
            child_query.select(1, 0) = player;
            query[0] = player;
            // std::cout << query << std::endl;
            // std::cout << child_query << std::endl;
            at::Tensor child_ev = net->compute_value(child_query);// [n_act,n_hand]
            at::Tensor ev = torch::zeros({N_CARD}, torch::kF64);// cfv:[n_hand]
            for(int i = 0; i < N_CARD; i++) {// act
                ev += tensor_mask[i].matmul(opp_prob) * child_ev[i];
            }
            ev *= chance_prob;
            ev /= hand_prob;// cfv --> ev
            ev.masked_fill_(hand_prob.eq(0), 0);
            net->add_training_data(query.clone(), ev.toType(torch::kF32));
        }
    }
    const BFSTree& get_tree() {
        return bfs_tree;
    }
    const Strategy& get_avg_strategy() {
        int size = bfs_tree.size();
        for(int i = 0; i < size; i++) {
            if(bfs_tree[i].is_leaf() || bfs_tree[i].node->type != ACTION_NODE) continue;
            avg_strategy[i] = cfr_data[i]->average_strategy();
        }
        return avg_strategy;
    }
    virtual const Strategy& get_sampling_strategy() {
        return curr_strategy;
    }
    virtual const Strategy& get_belief_propogation_strategy() {
        return curr_strategy;
    }

protected:
    int round_limit = INT_MAX, depth_limit = INT_MAX;
    int iter = 0;
    int task;
    int max_iter;
    int print_interval;
    double accuracy;
    double chance_prob;
    shared_ptr<Node> root = nullptr;
    TreeBuilder& builder;
    CFRParam cfr_param;
    HandVector2 init_prob;
    double norm;
    vector<HandSquare>& call_value;// 已经屏蔽了所有冲突的情况(ev为0)
    vector<HandSquare>& hand_mask;
    vector<at::Tensor> tensor_mask;
    vector<shared_ptr<CFRData>> cfr_data;
    Strategy curr_strategy, avg_strategy;
    BFSTree bfs_tree;
    HandVector2 root_cfv_mean;
    shared_ptr<ValueNet> net = nullptr;

    void set_data(shared_ptr<Node> node, int depth = 0, int idx = 0) {
        int round = node->round, type = node->type;
        if(type == FOLD_NODE || type == SHOWDOWN_NODE) return;
        int n_act = node->children.size();
        if(type == ACTION_NODE) {
            int board = node->board;
            shared_ptr<CFRData> data;
            if(cfr_param.discount) data = make_shared<DCFRData>(n_act, board, cfr_param);
            else data = make_shared<CFRData>(n_act, board, cfr_param);
            cfr_data.push_back(data);
            for(int i = 0; i < n_act; i++) set_data(node->children[i], depth+1);
        }
        else if(type == CHANCE_NODE) {
            for(int i = 0; i < n_act; i++) set_data(node->children[i], depth+1);
        }
        else throw runtime_error("unknown node type");
    }
    HandVector fold_node_cfv(int player, shared_ptr<Node> node, const HandVector2& reach_prob) {
        int fold_p = 1 - node->player;
        double ev = node->pots[fold_p];
        if(player == fold_p) ev = -ev;
        HandVector node_cfv = other_sum(hand_mask[node->board], reach_prob, player);
        node_cfv *= ev;
        return node_cfv;
    }
    HandVector2 fold_node_cfv(shared_ptr<Node> node, const HandVector2& reach_prob) {
        int fold_p = 1 - node->player;
        double ev = node->pots[fold_p];
        HandVector2 node_cfv = HandVector2::Constant(N_CARD, N_PLAYER, 0);
        HandSquare& mask = hand_mask[node->board];
        for(int player = 0; player < N_PLAYER; player++) {
            node_cfv.col(player) = other_sum(mask, reach_prob, player);// 对方到达概率
        }
        node_cfv.col(node->player) *= ev;
        node_cfv.col(fold_p) *= -ev;
        return node_cfv;
    }
    HandVector showdown_node_cfv(int player, shared_ptr<Node> node, const HandVector2& reach_prob) {
        HandVector node_cfv = other_sum(call_value[node->board], reach_prob, player);
        node_cfv *= node->pots[0];
        return node_cfv;
    }
    HandVector2 showdown_node_cfv(shared_ptr<Node> node, const HandVector2& reach_prob) {
        HandVector2 node_cfv = HandVector2::Constant(N_CARD, N_PLAYER, 0);
        HandSquare& value = call_value[node->board];
        for(int player = 0; player < N_PLAYER; player++) {
            node_cfv.col(player) = other_sum(value, reach_prob, player);// 对方到达概率
        }
        node_cfv *= node->pots[0];
        return node_cfv;
    }
    // 给定对方的到达概率,返回player玩家每种手牌的cfv
    HandVector cfr(int player, shared_ptr<Node> node, const HandVector2& reach_prob, int idx = 0) {
        // if(node.get() != bfs_tree[idx].node.get()) {
        //     throw runtime_error("bfs tree error");
        // }
        int type = node->type;
        if(type == FOLD_NODE) {
            return fold_node_cfv(player, node, reach_prob);
        }
        else if(type == SHOWDOWN_NODE) {
            return showdown_node_cfv(player, node, reach_prob);
        }
        else if(bfs_tree[idx].is_leaf()) {
            if(net == nullptr) throw runtime_error("need value net");
            if(net->return_cfv()) return net->compute_value(node, reach_prob, player);
            if(node->type != CHANCE_NODE) throw runtime_error("only query net at chance node");
            at::Tensor query = get_feature(node, reach_prob);
            query[0] = player;
            at::Tensor node_ev = net->compute_value(query.reshape({1,-1})).flatten();
            HandVector node_cfv = HandVector::Constant(N_CARD, 1, 0);
            // std::cout << "node_ev:\n" << node_ev << std::endl;
            // float* ptr = node_ev.data_ptr<float>();
            // for(int i = 0; i < N_CARD; i++, ptr++) std::cout << *ptr << ' '; std::cout << std::endl;
            std::copy_n(node_ev.data_ptr<float>(), N_CARD, node_cfv.data());
            node_cfv *= other_sum(hand_mask[node->board], reach_prob, player);
            return node_cfv;
        }
        else if(type == ACTION_NODE) {
            return action_node_cfr(player, node, reach_prob, idx);
        }
        else if(type == CHANCE_NODE) {
            return chance_node_cfr(player, node, reach_prob, idx);
        }
        else throw runtime_error("unknown node type");
    }
    HandVector2 cfv(shared_ptr<Node> node, const HandVector2& reach_prob, int idx = 0) {
        // if(node.get() != bfs_tree[idx].node.get()) {
        //     throw runtime_error("bfs tree error");
        // }
        int type = node->type;
        if(type == FOLD_NODE) {
            return fold_node_cfv(node, reach_prob);
        }
        else if(type == SHOWDOWN_NODE) {
            return showdown_node_cfv(node, reach_prob);
        }
        else if(bfs_tree[idx].is_leaf()) {
            if(net == nullptr) throw runtime_error("need value net");
            if(net->return_cfv()) return net->compute_value(node, reach_prob);
            if(node->type != CHANCE_NODE) throw runtime_error("only query net at chance node");
            at::Tensor query = get_feature(node, reach_prob).repeat({N_PLAYER,1});
            for(int player = 0; player < N_PLAYER; player++) query[player][0] = player;
            std::cout << query << std::endl;
            at::Tensor node_ev = net->compute_value(query);
            auto acc = node_ev.accessor<float, 2>();
            HandVector2 node_cfv = HandVector2::Constant(N_CARD, N_PLAYER, 0);
            for(int player = 0; player < N_PLAYER; player++) {
                HandVector ev = HandVector::Constant(N_CARD, 1, 0);
                for(int i = 0; i < N_CARD; i++) ev(i) = acc[player][i];
                node_cfv.col(player) = ev * other_sum(hand_mask[node->board], reach_prob, player);
            }
            return node_cfv;
        }
        else if(type == ACTION_NODE) {
            return action_node_cfv(node, reach_prob, idx);
        }
        else if(type == CHANCE_NODE) {
            return chance_node_cfv(node, reach_prob, idx);
        }
        else throw runtime_error("unknown node type");
    }
    HandVector2 action_node_cfv(shared_ptr<Node> node, const HandVector2& reach_prob, int idx) {
        int player = node->player, opp = 1 - player;
        int n_act = node->children.size(), child_begin = bfs_tree[idx].child_begin;
        // shared_ptr<CFRData> data = cfr_data[idx];
        // HandVectorX strategy = (task&AVG_STRATEGY) ? data->average_strategy() : data->curr_strategy();
        HandVectorX& strategy = (task&AVG_STRATEGY) ? avg_strategy[idx] : curr_strategy[idx];
        HandVectorX my_cfv = HandVectorX::Constant(N_CARD, n_act, 0);
        HandVectorX opp_cfv = HandVectorX::Constant(N_CARD, n_act, 0);
        HandVector2 new_prob = reach_prob;
        for(int i = 0; i < n_act; i++) {
            new_prob.col(player) = reach_prob.col(player) * strategy.col(i);
            HandVector2 value = cfv(node->children[i], new_prob, child_begin+i);
            my_cfv.col(i) = value.col(player);
            opp_cfv.col(i) = value.col(opp);
        }
        HandVector2 node_cfv = HandVector2::Constant(N_CARD, N_PLAYER, 0);
        node_cfv.col(opp) = opp_cfv.rowwise().sum();
        if(task & BEST_CFV) node_cfv.col(player) = my_cfv.rowwise().maxCoeff();
        else node_cfv.col(player) = (my_cfv * strategy).rowwise().sum();
        return node_cfv;
    }
    HandVector2 chance_node_cfv(shared_ptr<Node> node, const HandVector2& reach_prob, int idx) {
        HandSquare p0_cfv = HandSquare::Constant(N_CARD, N_CARD, 0);
        HandSquare p1_cfv = HandSquare::Constant(N_CARD, N_CARD, 0);
        int child_begin = bfs_tree[idx].child_begin;
        /*if(!multi_thread && n_thread > 1) {
            chance_node_parallel(reach_prob, child_begin);
            for(int i = 0; i < N_CARD; i++) {
                // std::cout << cfv_result[i] << std::endl;
                p0_cfv.col(i) = cfv_result[i].col(0);
                p1_cfv.col(i) = cfv_result[i].col(1);
            }
        }
        else*/ {
            #pragma omp parallel for
            for(int i = 0; i < N_CARD; i++) {
                HandVector2 new_prob = reach_prob * chance_prob;
                new_prob.row(i) = 0;
                HandVector2 value = cfv(node->children[i], new_prob, child_begin+i);
                p0_cfv.col(i) = value.col(0);
                p1_cfv.col(i) = value.col(1);
            }
        }
        HandVector2 node_cfv = HandVector2::Constant(N_CARD, N_PLAYER, 0);
        node_cfv.col(0) = p0_cfv.rowwise().sum();
        node_cfv.col(1) = p1_cfv.rowwise().sum();
        return node_cfv;
    }
    HandVector action_node_cfr(int player, shared_ptr<Node> node, const HandVector2& reach_prob, int idx) {
        int node_player = node->player;
        int n_act = node->children.size(), child_begin = bfs_tree[idx].child_begin;
        shared_ptr<CFRData> data = cfr_data[idx];
        // HandVectorX strategy = data->curr_strategy();
        HandVectorX& strategy = curr_strategy[idx];
        HandVector2 new_prob = reach_prob;
        HandVectorX child_cfv = HandVectorX::Constant(N_CARD, n_act, 0);
        for(int i = 0; i < n_act; i++) {
            new_prob.col(node_player) = reach_prob.col(node_player) * strategy.col(i);
            child_cfv.col(i) = cfr(player, node->children[i], new_prob, child_begin+i);
        }
        if(player != node_player) return child_cfv.rowwise().sum();
        HandVector node_cfv;
        if(cfr_param.hedge) {
            node_cfv = (child_cfv * strategy).rowwise().sum();
            HandVector ev = other_sum(hand_mask[node->board], reach_prob, player);
            for(int i = 0; i < N_CARD; i++) if(ev(i) != 0) ev(i) = node_cfv(i) / ev(i);
            HandVectorX regret = child_cfv.colwise() - node_cfv;
            HandVectorX weighted_strategy = strategy.colwise() * reach_prob.col(player);
            data->update(regret, weighted_strategy, ev, iter+1);
        }
        else {
            node_cfv = (child_cfv * strategy).rowwise().sum();
            HandVector my_prob = reach_prob.col(player);
            data->update(node_cfv, child_cfv, strategy, my_prob, iter+1);
        }
        strategy = data->curr_strategy();// 更新策略,为下一次迭代做准备
        return node_cfv;
    }
    HandVector chance_node_cfr(int player, shared_ptr<Node> node, const HandVector2& reach_prob, int idx) {
        int opp = 1 - player, child_begin = bfs_tree[idx].child_begin;
        HandSquare node_cfv = HandSquare::Constant(N_CARD, N_CARD, 0);
        /*if(!multi_thread && n_thread > 1) {
            chance_node_parallel(reach_prob, child_begin, player);
            return cfr_result.rowwise().sum();
        }*/
        #pragma omp parallel for
        for(int i = 0; i < N_CARD; i++) {
            HandVector2 new_prob = reach_prob;
            new_prob.row(i) = 0;
            new_prob.col(opp) *= chance_prob;// 发牌概率归于对方
            node_cfv.col(i) = cfr(player, node->children[i], new_prob, child_begin+i);
        }
        return node_cfv.rowwise().sum();
    }

private:
    int n_thread;
    /*struct TaskArg {
        TaskArg(int player, int act, const HandVector2& reach_prob, int idx)
            :player(player), act(act), reach_prob(reach_prob), idx(idx) {}
        int player, act, idx;
        const HandVector2& reach_prob;
    };
    std::atomic_bool stop = false;
    bool multi_thread = true;// 防止递归任务入队,只在最顶层的随机节点采用多线程
    int n_task = 0;
    std::deque<TaskArg> thread_tasks;
    std::mutex task_lock, done_lock;
    std::condition_variable task_cv, done_cv;
    vector<std::thread> threads;
    vector<HandVector2> cfv_result;
    HandSquare cfr_result;
    void start_thread_pool() {
        multi_thread = false;
        cfr_result = HandSquare::Constant(N_CARD, N_CARD, 0);
        cfv_result.resize(N_CARD);
        auto loop = [this]() {
            while(!stop) {
                std::unique_lock<std::mutex> lk(task_lock);
                task_cv.wait(lk, [this](){return stop || !thread_tasks.empty();});
                if(stop) return;
                TaskArg ta = std::move(thread_tasks.front());
                thread_tasks.pop_front();
                lk.unlock();// 释放锁
                if(ta.player != -1) {
                    HandVector2 new_prob = ta.reach_prob;
                    new_prob.row(ta.act) = 0;
                    new_prob.col(1-ta.player) *= chance_prob;
                    cfr_result.col(ta.act) = cfr(ta.player, bfs_tree[ta.idx].node, new_prob, ta.idx);
                }
                else {
                    HandVector2 new_prob = ta.reach_prob * chance_prob;
                    new_prob.row(ta.act) = 0;
                    cfv_result[ta.act] = cfv(bfs_tree[ta.idx].node, new_prob, ta.idx);
                }
                std::lock_guard<std::mutex> lk1(done_lock);
                if((--n_task) == 0) done_cv.notify_one();
            }
        };
        for(int i = 0; i < n_thread; i++) threads.emplace_back(loop);
    }
    void chance_node_parallel(const HandVector2& reach_prob, int child_begin, int player = -1) {
        multi_thread = true;
        task_lock.lock();
        for(int i = 0; i < N_CARD; i++) thread_tasks.emplace_back(player, i, reach_prob, child_begin+i);
        n_task = N_CARD;
        task_lock.unlock();
        task_cv.notify_all();
        std::unique_lock<std::mutex> lk(done_lock);
        done_cv.wait(lk, [this](){return n_task == 0;});
        multi_thread = false;
    }*/
};

#endif // _LIMIT_SOLVER_H_
