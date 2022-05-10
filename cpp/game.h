#if !defined(_GAME_H_)
#define _GAME_H_

#define EIGEN_USE_MKL_ALL
#include <omp.h>
#include <vector>
#include <deque>
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/Dense>
using namespace Eigen;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::swap;
using std::runtime_error;
using std::deque;

#define N_PLAYER 2
#define N_CARD 6
#define N_SUIT 2
#define N_RANK 3
#define N_ROUND 2
#define card2rank(x) ((x) >> 1)
#define card2suit(x) ((x) & 1)

typedef Array<double, N_CARD, 1> HandVector;
typedef Array<double, N_CARD, 2> HandVector2;
typedef Array<double, N_CARD, -1> HandVectorX;
typedef Array<double, N_CARD, N_CARD> HandSquare;
typedef Array<double, 1, -1> ActVector;

#define ACTION_NODE 0
#define CHANCE_NODE 1
#define FOLD_NODE 2
#define SHOWDOWN_NODE 3

struct Node {
    Node(int player, int round, int p0_pot, int p1_pot, int type, int board = N_CARD)
    :player(player), round(round), type(type), board(board) {
        pots[0] = p0_pot;
        pots[1] = p1_pot;
    }
    int player = 0;
    int round = 0;
    int board = N_CARD;
    int pots[N_PLAYER];
    int type = ACTION_NODE;
    vector<shared_ptr<Node>> children;// 动作节点的子节点
    bool is_leaf() {
        return type == FOLD_NODE || type == SHOWDOWN_NODE;
    }
};

struct Rule {
    bool stop_at_chance = false;
    int init_pot[N_PLAYER] = {1,1};
    int raise[N_ROUND] = {2,4};
    int max_bet_num = 3;
    int get_stack() {
        int stack = init_pot[0];
        for(int i = 0; i < N_ROUND; i++) {
            stack += raise[i] * max_bet_num;
        }
        return stack;
    }
};

struct BFSNode {
    BFSNode(shared_ptr<Node> node, int depth):node(node), depth(depth) {}
    shared_ptr<Node> node = nullptr;
    int child_begin = 0;
    int child_end = 0;
    int depth = 0;
    bool is_leaf() const {
        return child_begin == child_end;
    }
};
using BFSTree = vector<BFSNode>;

struct TreeBuilder {
    TreeBuilder(Rule& rule):rule(rule) {}
    Rule& rule;
    
    void build_tree(shared_ptr<Node> node, int bet_num = 0) {
        if(node == nullptr) return;
        int type = node->type, round = node->round;
        if(type == FOLD_NODE || type == SHOWDOWN_NODE) return;
        int* pots = node->pots;
        int player = node->player, opp = 1 - player;
        int my_pot = pots[player], opp_pot = pots[opp];
        if(my_pot > opp_pot) throw runtime_error("tree error");
        shared_ptr<Node> child = nullptr;
        if(type == CHANCE_NODE) {
            if(rule.stop_at_chance) return;
            if(pots[0] != pots[1] || round != N_ROUND - 1) throw runtime_error("tree error");
            for(int i = 0; i < N_CARD; i++) {
                child = make_shared<Node>(0, round, pots[0], pots[1], ACTION_NODE, i);
                node->children.push_back(child);
                build_tree(child, 0);
            }
            return;
        }
        if(opp_pot > my_pot) {// fold
            child = make_shared<Node>(opp, round, pots[0], pots[1], FOLD_NODE, node->board);
            node->children.push_back(child);
        }
        // call
        if(player == 0 && bet_num == 0) {// 初始call
            if(my_pot != opp_pot) throw runtime_error("tree error");
            child = make_shared<Node>(opp, round, opp_pot, opp_pot, ACTION_NODE, node->board);
        }
        else if(round != N_ROUND - 1) {// 不是最后一轮
            child = make_shared<Node>(opp, round+1, opp_pot, opp_pot, CHANCE_NODE, node->board);
        }
        else child = make_shared<Node>(opp, round, opp_pot, opp_pot, SHOWDOWN_NODE, node->board);
        build_tree(child, bet_num);
        node->children.push_back(child);
        // raise
        if(bet_num >= rule.max_bet_num) return;
        int p0_pot = opp_pot + rule.raise[round], p1_pot = opp_pot;
        if(player != 0) swap(p0_pot, p1_pot);
        child = make_shared<Node>(opp, round, p0_pot, p1_pot, ACTION_NODE, node->board);
        build_tree(child, bet_num+1);
        node->children.push_back(child);
    }
    // round_limit,depth_limit均为增量
    void dfs_to_bfs(shared_ptr<Node> node, BFSTree& bfs_tree, int round_limit = INT_MAX, int depth_limit = INT_MAX) {
        bfs_tree.clear();
        deque<shared_ptr<Node>> dq;
        dq.push_back(node);
        bfs_tree.emplace_back(node, 0);
        int idx = 0, init_round = node->round;
        while(!dq.empty()) {
            node = dq.front();dq.pop_front();
            if(node.get() != bfs_tree[idx].node.get()) {
                throw runtime_error("bfs tree error");
            }
            int depth = bfs_tree[idx].depth, diff = node->round - init_round;
            bfs_tree[idx].child_begin = bfs_tree[idx].child_end = bfs_tree.size();
            if(node->is_leaf() || diff >= round_limit || depth >= depth_limit) {
                idx++;
                continue;
            }
            bfs_tree[idx].child_end += node->children.size();
            for(shared_ptr<Node> child : node->children) {
                dq.push_back(child);
                bfs_tree.emplace_back(child, depth+1);
            }
            idx++;
        }
    }
};

struct Game {
    Game(int p0_pot = 1, int p1_pot = 1, int raise0 = 2, int raise1 = 4, int max_bet_num = 3):builder(rule) {
        rule.init_pot[0] = p0_pot;
        rule.init_pot[1] = p1_pot;
        rule.raise[0] = raise0;
        rule.raise[1] = raise1;
        rule.max_bet_num = max_bet_num;
        root = make_shared<Node>(0, 0, p0_pot, p1_pot, ACTION_NODE);
        builder.build_tree(root);

        init_prob = HandVector2::Constant(N_CARD, N_PLAYER, (double)1/N_CARD);
        hand_mask.resize(N_CARD+1);
        hand_mask[N_CARD] = HandSquare::Constant(N_CARD, N_CARD, 1);
        HandSquare& base_mask = hand_mask[N_CARD];
        for(int i = 0; i < N_CARD; i++) base_mask(i,i) = 0;
        call_value.resize(N_CARD);
        for(int i = 0; i < N_CARD; i++) {
            hand_mask[i] = base_mask;
            for(int j = 0; j < N_CARD; j++) {
                hand_mask[i](i,j) = 0;
                hand_mask[i](j,i) = 0;
            }
            call_value[i] = call_value_matrix(i);
        }
        // std::cout << hand_mask[N_CARD] << std::endl;
        // for(int i = 0; i < N_CARD; i++) {
        //     std::cout << hand_mask[i] << std::endl;
        //     std::cout << call_value[i] << std::endl;
        // }
    }
    // 行玩家每种手牌对战列玩家每种手牌的收益
    HandSquare call_value_matrix(int board) {
        int board_rank = card2rank(board);
        int hand_rank[N_CARD]{ 0 };
        for(int i = 0; i < N_CARD; i++) hand_rank[i] = card2rank(i);
        HandSquare value = HandSquare::Constant(N_CARD, N_CARD, 0);
        for(int i = 0; i < N_CARD; i++) {
            if(i == board) continue;
            for(int j = 0; j < N_CARD; j++) {
                if(j == i || j == board) continue;
                if(hand_rank[i] == board_rank) value(i,j) = 1;
                else if(hand_rank[j] == board_rank) value(i,j) = -1;
                else if(hand_rank[i] > hand_rank[j]) value(i,j) = 1;
                else if(hand_rank[i] < hand_rank[j]) value(i,j) = -1;
            }
        }
        return value;
    }
    void reset(Rule& rule_) {
        rule.init_pot[0] = rule_.init_pot[0];
        rule.init_pot[1] = rule_.init_pot[1];
        rule.raise[0] = rule_.raise[0];
        rule.raise[1] = rule_.raise[1];
        rule.max_bet_num = rule_.max_bet_num;
        root = make_shared<Node>(0, 0, rule.init_pot[0], rule.init_pot[1], ACTION_NODE);
        builder.build_tree(root);
    }

    double chance_prob = (double)1 / (N_CARD - N_PLAYER);
    HandVector2 init_prob;
    vector<HandSquare> hand_mask, call_value;
    shared_ptr<Node> root = nullptr;
    Rule rule;
    TreeBuilder builder;
};

#endif // _GAME_H_
