#if !defined(_NET_INTERFACE_H_)
#define _NET_INTERFACE_H_

#include <torch/torch.h>
//#include <torch/extension.h>
#include "game.h"

class ValueNet {
public:
    virtual HandVector compute_value(shared_ptr<Node> node, const HandVector2& reach_prob, int player) {
        return HandVector::Constant(N_CARD, 0);
    }
    virtual HandVector2 compute_value(shared_ptr<Node> node, const HandVector2& reach_prob) {
        return HandVector2::Constant(N_CARD, N_PLAYER, 0);
    }
    virtual at::Tensor compute_value(shared_ptr<Node> node, const at::Tensor& reach_prob) {
        return torch::zeros({N_PLAYER,N_CARD}, torch::kF32);
    }
    virtual at::Tensor compute_value(shared_ptr<Node> node, const at::Tensor& reach_prob, int player) {
        return torch::zeros({N_CARD}, torch::kF32);
    }
    virtual at::Tensor compute_value(at::Tensor query) {
        return torch::zeros({query.size(0), N_CARD}, torch::kF32);
    }
    virtual bool return_cfv() {
        return false;
    }
    virtual void add_training_data(at::Tensor query, at::Tensor value) {}
    int query_size() {
        // 只在发牌节点和发牌节点的子节点使用神经网络
        // 也就是每轮的结束和下一轮的开始,此时双方的累计投注额相等
        // 输出对应哪个玩家+节点激活玩家+投注额+公共牌onehot+双方belief
        return 1 + 1 + 1 + N_CARD + N_PLAYER * N_CARD;
    }
};

#endif // _NET_INTERFACE_H_
