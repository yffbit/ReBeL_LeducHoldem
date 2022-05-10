#if !defined(_NET_H_)
#define _NET_H_

#include "net_interface.h"
#include "limit_solver.h"

// 在任意节点求解
class OracleNet : public ValueNet {
public:
    OracleNet(shared_ptr<Game> game, CFRParam& cfr_param, SolverParam& param):solver(game, cfr_param, param) {}
    HandVector2 compute_value(shared_ptr<Node> node, const HandVector2& reach_prob) {
        solve(node, reach_prob);
        HandVector2 cfv = HandVector2::Constant(N_CARD, N_PLAYER, 0);
        if(any_zero) return cfv;
        for(int player = 0; player < N_PLAYER; player++) {
            cfv.col(player) = solver.get_root_value(player);
        }
        return cfv;
    }
    HandVector compute_value(shared_ptr<Node> node, const HandVector2& reach_prob, int player) {
        solve(node, reach_prob);
        if(any_zero) return HandVector::Constant(N_CARD, 0);
        return solver.get_root_value(player);
    }
    bool return_cfv() {
        return true;
    }
private:
    bool any_zero = false;
    LimitSolver solver;
    void solve(shared_ptr<Node> node, const HandVector2& reach_prob) {
        Array<double, 1, N_PLAYER> prob_sum = reach_prob.colwise().sum();
        if(prob_sum(0) == 0 || prob_sum(1) == 0) {
            any_zero = true;
            return;
        }
        any_zero = false;
        solver.set_subtree_data(node, reach_prob);
        // solver.train();
        solver.multi_step(-1);
    }
};

class TorchScriptNet : public ValueNet {
public:
    TorchScriptNet(const string& path, const string& device):device(device) {
        try {
            model = torch::jit::load(path);
        } catch(const c10::Error& e) {
            std::cout << e.what() << std::endl;
        }
        std::cout << "load: " << path << std::endl;
        model.to(this->device);
    }
    at::Tensor compute_value(at::Tensor query) {
        torch::NoGradGuard no_grad;
        return model.forward({query.to(device)}).toTensor().to(torch::kCPU);
    }
private:
    torch::jit::Module model;
    torch::Device device;
};

#endif // _NET_H_
