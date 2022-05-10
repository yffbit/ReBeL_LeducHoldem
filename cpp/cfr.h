#if !defined(_CFR_H_)
#define _CFR_H_

#include <cmath>
#include "game.h"

struct CFRParam {
    bool linear = false, rm_plus = false, discount = false, hedge = false;
    double alpha = 1, beta = 1, gamma = 1;
};

class CFRData {
public:
    CFRData(int n_act, int board, CFRParam& param):n_act(n_act), board(board), param(param) {
        regret_sum = HandVectorX::Constant(N_CARD, n_act, 0);
        strategy_sum = regret_sum;
        eta = std::sqrt(std::log(n_act)) / 3;
        if(param.hedge) {
            ev_mean = HandVector::Constant(N_CARD, 0);
            ev_var = HandVector::Constant(N_CARD, 0);
        }
    }
    virtual void clear() {
        regret_sum.fill(0);
        strategy_sum.fill(0);
        iter = 0;
        if(param.hedge) {
            ev_mean.fill(0);
            ev_var.fill(0);
        }
    }
    virtual HandVectorX average_strategy() {
        return norm_strategy(strategy_sum);
    }
    virtual HandVectorX curr_strategy() {
        if(param.hedge && iter >= 2) {
            HandVectorX strategy = (regret_sum.colwise() * (eta / (std::sqrt(iter) * ev_var.sqrt()))).exp();
            strategy = strategy.colwise() / strategy.rowwise().sum();
            if(board != N_CARD) strategy.row(board) = (double)1 / n_act;
            //if(strategy.isNaN().any()) {
            //    std::cout << ev_var << std::endl;
            //    std::cout << strategy << std::endl;
            //}
            return strategy;
        }
        if(param.rm_plus) return norm_strategy(regret_sum);
        HandVectorX data = regret_sum.max(0);
        return norm_strategy(data);
    }
    virtual ActVector curr_strategy(int hand) {
        ActVector strategy = regret_sum.row(hand);
        if(!param.rm_plus) strategy = strategy.max(0);
        double act_sum = strategy.sum();
        if(act_sum == 0) return ActVector::Constant(n_act, (double)1 / n_act);
        else return strategy / act_sum;
    }
    virtual void update(const HandVector& cfv, const HandVectorX& child_cfv, const HandVectorX& strategy, const HandVector& reach_prob, int iter1) {
        if(param.linear) {
            double coef = double(iter) / (iter + 1);
            regret_sum = regret_sum * coef + (child_cfv.colwise() - cfv);
            strategy_sum = strategy_sum * coef + strategy.colwise() * reach_prob;
        }
        else {
            regret_sum += child_cfv.colwise() - cfv;
            strategy_sum += strategy.colwise() * reach_prob;
        }
        if(param.rm_plus) regret_sum = regret_sum.max(0);
        iter++;
    }
    virtual void update(const HandVectorX& regret, const HandVectorX& weighted_strategy, const HandVector& ev, int iter1) {
        if(param.linear) {
            double coef = double(iter) / (iter + 1);
            regret_sum = regret_sum * coef + regret;
            strategy_sum = strategy_sum * coef + weighted_strategy;
        }
        else {
            regret_sum += regret;
            strategy_sum += weighted_strategy;
        }
        if(param.rm_plus) regret_sum = regret_sum.max(0);
        if(param.hedge) update_mean_var(ev);
        iter++;
    }
    virtual void update_regret(int hand, ActVector& regret, int iter) {
        if(param.linear) regret_sum.row(hand) *= double(iter - 1) / iter;
        regret_sum.row(hand) += regret;
        if(param.rm_plus) regret_sum.row(hand) = regret_sum.row(hand).max(0);
    }
    virtual void update_avg_strategy(int hand, ActVector& weighted_strategy, int iter) {
        if(param.linear) strategy_sum.row(hand) *= double(iter - 1) / iter;
        strategy_sum.row(hand) += weighted_strategy;
    }
protected:
    int n_act = 0, board = N_CARD, iter = 0;
    CFRParam param;
    double eta;
    HandVector ev_mean, ev_var;// 无偏估计
    HandVectorX regret_sum, strategy_sum;
    void update_mean_var(const HandVector& ev) {
        if(iter == 0) {
            ev_mean += (ev - ev_mean) / (iter + 1);
        }
        else {
            HandVector new_mean = ev_mean + (ev - ev_mean) / (iter + 1);
            ev_var = ((iter-1)*ev_var + iter*(ev_mean-new_mean).square() + (ev-new_mean).square()) / iter;
            ev_mean = new_mean;
        }
        //std::cout << "ev_mean: " << ev_mean.transpose() << std::endl;
        //std::cout << "ev_var:\n" << ev_var << std::endl;
    }
private:
    inline HandVectorX norm_strategy(HandVectorX& data) {
        double uniform = (double)1 / n_act;
        HandVector norm = data.rowwise().sum();
        HandVectorX strategy = data.colwise() / norm;
        for(int i = 0; i < N_CARD; i++) {
            if(norm(i) == 0) strategy.row(i).fill(uniform);
        }
        return strategy;
    }
};

class DCFRData : public CFRData {
protected:
    double alpha = 1, beta = 1, gamma = 1;
public:
    DCFRData(int n_act, int board, CFRParam& param):CFRData(n_act, board, param) {
        alpha = param.alpha;
        beta = param.beta;
        gamma = param.gamma;
        this->param.linear = false;
        this->param.rm_plus = false;
    }
    void update(const HandVector& cfv, const HandVectorX& child_cfv, const HandVectorX& strategy, const HandVector& reach_prob, int iter1) {
        double pos_coef = 0, neg_coef = 0, st_coef = 0;
        if(iter > 0) {
            double pow_res = powf(iter, alpha);
            pos_coef = pow_res / (pow_res + 1);
            pow_res = powf(iter, beta);
            neg_coef = pow_res / (pow_res + 1);
            st_coef = powf(double(iter)/(iter+1), gamma);
        }
        HandVectorX coef_mat = HandVectorX::Constant(N_CARD, n_act, pos_coef);
        regret_sum *= (regret_sum > 0).select(coef_mat, neg_coef);
        regret_sum += child_cfv.colwise() - cfv;
        strategy_sum = strategy_sum * st_coef + strategy.colwise() * reach_prob;
        iter++;
    }
};

#endif // _CFR_H_
