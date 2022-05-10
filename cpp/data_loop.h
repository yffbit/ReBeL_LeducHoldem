#if !defined(_DATA_LOOP_H_)
#define _DATA_LOOP_H_

#include <random>
#include <deque>
#include "thread_loop.h"
#include "net_interface.h"
#include "recursive_solver.h"
using std::deque;

class TrainSample {
public:
    TrainSample() {}
    TrainSample(at::Tensor feature, at::Tensor target):feature(feature), target(target) {}
    at::Tensor feature, target;// [?, dim]
};

class ReplayBuffer {
public:
    ReplayBuffer(int capacity, int seed):capacity(capacity), engine(seed) {
        if(capacity <= 0) throw std::runtime_error("capacity must be greater than 0");
    }
    void add(TrainSample& val) {
        // if(stop) return;
        {
            lock_guard<mutex> lk(m);
            if(dq.size() >= capacity) dq.pop_front();
            dq.push_back(val);
        }
        cv.notify_one();
        num_add_++;// 每次添加一个训练样本
    }
    /*T pop(bool wait = true) {
        unique_lock<mutex> lk(m);
        if (wait) cv.wait(lk, [this]() { return stop || !dq.empty(); });
        if (dq.empty()) throw std::out_of_range("container is empty");
        T val = dq.front();
        dq.pop_front();
        return val;
    }*/
    TrainSample sample(int batch_size) {
        unique_lock<mutex> lk(m);
        cv.wait(lk, [=](){return stop || dq.size()>=batch_size;});
        if(dq.empty()) return TrainSample();
        std::uniform_int_distribution<int> generator(0, dq.size()-1);
        vector<at::Tensor> query, value;
        for(int i = 0; i < batch_size; i++) {
            int idx = generator(engine);
            TrainSample& s = dq[idx];
            query.push_back(s.feature);
            value.push_back(s.target);
        }
        return TrainSample(torch::stack(query, 0), torch::stack(value, 0));
    }
    void set(bool stop = true) {
        this->stop = stop;
        cv.notify_all();
    }
    unsigned int size() {
        lock_guard<mutex> lk(m);
        return dq.size();
    }
    void pop_until(unsigned int new_size) {
        lock_guard<mutex> lk(m);
        int size = dq.size();
        if(new_size >= size) return;
        int cnt = size - new_size;
        while((cnt--) > 0) dq.pop_front();
    }
    unsigned int num_add() {return num_add_;}
private:
    std::atomic_bool stop = false;
    std::atomic_uint num_add_ = 0;
    int capacity;
    mutex m;
    condition_variable cv;
    deque<TrainSample> dq;
    std::mt19937 engine;
};

class DataLoop : public ThreadLoop {
public:
    DataLoop(shared_ptr<Game> game, CFRParam& cfr_param, SolverParam& param, shared_ptr<ValueNet> net, int seed)
    :data_generator(game, cfr_param, param, net, seed) {}
    void loop() {
        std::cout << "loop start" << std::endl;
        while(run) {
            wait_until_resume();
            if(!run) break;
            data_generator.step();
        }
    }
private:
    RecursiveSolver data_generator;
};

#endif // _DATA_LOOP_H_
