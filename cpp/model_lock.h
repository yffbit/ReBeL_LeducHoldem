#if !defined(_MODEL_LOCK_H_)
#define _MODEL_LOCK_H_

#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <mutex>
#include <exception>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using std::vector;
using std::deque;
using std::string;
using std::mutex;
using std::lock_guard;
using std::runtime_error;

class ModelLock {
public:
    // python api
    ModelLock(vector<py::object>& py_model, string& device):py_model(py_model), device(device), n(py_model.size()) {
        for(int i = 0; i < n; i++) {
            model.push_back(this->py_model[i].attr("_c").cast<torch::jit::Module*>());
        }
        init();
    }
    // c++ api
    ModelLock(string& path, string& device, int n):device(device), n(n) {
        for(int i = 0; i < n; i++) {
            jit_model.push_back(torch::jit::load(path));
            model.push_back(&jit_model[i]);
        }
        init();
    }
    // python api
    void load_state_dict(py::object& py_model) {
        for(int i = 0; i < n; i++) mlock[i].lock();
        for (int i = 0; i < n; i++) {
            this->py_model[i].attr("load_state_dict")(py_model.attr("state_dict")());
            mlock[i].unlock();
        }
    }
    at::Tensor forward(at::Tensor query, int id) {
        if(id < 0 || id >= n) throw runtime_error("model index out of range");
        lock_guard<mutex> lg(mlock[id]);
        // std::cout << "query:\n" << query << std::endl;
        // std::cout << id << std::endl;
        torch::NoGradGuard no_grad;
        // try {
        return model[id]->forward({query.to(device)}).toTensor().to(torch::kCPU);
        // } catch (std::exception& e) {
        //     std::cout << e.what() << std::endl;
        // }
    }
private:
    int n;
    torch::Device device;
    deque<mutex> mlock;
    vector<py::object> py_model;
    vector<torch::jit::Module> jit_model;
    vector<torch::jit::Module*> model;
    void init() {
        for(int i = 0; i < n; i++) {
            model[i]->to(device);
            model[i]->train(false);
        }
        mlock.resize(n);
    }
};

#include "data_loop.h"

class TrainDataNet : public ValueNet {
public:
    TrainDataNet(shared_ptr<ModelLock> model_lock, int id, shared_ptr<ReplayBuffer> buffer):model_lock(model_lock), id(id), buffer(buffer) {}
    at::Tensor compute_value(at::Tensor query) {
        return model_lock->forward(query, id);
    }
    void add_training_data(at::Tensor query, at::Tensor value) {
        TrainSample data(query, value);
        buffer->add(data);
    }
private:
    int id;
    shared_ptr<ModelLock> model_lock;
    shared_ptr<ReplayBuffer> buffer;
};

#endif // _MODEL_LOCK_H_
