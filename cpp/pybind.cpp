#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "model_lock.h"
#include "thread_loop.h"
#include "data_loop.h"
#include "recursive_solver.h"
using std::shared_ptr;

PYBIND11_MODULE(rela, m) {
    py::class_<Game, shared_ptr<Game>>(m, "Game")
        .def(py::init<>())
        .def(py::init<int, int, int, int, int>(), 
            py::arg("p0_pot"), py::arg("p1_pot"), 
            py::arg("raise0"), py::arg("raise1"), py::arg("max_bet_num"));
    py::class_<SolverParam>(m, "SolverParam")
        .def(py::init<>())
        .def_readwrite("accuracy", &SolverParam::accuracy)
        .def_readwrite("max_iter", &SolverParam::max_iter)
        .def_readwrite("n_thread", &SolverParam::n_thread)
        .def_readwrite("print_interval", &SolverParam::print_interval);
    py::class_<CFRParam>(m, "CFRParam")
        .def(py::init<>())
        .def_readwrite("alpha", &CFRParam::alpha)
        .def_readwrite("beta", &CFRParam::beta)
        .def_readwrite("gamma", &CFRParam::gamma)
        .def_readwrite("discount", &CFRParam::discount)
        .def_readwrite("hedge", &CFRParam::hedge)
        .def_readwrite("linear", &CFRParam::linear)
        .def_readwrite("rm_plus", &CFRParam::rm_plus);
    
    py::class_<TrainSample, shared_ptr<TrainSample>>(m, "TrainSample")
        .def(py::init<>())
        .def_readwrite("feature", &TrainSample::feature)
        .def_readwrite("target", &TrainSample::target);
    py::class_<ReplayBuffer, shared_ptr<ReplayBuffer>>(m, "ReplayBuffer")
        .def(py::init<int, int>(), py::arg("capacity"), py::arg("seed"))
        .def("sample", &ReplayBuffer::sample)
        .def("size", &ReplayBuffer::size)
        .def("pop_until", &ReplayBuffer::pop_until)
        .def("num_add", &ReplayBuffer::num_add)
        .def("set", &ReplayBuffer::set);
    py::class_<ModelLock, shared_ptr<ModelLock>>(m, "ModelLock")
        .def(py::init<vector<py::object>&, string&>())
        .def("load_state_dict", &ModelLock::load_state_dict);
    py::class_<ValueNet, shared_ptr<ValueNet>>(m, "ValueNet");
    py::class_<TrainDataNet, ValueNet, shared_ptr<TrainDataNet>>(m, "TrainDataNet")
        .def(py::init<shared_ptr<ModelLock>, int, shared_ptr<ReplayBuffer>>(),
            py::arg("model_lock"), py::arg("id"), py::arg("buffer"));
    
    py::class_<ThreadLoop, shared_ptr<ThreadLoop>>(m, "ThreadLoop");
    py::class_<DataLoop, ThreadLoop, shared_ptr<DataLoop>>(m, "DataLoop")
        .def(py::init<shared_ptr<Game>, CFRParam&, SolverParam&, shared_ptr<ValueNet>, int>(), 
            py::arg("game"), py::arg("cfr_param"), py::arg("param"), py::arg("net"), py::arg("seed"));
    py::class_<Context>(m, "Context")
        .def(py::init<>())
        .def("push", &Context::push, py::keep_alive<1,2>())
        .def("start", &Context::start)
        .def("stop", &Context::stop)
        .def("join", &Context::join);
}
