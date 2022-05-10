#include "recursive_solver.h"
#include "data_loop.h"
#include "net.h"

int main(int argc, char* argv[]) {
	shared_ptr<ValueNet> net = nullptr;
	string path;
	if(argc == 1) net = make_shared<ValueNet>();
	else {
		path = argv[1];
		std::cout << path << std::endl;
		net = make_shared<TorchScriptNet>(path, "cpu");
	}
	shared_ptr<Game> game = make_shared<Game>();
	CFRParam cfr_param;
	SolverParam param;
	Context ctx;
	shared_ptr<ThreadLoop> loop = make_shared<DataLoop>(game, cfr_param, param, net, 2022);
	ctx.push(loop);
	ctx.start();
	ctx.join();
	return 0;
}