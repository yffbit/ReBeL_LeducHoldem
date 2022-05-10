#if !defined(_COMMAND_H_)
#define _COMMAND_H_

#include <fstream>
// #include "solver.h"
#include "limit_solver.h"
using std::ifstream;
using std::cin;
using std::cout;
using std::endl;

class Command {
public:
    Command() {
        // const char* file_bin = "./resources/card5_dic_sorted.dat";
        // if(!fcs.load(file_bin)) {
        //     string s = file_bin;
        //     s += " does not exist";
        //     throw runtime_error(s);
        // }
        net = make_shared<ValueNet>();
        game = make_shared<Game>();
    }
    void input_cmd() {
        string line;
        while(getline(cin, line)) {
            if(line.empty()) continue;
            if(tolower(line[0]) == 'q') break;
            execute_cmd(line);
        }
    }
    void load_file_cmd(const char* file_path) {
        ifstream file(file_path);
        if(!file) {
            printf("%s does not exist\n", file_path);
            return;
        }
        string line;
        while(getline(file, line)) {
            execute_cmd(line);
        }
        file.close();
    }
    void execute_cmd(string& line) {
        if(line.empty() || line[0] == '#') return;
        vector<string> contents = string_split(line, ' ');
        int size = contents.size();
        if(size == 0 || size > 2) {
            printf("command not valid: %s\n", line.c_str());
            return;
        }
        string command = contents[0];
        string param = size == 1 ? "" : contents[1];
        if(command == "set_pot") rule.init_pot[0] = rule.init_pot[1] = stoi(param) >> 1;
        // else if(command == "set_effective_stack") rule.effective_stack = stoi(param);
        // else if(command == "set_board") {
        //     vector<string> board_str = string_split(param, ',');
        //     vector<int> board = card_str_to_idx(board_str);
        //     if(!cards_valid(board)) {
        //         printf("board not valid: %s\n", param.c_str());
        //         return;
        //     }
        //     int board_size = board.size();
        //     if(board_size == 5) init_round = RIVER;
        //     else if(board_size == 4) init_round = TURN;
        //     else if(board_size == 3) init_round = FLOP;
        //     else {
        //         printf("board not support: %s\n", param.c_str());
        //         return;
        //     }
        //     init_board = cards2hash(board);
        // }
        // else if(command == "set_range_ip") range.parse_range(IP, param, init_board);
        // else if(command == "set_range_oop") range.parse_range(OOP, param, init_board);
        // else if(command == "set_bet_sizes") {
        //     vector<string> params = string_split(param, ',');
        //     if(params.size() < 3) {
        //         printf("format error: %s\n", param.c_str());
        //         return;
        //     }
        //     // oop,turn,bet,30,70,100
        //     string& player = params[0];
        //     string& round = params[1];
        //     string& bet_type = params[2];
        //     int player_idx = OOP, round_idx = FLOP;
        //     if(player == "oop") player_idx = OOP;
        //     else if(player == "ip") player_idx = IP;
        //     else {
        //         printf("player must in {oop,ip}: %s\n", param.c_str());
        //         return;
        //     }
        //     if(round == "flop") round_idx = FLOP;
        //     else if(round == "turn") round_idx = TURN;
        //     else if(round == "river") round_idx = RIVER;
        //     else {
        //         printf("round must in {flop,turn,river}: %s\n", param.c_str());
        //         return;
        //     }
        //     BetRule& bet_rule = rule.bet_rule[player_idx][round_idx];
        //     if(bet_type == "allin") bet_rule.allin = true;
        //     else if(bet_type == "bet") parse_bet_size(params, 3, bet_rule.bet_size);
        //     else if(bet_type == "raise") parse_bet_size(params, 3, bet_rule.raise_size);
        //     else if(bet_type == "donk" && player_idx == OOP) parse_bet_size(params, 3, bet_rule.donk_size);
        //     else {
        //         printf("type must in {allin,bet,raise,donk} and donk only for oop: %s\n", param.c_str());
        //         return;
        //     }
        // }
        else if(command == "set_accuracy") solver_param.accuracy = stof(param);
        // else if(command == "set_allin_threshold") rule.allin_threshold = stof(param);
        else if(command == "set_raise_limit") rule.max_bet_num = stoi(param);
        else if(command == "set_thread_num") solver_param.n_thread = stoi(param);
        else if(command == "build_tree") {
            // if(init_round < FLOP || rule.init_pot < 0 || rule.effective_stack < 0) {
            //     printf("round must in {flop,turn,river} and initial pot >= 0 and effective stack >= 0\n");
            //     return;
            // }
            // if(rule.effective_stack) root = make_shared<ActionNode>(OOP, init_round, rule.init_pot, rule.init_pot);
            // else root = make_shared<ShowdownNode>(OOP, init_round, rule.init_pot, rule.init_pot);
            // rule.init();
            // root = make_shared<Node>(0, 0, rule.init_pot[0], rule.init_pot[1], ACTION_NODE);
            // builder.build_tree(root);
            game->reset(rule);
        }
        // else if (command == "dump_tree") {
        //     if(root == nullptr) {
        //         printf("tree is empty\n");
        //         return;
        //     }
        //     tree.save_tree(root, param);
        // }
        else if(command == "set_max_iteration") solver_param.max_iter = stoi(param);
        // else if(command == "set_use_isomorphism") use_isomorphism = stoi(param);
        else if(command == "set_print_interval") solver_param.print_interval = stoi(param);
        else if(command == "set_log_file") log_file = param;
        else if(command == "set_cfr_param") {
            vector<string> params = string_split(param, ',');
            int size = params.size();
            if(size == 3) {
                cfr_param.linear = stoi(params[0]);
                cfr_param.rm_plus = stoi(params[1]);
                cfr_param.hedge = stoi(params[2]);
                cfr_param.discount = false;
            }
            else if(size == 4) {
                cfr_param.alpha = stof(params[0]);
                cfr_param.beta = stof(params[1]);
                cfr_param.gamma = stof(params[2]);
                cfr_param.hedge = stoi(params[3]);
                cfr_param.discount = true;
            }
            else {
                printf("vanilla cfr need 2 boolean params: linear, rm+\ndiscounted cfr need 3 float params: alpha, beta, gamma\n");
                return;
            }
        }
        // else if(command == "set_monte_carlo") {
        //     if(param == "none") monte_carlo = MC_NONE;
        //     else if(param == "cs") monte_carlo = MC_CHANCE;
        //     else if(param == "es") monte_carlo = MC_EXTERNAL;
        //     else if(param == "os") monte_carlo = MC_OUTCOME;
        //     else printf("sampling method must in {none,cs,es,os}\n");
        // }
        else if(command == "start_solve") {
            // const vector<vector<Hand>>& hands = range.get_hands();
            // if(root == nullptr || hands[IP].empty() || hands[OOP].empty()) {
            //     printf("tree or player's range is empty\n");
            //     return;
            // }
            /*printf("IP:\n");
            for(const Hand& hand : hands[IP]) printf("%d,%d,%g\n", hand.card2, hand.card1, hand.weight);
            printf("OOP:\n");
            for(const Hand& hand : hands[OOP]) printf("%d,%d,%g\n", hand.card2, hand.card1, hand.weight);*/
            // if(root->type() == SHOWDOWN_NODE) printf("warning: tree is rooted at leaf\n");
            printf("<<<START SOLVING>>>\n");
            // if(monte_carlo != MC_NONE) solver = make_shared<MonteCarloSolver>(root, cfr_param, init_prob, game, n_thread, max_iter, print_interval, accuracy, log_file, monte_carlo);
            // else solver = make_shared<Solver>(root, cfr_param, init_prob, game, n_thread, max_iter, print_interval, accuracy, log_file);
            // solver->train();
            solver = make_shared<LimitSolver>(game, cfr_param, solver_param, net);
            solver->set_subtree_data(game->root, game->init_prob);
            solver->train();
        }
        // else if(command == "set_dump_rounds") dump_rounds = stoi(param);
        // else if(command == "dump_result") {}
        else {
            printf("unknow command: %s\n", line.c_str());
        }
    }
private:
    // float accuracy = 0.001;
    // int max_iter = 100;
    // bool use_isomorphism = 0;
    // int print_interval = 10;
    // int dump_rounds = 1;
    // int init_round = PREFLOP;
    // int n_thread = 6;
    // size_t init_board = 0LL;
    string log_file;
    CFRParam cfr_param;
    SolverParam solver_param;
    // int monte_carlo = MC_NONE;
    Rule rule;
    shared_ptr<Game> game;
    // Range range;
    // TreeBuilder builder;
    // shared_ptr<Node> root = nullptr;
    // FiveCardsStrength fcs;
    // FinalStrengthManager fsm;
    // shared_ptr<Solver> solver = nullptr;
    shared_ptr<LimitSolver> solver = nullptr;
    shared_ptr<ValueNet> net = nullptr;
};

void print_possible_cmd() {
    cout << "possible cmd:\n";
    // cout << "set_pot\n";
    // cout << "set_effective_stack\n";
    // cout << "set_board\n";
    // cout << "set_range_oop\n";
    // cout << "set_range_ip\n";
    // cout << "set_bet_sizes\n";
    cout << "set_raise_limit\n";
    // cout << "set_allin_threshold\n";
    cout << "build_tree\n";
    // cout << "dump_tree\n";
    cout << "set_thread_num\n";
    cout << "set_accuracy\n";
    cout << "set_max_iteration\n";
    cout << "set_print_interval\n";
    cout << "set_log_file\n";
    cout << "set_cfr_param\n";
    // cout << "set_use_isomorphism\n";
    cout << "set_monte_carlo\n";
    cout << "start_solve\n";
    // cout << "set_dump_rounds\n";
    // cout << "dump_result\n\n";
}

#endif // _COMMAND_H_
