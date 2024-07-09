#pragma once

#include <sstream>
#include <cmath>
#include <random>
#include "../state.h"

namespace sliding_puzzle {
    template<unsigned int SIZE>
    class sp_state : public state, public std::enable_shared_from_this<sp_state<SIZE>> {
    private:
        std::vector<unsigned int> conf;
        unsigned long long id;

        const unsigned int BLANK = SIZE * SIZE - 1;

    public:
        sp_state(const state_ptr predecessor, unsigned int cost, std::vector<unsigned int> conf)
            : state(predecessor, cost), conf(conf) {
            id = 0ull;
            unsigned long long id2 = 0ull;
            unsigned long long mult = 1;
            for (unsigned int i = 0; i < SIZE; i++) {
                for (unsigned int j = 0; j < SIZE; j++) {
                    id += mult * conf[i * SIZE + j];
                    id2 += pow(BLANK + 1, i * SIZE + j) * conf[i * SIZE + j];
                    mult *= BLANK + 1;
                }
            }

            unsigned long long vmin = std::min<unsigned long long>(id, id2);
            id = std::max<unsigned long long>(vmin, id);
        }

        ~sp_state() {}


        std::vector<state_ptr> next_states() const override {
            auto tmp_conf = conf;
            std::vector<state_ptr> succ;

            int blank_x = 0;
            int blank_y = 0;

            // find blank
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t k = 0; k < SIZE; k++) {
                    if (conf[i * SIZE + k] == BLANK) {
                        blank_x = i;
                        blank_y = k;
                    }
                }
            }

            // 4 possibilities
            if (blank_x - 1 >= 0) {
                tmp_conf[blank_x * SIZE + blank_y] = conf[(blank_x - 1) * SIZE + blank_y];
                tmp_conf[(blank_x - 1) * SIZE + blank_y] = BLANK;
                succ.emplace_back(
                    std::make_shared<sp_state<SIZE>>(this->shared_from_this(), current_cost() + 1, tmp_conf));
                tmp_conf[blank_x * SIZE + blank_y] = conf[blank_x * SIZE + blank_y];
                tmp_conf[(blank_x - 1) * SIZE + blank_y] = conf[(blank_x - 1) * SIZE + blank_y];
            }

            if (blank_x + 1 < (int)SIZE) {
                tmp_conf[blank_x * SIZE + blank_y] = conf[(blank_x + 1) * SIZE + blank_y];
                tmp_conf[(blank_x + 1) * SIZE + blank_y] = BLANK;
                succ.emplace_back(
                    std::make_shared<sp_state<SIZE>>(this->shared_from_this(), current_cost() + 1, tmp_conf));
                tmp_conf[blank_x * SIZE + blank_y] = conf[blank_x * SIZE + blank_y];
                tmp_conf[(blank_x + 1) * SIZE + blank_y] = conf[(blank_x + 1) * SIZE + blank_y];
            }

            if (blank_y - 1 >= 0) {
                tmp_conf[blank_x * SIZE + blank_y] = conf[blank_x * SIZE + blank_y - 1];
                tmp_conf[blank_x * SIZE + blank_y - 1] = BLANK;
                succ.emplace_back(
                    std::make_shared<sp_state<SIZE>>(this->shared_from_this(), current_cost() + 1, tmp_conf));
                tmp_conf[blank_x * SIZE + blank_y] = conf[blank_x * SIZE + blank_y];
                tmp_conf[blank_x * SIZE + blank_y - 1] = conf[blank_x * SIZE + blank_y - 1];
            }

            if (blank_y + 1 < (int)SIZE) {
                tmp_conf[blank_x * SIZE + blank_y] = conf[blank_x * SIZE + blank_y + 1];
                tmp_conf[blank_x * SIZE + blank_y + 1] = BLANK;
                succ.emplace_back(
                    std::make_shared<sp_state<SIZE>>(this->shared_from_this(), current_cost() + 1, tmp_conf));
            }
            return succ;
        }

        bool is_goal() const override {
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t k = 0; k < SIZE; k++) {
                    if (i == SIZE - 1 && k == SIZE - 1) return true;
                    if (conf[i * SIZE + k] != i * SIZE + k)
                        return false;
                }
            }
            return false;
        }

        unsigned long long int get_identifier() const override {
            return id;
        }

        std::string to_string() const override {
            std::ostringstream out;
            out << "[ ";
            for (unsigned int i = 0; i < SIZE; i++) {
                for (unsigned int j = 0; j < SIZE; j++) {
                    out << conf[i * SIZE + j] << " ";
                }
                if (i != SIZE - 1) out << "| ";
                else out << "]";
            }
            return out.str();
        }

        std::vector<unsigned int> get_conf() const {
            return conf;
        }
    };

    template<unsigned int SIZE, unsigned int SOLUTION_DEPTH, unsigned int SEED>
    class domain {
    public:
        state_ptr get_root() {

            std::cout << "Domena Loyduv hlavolam\n";
            std::cout << "Sirka desky = " << SIZE << ", seed = " << SEED << "\n";
            std::cout << "Pocatecni deska = ";

            std::vector<unsigned int> rootState(SIZE * SIZE);
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t k = 0; k < SIZE; k++) {
                    rootState[i * SIZE + k] = i * SIZE + k;
                }
            }

            std::mt19937 rng(SEED);
            std::uniform_int_distribution<int> uni(0, 3);
            state_ptr s = std::make_shared<sp_state<SIZE>>(std::shared_ptr<state>(), 0, rootState);
            for (size_t i = 0; i < SOLUTION_DEPTH; i++) {
                std::vector<state_ptr> succ = s->next_states();
                s = succ[uni(rng) % succ.size()];
            }

            auto ss = std::static_pointer_cast<const sp_state<SIZE>>(s);

            std::cout << ss->to_string() << "\n\n";

            return std::make_shared<const sp_state<SIZE>>(state_ptr(), 0, ss->get_conf());
        }

    };

    // explicit instantiation to get type checking
    template
    class domain<1, 1, 0>;
}
