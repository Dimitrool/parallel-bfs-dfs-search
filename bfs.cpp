#include <atomic>
#include <unordered_set>
#include <omp.h>
#include "bfs.h"
#include <climits>

// Metoda ma za ukol vratit ukazatel na cilovy stav, ktery je dosazitelny pomoci
// nejkratsi cesty.

state_ptr bfs(state_ptr root) {
    state_ptr goal = nullptr;
    unsigned long long min = ULLONG_MAX;
    std::unordered_set<unsigned long long> visited = {}; // store identificators of visited states
    std::vector<state_ptr> frontier = {root}; // list of states to visit
    std::vector<state_ptr> next_frontier = {};
    std::vector<state_ptr> results = {};

    visited.insert(root->get_identifier());

    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int nthreads = omp_get_num_threads();
        while(goal == nullptr) {
            for (unsigned int i = tid;i < frontier.size();i+=nthreads){
                std::vector<state_ptr> next_states = frontier[i]->next_states();
                for (const state_ptr &next_state: next_states) {
                    if (visited.find(next_state->get_identifier()) == visited.end()) {
                        #pragma omp critical
                        {
                            if (visited.find(next_state->get_identifier()) == visited.end()) {
                                next_frontier.push_back(next_state);
                                visited.insert(next_state->get_identifier());
                            }
                        }

                        if (next_state->is_goal() && min > next_state->get_identifier()) {
                            #pragma omp critical
                            {
                                goal = next_state;
                                min = next_state->get_identifier();
                            }
                        }
                    }
                }
            }

            if (goal != nullptr) {
                break;
            }
            #pragma omp barrier
            #pragma omp single
            {
                frontier.swap(next_frontier);
                next_frontier.clear();
            }
            #pragma omp barrier
        }
    };

    return goal;
}

