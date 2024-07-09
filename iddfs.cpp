#include "iddfs.h"
#include <omp.h>
#include <climits>
#include <atomic>

// Naimplementujte efektivni algoritmus pro nalezeni nejkratsi (respektive nej-
// levnejsi) cesty v grafu. V teto metode mate ze ukol naimplementovat pametove
// efektivni algoritmus pro prohledavani velkeho stavoveho prostoru. Pocitejte
// s tim, ze Vami navrzeny algoritmus muze bezet na stroji s omezenym mnozstvim
// pameti (radove nizke stovky megabytu). Vhodnym pristupem tak muze byt napr.
// iterative-deepening depth-first search.
//
// Metoda ma za ukol vratit ukazatel na cilovy stav, ktery je dosazitelny pomoci
// nejkratsi/nejlevnejsi cesty.


state_ptr dfs(state_ptr nd, size_t depth){
    if(nd->is_goal()){
        return nd;
    }
    else if(depth == 0){
        return nullptr;
    }

    state_ptr goal = nullptr;
    for (const state_ptr& next_state: nd->next_states()) {
        if(nd->get_predecessor() != nullptr and ((nd->get_predecessor())->get_identifier() == next_state->get_identifier())) {
            continue;
        }

        if (depth > 3) {
            #pragma omp task shared(goal)
            {
                auto ret = dfs4(next_state, depth - 1);
                if (ret != nullptr) {
                    #pragma omp critical
                    {
                        if (goal == nullptr or ret->get_identifier() < goal->get_identifier()) {
                            goal = ret;
                        }
                    }
                }
            }
        } else {
            auto ret = dfs4(next_state, depth - 1);
            if (ret != nullptr) {
                #pragma omp critical
                {
                    if (goal == nullptr or ret->get_identifier() < goal->get_identifier()) {
                        goal = ret;
                    }
                }
            }
        }
    }
    #pragma omp taskwait

    return goal;
}

state_ptr iddfs(state_ptr root) {
    state_ptr goal = nullptr;
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (size_t depth = 0; goal == nullptr; depth++) {
                goal = dfs(root, depth);
            }
        }
    };
    return goal;
}