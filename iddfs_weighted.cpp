#include "iddfs_weighted.h"
#include <climits>
#include <omp.h>
#include <atomic>


state_ptr get_first_goal(state_ptr nd, size_t depth){
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
                auto ret = get_first_goal(next_state, depth - 1);
                if (ret != nullptr) {
                    #pragma omp critical
                    {
                        if (goal == nullptr or
                            (goal->current_cost() == ret->current_cost() and ret->get_identifier() < goal->get_identifier()) or
                            (goal->current_cost() > ret->current_cost())) {
                            goal = ret;
                        }
                    }
                }
            }
        } else {
            auto ret = get_first_goal(next_state, depth - 1);
            if (ret != nullptr) {
                #pragma omp critical
                {
                    if (goal == nullptr or
                        (goal->current_cost() == ret->current_cost() and ret->get_identifier() < goal->get_identifier()) or
                        (goal->current_cost() > ret->current_cost())) {
                        goal = ret;
                    }
                }
            }
        }
    }
    #pragma omp taskwait

    return goal;
}


state_ptr get_cheapest_goal(state_ptr nd, size_t min_cost){
    if(nd->is_goal()){
        return nd;
    }
    else if(nd->current_cost() > min_cost){
        return nullptr;
    }

    state_ptr goal = nullptr;
    for (auto& next_state: nd->next_states()) {
        if(nd->get_predecessor() != nullptr and ((nd->get_predecessor())->get_identifier() == next_state->get_identifier())) {
            continue;
        }

        if (min_cost < next_state->current_cost()) {
            continue;
        }

        #pragma omp task shared(goal)
        {
            auto ret = get_cheapest_goal(next_state, min_cost);
            if (ret != nullptr) {
                #pragma omp critical
                {
                    if (goal == nullptr or
                        (goal->current_cost() == ret->current_cost() and
                         ret->get_identifier() < goal->get_identifier()) or
                        (goal->current_cost() > ret->current_cost())) {
                        goal = ret;
                    }
                }
            }
        }

    }
    #pragma omp taskwait
    return goal;
}


state_ptr iddfs_weighted(state_ptr root) {
    state_ptr goal = nullptr;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for(size_t depth = 0;goal == nullptr;depth++) {
                goal = get_first_goal(root, depth);
            }

            size_t min_cost = goal->current_cost();
            goal = get_cheapest_goal(root, min_cost);
        }
    };
    return goal;
}







