#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>

#include "state.h"
#include "domains/hanoi.h"
#include "domains/maze.h"
#include "domains/sat.h"
#include "domains/sliding_puzzle.h"

#include "bfs.h"
#include "iddfs.h"
#include "iddfs_weighted.h"


// Vsechny funkce, ktere budete implementovat by mely implementovat nasledujici rozhrani.
// Mely by to tedy byt funkce s nasledujici hlavickou:
//   std::shared_ptr<const state> jmeno_funkce(std::shared_ptr<const state> root)
//
// Funkce dostane na vstupu ukazatel na pocatecni stav problemu (zdrojovy vrchol prohledavani,
// a mely by vratit ukazatel na cilovy stav, ktery je dosazitelny pomoci nejkratsi/nejlevnejsi
// cesty.
//
// Vsimnete si, ze ukazatele jsou typu 'std::shared_ptr' - coz Vam doufame usnadni hodne prace
// pri sprave pameti:
//   1) Na rozdil od 'const state *', 'std::shared_ptr<const state>' se stara o spravu pameti
//      automaticky (alokovana pamet pro stav zanikne automaticky po tom, co zanikne posledni
//      std::shared_ptr, ktery na ni ukazuje.
//   2) Na rozdil od 'std::unique_ptr<const state>' vlastnictvi ukazatele neni unikatni. To
//      znamena, ze pointer muzete predavat do ostatnich funkci/datovych struktur bez omezeni.
// Cenou za to je mirne zvysena rezie pri pristupech do pameti pres 'std::shared_ptr<...>'.
//
// Pro zjednoduseni zapisu typu pouzivame typovy alias `state_ptr`, definovany ve `state.h`.
// Pokud chcete byt radeji explicitni, muzete ve svem kodu pouzivat cely typ.
using search_fn = state_ptr (*)(state_ptr);


// Evaluacni funkce, ktera spusti prohledavaci algoritmus 'search' z pocatecniho stavu 'root'.
void evaluate(state_ptr& root, search_fn search) {
    using namespace std::chrono;

    std::cout << " ****\n";
    auto begin = steady_clock::now();
    auto result = search(root);
    auto end = steady_clock::now();

    if (result != nullptr) {
        if (result->is_goal()) {
            std::cout << "Solution found. Cost=" << result->current_cost() << "\n";
        } else {
            std::cout << "Search returned a solution - but it is not a goal!\n";
        }
    } else {
        std::cout << "No solution found.\n";
    }

    std::cout << "Time: " << duration_cast<milliseconds>(end - begin).count() << "ms\n";

    // Pro snazsi ladeni zrekonstruujeme a vypiseme nalezenou cestu
    std::vector<state_ptr> path;
    while (result) {
        path.push_back(result);
        result = result->get_predecessor();
    }
    std::reverse(path.begin(), path.end());
    for (auto& s : path) {
        std::cout << s->to_string() << "\n";
    }

    std::cout << " ****\n";
}


int main() {
    // Vytvoreni instance hanojskych vezi s 3 koliky, 1 vezi (umistenou na
    // prvnim koliku a 4 kotouci ve vezi.
//    auto d = hanoi::domain<3, 1, 4>();
    auto d = sliding_puzzle::domain<3, 200, 0>();
//    auto d = maze::domain<75, 55, 5, true>();
//      auto d = sat::domain<30, 8, 5, 10, false>();

    // Vytvorit domenu "splnovani booleovskych funkci" muzete vytvorit nasledovne:
    //auto d = sat::domain<30,7,3,1,true>();
    // Tato domena pak ma:
    //   - 30 booleovskych promennych
    //   - 7 termu (ktere jsou spojeny disjunkci)
    //   - kazdy term obsahuje maximalne 3 literaly
    //   - seed nahodneho generatoru je 1
    //   - cena za prirazeni hodnoty jednomu literalu je uniformni (1)
    //     (v pripade 'false' je cena za prirazeni hodnoty i-te promenne i)


    // Vytvorit domenu sliding-puzzle hranou na hraci plose 4x4 (15-puzzle) muzete takto:
    //auto d = sliding_puzzle::domain<4, 70, 0>();
    // Inicialni pozice je generovana provedenim 70 nahodnych tahu (nahodny
    // generator je inicializovany seedem 0).

    // Posledni domenou jsou bludiste. Bludiste o rozmerech 31x21 muzete vytvorit pomoci:
    //auto d = maze::domain<31, 21, 0, false>();
    // Bludiste je generovano nahodne za pouziti seedu 0. V pripade, ze
    // nastavite posledni parametr na 'true', cena za jeden pohyb v bludisti
    // nebude uniformni.
    //
    // POZOR! Rozmery bludiste musi byt licha cisla!

    auto root = d.get_root();

    evaluate(root, bfs);
    evaluate(root, iddfs);
    evaluate(root, iddfs_weighted);

    return 0;
}
