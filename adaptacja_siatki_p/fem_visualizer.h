#pragma once

// ==========================================================
// fem_visualizer.h
//
// Wizualizacja na zywo adaptacji siatki MES za pomoca gnuplot.
// Wymaga zainstalowanego gnuplot.
//
// Wersja zgodna z kodem, w ktorym rozdzielamy:
//
//   meshNodes  - wezly siatki elementow, czyli konce elementow
//   dofCoords  - wspolrzedne globalnych stopni swobody
//   d          - wartosci rozwiazania w globalnych stopniach swobody
//
// Dla p_deg = 1:
//   meshNodes.size() == dofCoords.size() == d.size()
//
// Dla p_deg = 2:
//   dofCoords zawiera dodatkowe punkty srodkowe elementow,
//   wiec dofCoords.size() > meshNodes.size().
//
// Uwaga: gnuplot 6.x nie obsluguje plot '-' wewnatrz multiplot.
// Uzywamy datablock-ow ($nazwa << EOD ... EOD).
//
// Uzycie w main.cpp:
//
//   FEMVisualizer viz;
//
//   viz.plotStep(
//       step,
//       meshNodes,
//       result.dofCoords,
//       result.d,
//       result.eta2List,
//       etaGlobal,
//       uhEnergyNorm,
//       TOL,
//       alpha,
//       p_deg
//   );
//
// ==========================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include "../Eigen/Dense"

struct FEMVisualizer {

    FILE* gp = nullptr;

    // Historia zbieznosci — akumulowana miedzy krokami
    std::vector<double> etaHistory;
    std::vector<double> tolHistory;

    // ==========================================================
    // Konstruktor: otwiera pipe do gnuplot i konfiguruje terminal
    // ==========================================================

    FEMVisualizer() {
        gp = popen("gnuplot", "w");

        if (!gp) {
            throw std::runtime_error(
                "Nie mozna otworzyc gnuplot. Upewnij sie, ze gnuplot jest zainstalowany.\n"
                "  macOS (MacPorts): sudo port install gnuplot\n"
                "  macOS (Homebrew): brew install gnuplot\n"
                "  Linux:            sudo apt install gnuplot"
            );
        }

        fprintf(
            gp,
            "set terminal qt size 1200,800 font 'Helvetica,11' "
            "title 'FEM - Adaptacja siatki'\n"
        );

        fprintf(gp, "set encoding utf8\n");
        fflush(gp);
    }

    // ==========================================================
    // Destruktor: zamkniecie pipe.
    // ==========================================================

    ~FEMVisualizer() {
        if (gp) {
            fprintf(gp, "pause mouse close\n");
            fflush(gp);
            pclose(gp);
        }
    }

    // ==========================================================
    // Pomocnik: czy wartosc jest poprawna i dodatnia?
    // ==========================================================

    static bool isFinitePos(double v) {
        return std::isfinite(v) && v > 0.0;
    }

    // ==========================================================
    // Pomocnik: bezpieczna eta_K z eta_K^2.
    // ==========================================================

    static double etaFromEta2(double eta2) {
        if (std::isfinite(eta2) && eta2 > 0.0) {
            return std::sqrt(eta2);
        }

        return 0.0;
    }

    // ==========================================================
    // Walidacja danych wejściowych.
    // ==========================================================

    static void validateInput(
        const std::vector<double>& meshNodes,
        const std::vector<double>& dofCoords,
        const Eigen::VectorXd& d,
        const std::vector<double>& eta2List
    ) {
        if (meshNodes.size() < 2) {
            throw std::runtime_error(
                "FEMVisualizer::plotStep: meshNodes musi zawierac co najmniej 2 wezly."
            );
        }

        int nElem = static_cast<int>(meshNodes.size()) - 1;

        if (static_cast<int>(eta2List.size()) != nElem) {
            throw std::runtime_error(
                "FEMVisualizer::plotStep: eta2List.size() musi byc rowne liczbie elementow meshNodes.size() - 1."
            );
        }

        if (static_cast<int>(dofCoords.size()) != d.size()) {
            throw std::runtime_error(
                "FEMVisualizer::plotStep: dofCoords.size() musi byc rowne d.size()."
            );
        }

        if (dofCoords.empty()) {
            throw std::runtime_error(
                "FEMVisualizer::plotStep: dofCoords jest pusty."
            );
        }

        for (int i = 0; i < static_cast<int>(meshNodes.size()); ++i) {
            if (!std::isfinite(meshNodes[i])) {
                throw std::runtime_error(
                    "FEMVisualizer::plotStep: meshNodes zawiera NaN albo Inf."
                );
            }
        }

        for (int i = 0; i < static_cast<int>(dofCoords.size()); ++i) {
            if (!std::isfinite(dofCoords[i])) {
                throw std::runtime_error(
                    "FEMVisualizer::plotStep: dofCoords zawiera NaN albo Inf."
                );
            }

            if (!std::isfinite(d(i))) {
                throw std::runtime_error(
                    "FEMVisualizer::plotStep: wektor d zawiera NaN albo Inf."
                );
            }
        }
    }

    // ==========================================================
    // Glowna funkcja: rysuje 4 panele po kazdym kroku adaptacji.
    //
    // meshNodes:
    //   konce elementow, np. dla 2 elementow:
    //     {0.0, 0.5, 1.0}
    //
    // dofCoords:
    //   wspolrzedne globalnych DOF-ow.
    //   Dla p_deg = 1:
    //     {0.0, 0.5, 1.0}
    //   Dla p_deg = 2:
    //     {0.0, 0.25, 0.5, 0.75, 1.0}
    //
    // d:
    //   wartosci rozwiazania w dofCoords.
    //
    // eta2List:
    //   eta_K^2 dla kazdego elementu meshNodes.
    // ==========================================================

    void plotStep(
        int step,
        const std::vector<double>& meshNodes,
        const std::vector<double>& dofCoords,
        const Eigen::VectorXd& d,
        const std::vector<double>& eta2List,
        double etaGlobal,
        double uhEnergyNorm,
        double TOL,
        double alpha,
        int p_deg
    ) {
        validateInput(meshNodes, dofCoords, d, eta2List);

        int nMeshNodes = static_cast<int>(meshNodes.size());
        int nElem = nMeshNodes - 1;
        int nDof = static_cast<int>(dofCoords.size());

        // Akumuluj historie zbieznosci
        etaHistory.push_back(etaGlobal);
        tolHistory.push_back(TOL * uhEnergyNorm);

        // Przelicz etaMax i prog refinementu
        double etaMax = 0.0;

        for (int k = 0; k < nElem; ++k) {
            double etaK = etaFromEta2(eta2List[k]);

            if (etaK > etaMax) {
                etaMax = etaK;
            }
        }

        double threshold = alpha * etaMax;

        // --------------------------------------------------
        // Zdefiniuj databloki PRZED set multiplot.
        //
        // Jesli datablock bylby pusty, gnuplot wyrzuca blad.
        // Zabezpieczenie: jesli zaden punkt nie spelnia warunku,
        // wstawiamy jeden wiersz z NaN — gnuplot go pominie,
        // ale nie zglosi bledu "empty data file".
        // --------------------------------------------------

        // ==================================================
        // $dSolution : x  u_h(x)
        //
        // Uzywamy dofCoords, a nie meshNodes.
        // Dzieki temu p_deg = 2 rysuje tez srodkowe DOF-y.
        // ==================================================

        fprintf(gp, "$dSolution << EOD\n");

        for (int i = 0; i < nDof; ++i) {
            fprintf(
                gp,
                "%.12f %.12f\n",
                dofCoords[i],
                d(i)
            );
        }

        fprintf(gp, "EOD\n");

        // ==================================================
        // $dDofs : x  1
        //
        // Pozycje globalnych stopni swobody.
        // Osobny datablock pozwala odroznic DOF-y od siatki elementow.
        // ==================================================

        fprintf(gp, "$dDofs << EOD\n");

        for (int i = 0; i < nDof; ++i) {
            fprintf(
                gp,
                "%.12f 1\n",
                dofCoords[i]
            );
        }

        fprintf(gp, "EOD\n");

        // ==================================================
        // $dEtaOk : x_mid  eta_K  width
        // Elementy z eta_K <= prog
        // ==================================================

        fprintf(gp, "$dEtaOk << EOD\n");

        bool anyOk = false;

        for (int k = 0; k < nElem; ++k) {
            double etaK = etaFromEta2(eta2List[k]);

            if (etaK <= threshold) {
                double xMid = 0.5 * (meshNodes[k] + meshNodes[k + 1]);
                double width = meshNodes[k + 1] - meshNodes[k];

                fprintf(
                    gp,
                    "%.12f %.12f %.12f\n",
                    xMid,
                    etaK,
                    width
                );

                anyOk = true;
            }
        }

        if (!anyOk) {
            fprintf(gp, "NaN NaN NaN\n");
        }

        fprintf(gp, "EOD\n");

        // ==================================================
        // $dEtaHigh : x_mid  eta_K  width
        // Elementy z eta_K > prog
        // ==================================================

        fprintf(gp, "$dEtaHigh << EOD\n");

        bool anyHigh = false;

        for (int k = 0; k < nElem; ++k) {
            double etaK = etaFromEta2(eta2List[k]);

            if (etaK > threshold) {
                double xMid = 0.5 * (meshNodes[k] + meshNodes[k + 1]);
                double width = meshNodes[k + 1] - meshNodes[k];

                fprintf(
                    gp,
                    "%.12f %.12f %.12f\n",
                    xMid,
                    etaK,
                    width
                );

                anyHigh = true;
            }
        }

        if (!anyHigh) {
            fprintf(gp, "NaN NaN NaN\n");
        }

        fprintf(gp, "EOD\n");

        // ==================================================
        // $dMesh : x  1
        //
        // Pozycje wezlow siatki elementow.
        // To NIE sa wszystkie DOF-y dla p_deg = 2.
        // ==================================================

        fprintf(gp, "$dMesh << EOD\n");

        for (int i = 0; i < nMeshNodes; ++i) {
            fprintf(
                gp,
                "%.12f 1\n",
                meshNodes[i]
            );
        }

        fprintf(gp, "EOD\n");

        // ==================================================
        // $dEtaHist / $dTolHist : krok  wartosc
        // Pomijamy NaN — logscale nie lubi NaN na osi Y.
        // ==================================================

        fprintf(gp, "$dEtaHist << EOD\n");

        for (int s = 0; s < static_cast<int>(etaHistory.size()); ++s) {
            if (isFinitePos(etaHistory[s])) {
                fprintf(
                    gp,
                    "%d %.12f\n",
                    s,
                    etaHistory[s]
                );
            }
        }

        fprintf(gp, "EOD\n");

        fprintf(gp, "$dTolHist << EOD\n");

        for (int s = 0; s < static_cast<int>(tolHistory.size()); ++s) {
            if (isFinitePos(tolHistory[s])) {
                fprintf(
                    gp,
                    "%d %.12f\n",
                    s,
                    tolHistory[s]
                );
            }
        }

        fprintf(gp, "EOD\n");

        // --------------------------------------------------
        // Wyznacz yrange dla panelu zbieznosci.
        // --------------------------------------------------

        double yMin = std::numeric_limits<double>::max();
        double yMax = -std::numeric_limits<double>::max();

        for (double v : etaHistory) {
            if (isFinitePos(v)) {
                yMin = std::min(yMin, v);
                yMax = std::max(yMax, v);
            }
        }

        for (double v : tolHistory) {
            if (isFinitePos(v)) {
                yMin = std::min(yMin, v);
                yMax = std::max(yMax, v);
            }
        }

        bool hasValidConv =
            (yMin < yMax) ||
            (yMin == yMax && isFinitePos(yMin));

        if (!hasValidConv) {
            yMin = 1e-6;
            yMax = 1.0;
        }

        double yLo = yMin * 0.5;
        double yHi = yMax * 2.0;

        if (!(std::isfinite(yLo) && std::isfinite(yHi)) || yLo <= 0.0 || yHi <= yLo) {
            yLo = 1e-6;
            yHi = 1.0;
        }

        // --------------------------------------------------
        // Rysuj 4 panele
        // --------------------------------------------------

        fprintf(
            gp,
            "set multiplot layout 2,2 "
            "title 'Adaptacja siatki MES  |  krok %d  |  p = %d  |  %d elementow  |  %d DOF  |  eta = %.2e'\n",
            step,
            p_deg,
            nElem,
            nDof,
            etaGlobal
        );

        // ==================================================
        // PANEL 1 (lewy gorny): Rozwiazanie u_h(x)
        // ==================================================

        fprintf(gp, "unset logscale\n");
        fprintf(gp, "unset xrange\n");
        fprintf(gp, "unset yrange\n");
        fprintf(gp, "set title 'Rozwiazanie u_h(x) - punkty DOF'\n");
        fprintf(gp, "set xlabel 'x'\n");
        fprintf(gp, "set ylabel 'u_h'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "unset key\n");

        fprintf(
            gp,
            "plot $dSolution using 1:2 with linespoints "
            "pt 7 ps 0.8 lw 2 lc rgb '#2563eb' notitle\n"
        );

        // ==================================================
        // PANEL 2 (prawy gorny): Estymatory bledu eta_K
        // ==================================================

        fprintf(gp, "unset logscale\n");
        fprintf(gp, "unset xrange\n");
        fprintf(gp, "unset yrange\n");
        fprintf(gp, "set title 'Estymatory bledu eta_K'\n");
        fprintf(gp, "set xlabel 'x (srodek elementu)'\n");
        fprintf(gp, "set ylabel 'eta_K'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "set key top right\n");
        fprintf(gp, "set style fill solid 0.7\n");
        fprintf(gp, "set boxwidth 1 relative\n");

        fprintf(
            gp,
            "plot $dEtaOk   using 1:2:3 with boxes lc rgb '#93c5fd' title 'eta_K <= prog', "
                 "$dEtaHigh using 1:2:3 with boxes lc rgb '#dc2626' title 'eta_K > prog', "
                 "%.12f with lines lc rgb '#f59e0b' lw 2 dt 2 title 'prog = %.2f*eta_max'\n",
            threshold,
            alpha
        );

        fprintf(gp, "unset style\n");

        // ==================================================
        // PANEL 3 (lewy dolny): Siatka i DOF-y
        //
        // Impulsy wysokie: meshNodes, czyli konce elementow.
        // Punkty niskie:  dofCoords, czyli wszystkie DOF-y.
        //
        // Dla p_deg = 1 punkty DOF pokrywaja sie z siatka.
        // Dla p_deg = 2 widac dodatkowe DOF-y w srodkach elementow.
        // ==================================================

        fprintf(gp, "unset logscale\n");
        fprintf(gp, "unset xrange\n");
        fprintf(gp, "set title 'Siatka elementow i globalne DOF-y'\n");
        fprintf(gp, "set xlabel 'x'\n");
        fprintf(gp, "set yrange [0:1.5]\n");
        fprintf(gp, "unset ylabel\n");
        fprintf(gp, "unset ytics\n");
        fprintf(gp, "unset grid\n");
        fprintf(gp, "set grid xtics\n");
        fprintf(gp, "set key top right\n");

        fprintf(
            gp,
            "plot $dMesh using 1:2 with impulses lc rgb '#2563eb' lw 1.8 title 'wezly elementow', "
                 "$dDofs using 1:(0.35) with points pt 7 ps 0.8 lc rgb '#111827' title 'DOF'\n"
        );

        fprintf(gp, "set ytics\n");
        fprintf(gp, "unset yrange\n");
        fprintf(gp, "set grid\n");

        // ==================================================
        // PANEL 4 (prawy dolny): Zbieznosc eta vs krok
        // ==================================================

        fprintf(gp, "unset xrange\n");
        fprintf(gp, "unset yrange\n");
        fprintf(gp, "set title 'Zbieznosc estymatora bledu'\n");
        fprintf(gp, "set xlabel 'Krok adaptacji'\n");
        fprintf(gp, "set ylabel 'eta'\n");
        fprintf(gp, "set logscale y\n");
        fprintf(gp, "set yrange [%.6e:%.6e]\n", yLo, yHi);
        fprintf(gp, "set xrange [-0.5:%d]\n", static_cast<int>(etaHistory.size()) + 1);
        fprintf(gp, "set grid\n");
        fprintf(gp, "set key top right\n");

        fprintf(
            gp,
            "plot $dEtaHist using 1:2 with linespoints "
                 "pt 7 ps 1.0 lw 2 lc rgb '#2563eb' title 'eta globalny', "
                 "$dTolHist using 1:2 with linespoints "
                 "pt 5 ps 1.0 lw 2 lc rgb '#dc2626' dt 2 title 'TOL * ||u_h||_E'\n"
        );

        // Reset przed nastepnym multiplot
        fprintf(gp, "unset logscale\n");
        fprintf(gp, "unset xrange\n");
        fprintf(gp, "unset yrange\n");

        fprintf(gp, "unset multiplot\n");

        fflush(gp);
    }
};