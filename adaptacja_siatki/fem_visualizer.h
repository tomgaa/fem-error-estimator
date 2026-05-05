#pragma once

// ==========================================================
// fem_visualizer.h
//
// Wizualizacja na zywo adaptacji siatki MES za pomoca gnuplot.
// Wymaga zainstalowanego gnuplot (sudo port install gnuplot).
//
// Uwaga: gnuplot 6.x nie obsluguje plot '-' wewnatrz multiplot.
// Uzywamy datablock-ow ($nazwa << EOD ... EOD).
//
// Uzycie w main.cpp:
//   FEMVisualizer viz;
//   viz.plotStep(step, nodes, result.d, result.eta2List,
//                etaGlobal, uhEnergyNorm, TOL, alpha);
// ==========================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>
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
                "Nie mozna otworzyc gnuplot. Upewnij sie ze gnuplot jest zainstalowany.\n"
                "  macOS (MacPorts): sudo port install gnuplot\n"
                "  macOS (Homebrew): brew install gnuplot\n"
                "  Linux:            sudo apt install gnuplot"
            );
        }

        fprintf(gp, "set terminal qt size 1200,800 font 'Helvetica,11'"
                    " title 'FEM - Adaptacja siatki'\n");
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
    // Pomocnik: czy wartosc jest poprawna (nie NaN, nie Inf)?
    // ==========================================================
    static bool isFinitePos(double v) {
        return std::isfinite(v) && v > 0.0;
    }

    // ==========================================================
    // Glowna funkcja: rysuje 4 panele po kazdym kroku adaptacji.
    // ==========================================================
    void plotStep(
        int step,
        const std::vector<double>& nodes,
        const Eigen::VectorXd& d,
        const std::vector<double>& eta2List,
        double etaGlobal,
        double uhEnergyNorm,
        double TOL,
        double alpha
    ) {
        int nNodes = static_cast<int>(nodes.size());
        int nElem  = nNodes - 1;

        // Akumuluj historię zbieznosci
        etaHistory.push_back(etaGlobal);
        tolHistory.push_back(TOL * uhEnergyNorm);

        // Przelicz etaMax i prog refinementu
        // etaK moze byc NaN gdy eta2 < 0 — traktujemy to jako 0
        double etaMax = 0.0;
        for (int k = 0; k < nElem; k++) {
            double e2 = eta2List[k];
            if (std::isfinite(e2) && e2 > 0.0) {
                double etaK = std::sqrt(e2);
                if (etaK > etaMax) etaMax = etaK;
            }
        }
        double threshold = alpha * etaMax;

        // --------------------------------------------------
        // Zdefiniuj databloki PRZED set multiplot.
        //
        // Jesli datablock bylby pusty, gnuplot wyrzuca blad.
        // Zabezpieczenie: jesli zaden punkt nie spelnia warunku,
        // wstawiamy jeden wiersz z NaN — gnuplot go pominie,
        // ale nie zglosci bledu "empty data file".
        // --------------------------------------------------

        // $dSolution : x  u_h(x)
        fprintf(gp, "$dSolution << EOD\n");
        for (int i = 0; i < nNodes; i++) {
            fprintf(gp, "%.10f %.10f\n", nodes[i], d(i));
        }
        fprintf(gp, "EOD\n");

        // $dEtaOk : x_mid  eta_K  width  (elementy <= progu, kolor niebieski)
        fprintf(gp, "$dEtaOk << EOD\n");
        bool anyOk = false;
        for (int k = 0; k < nElem; k++) {
            double e2   = eta2List[k];
            double etaK = (std::isfinite(e2) && e2 >= 0.0) ? std::sqrt(e2) : 0.0;
            if (etaK <= threshold) {
                double xMid  = 0.5 * (nodes[k] + nodes[k + 1]);
                double width = nodes[k + 1] - nodes[k];
                fprintf(gp, "%.10f %.10f %.10f\n", xMid, etaK, width);
                anyOk = true;
            }
        }
        if (!anyOk) fprintf(gp, "NaN NaN NaN\n");
        fprintf(gp, "EOD\n");

        // $dEtaHigh : x_mid  eta_K  width  (elementy > progu, kolor czerwony)
        fprintf(gp, "$dEtaHigh << EOD\n");
        bool anyHigh = false;
        for (int k = 0; k < nElem; k++) {
            double e2   = eta2List[k];
            double etaK = (std::isfinite(e2) && e2 >= 0.0) ? std::sqrt(e2) : 0.0;
            if (etaK > threshold) {
                double xMid  = 0.5 * (nodes[k] + nodes[k + 1]);
                double width = nodes[k + 1] - nodes[k];
                fprintf(gp, "%.10f %.10f %.10f\n", xMid, etaK, width);
                anyHigh = true;
            }
        }
        if (!anyHigh) fprintf(gp, "NaN NaN NaN\n");
        fprintf(gp, "EOD\n");

        // $dMesh : x  1  (pozycje wezlow do impulses)
        fprintf(gp, "$dMesh << EOD\n");
        for (int i = 0; i < nNodes; i++) {
            fprintf(gp, "%.10f 1\n", nodes[i]);
        }
        fprintf(gp, "EOD\n");

        // $dEtaHist / $dTolHist : krok  wartosc
        // Pomijamy NaN — log scale nie lubi NaN na osi Y
        fprintf(gp, "$dEtaHist << EOD\n");
        for (int s = 0; s < (int)etaHistory.size(); s++) {
            if (isFinitePos(etaHistory[s]))
                fprintf(gp, "%d %.10f\n", s, etaHistory[s]);
        }
        fprintf(gp, "EOD\n");

        fprintf(gp, "$dTolHist << EOD\n");
        for (int s = 0; s < (int)tolHistory.size(); s++) {
            if (isFinitePos(tolHistory[s]))
                fprintf(gp, "%d %.10f\n", s, tolHistory[s]);
        }
        fprintf(gp, "EOD\n");

        // Wyznacz yrange dla panelu zbieznosci z poprawnych danych
        double yMin =  std::numeric_limits<double>::max();
        double yMax = -std::numeric_limits<double>::max();
        for (double v : etaHistory) if (isFinitePos(v)) { yMin = std::min(yMin, v); yMax = std::max(yMax, v); }
        for (double v : tolHistory) if (isFinitePos(v)) { yMin = std::min(yMin, v); yMax = std::max(yMax, v); }
        bool hasValidConv = (yMin < yMax || (yMin == yMax && isFinitePos(yMin)));
        if (!hasValidConv) { yMin = 1e-6; yMax = 1.0; } // fallback
        // Dodaj marginesy: *0.5 i *2
        double yLo = yMin * 0.5;
        double yHi = yMax * 2.0;

        // --------------------------------------------------
        // Rysuj 4 panele
        // --------------------------------------------------
        fprintf(gp,
            "set multiplot layout 2,2 "
            "title 'Adaptacja siatki MES  |  krok %d  |  %d elementow  |  eta = %.2e'\n",
            step, nElem, etaGlobal
        );

        // ==================================================
        // PANEL 1 (lewy gorny): Rozwiazanie u_h(x)
        // ==================================================
        fprintf(gp, "unset logscale\n");
        fprintf(gp, "unset xrange\n");
        fprintf(gp, "unset yrange\n");
        fprintf(gp, "set title 'Rozwiazanie u_h(x)'\n");
        fprintf(gp, "set xlabel 'x'\n");
        fprintf(gp, "set ylabel 'u_h'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "unset key\n");
        fprintf(gp,
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
        fprintf(gp,
            "plot $dEtaOk   using 1:2:3 with boxes lc rgb '#93c5fd' title 'eta_K <= prog', "
                 "$dEtaHigh using 1:2:3 with boxes lc rgb '#dc2626' title 'eta_K > prog', "
                 "%.10f with lines lc rgb '#f59e0b' lw 2 dt 2 title 'prog = %.2f*eta_max'\n",
            threshold, alpha
        );
        fprintf(gp, "unset style\n");

        // ==================================================
        // PANEL 3 (lewy dolny): Siatka — pozycje wezlow
        // ==================================================
        fprintf(gp, "unset logscale\n");
        fprintf(gp, "unset xrange\n");
        fprintf(gp, "set title 'Siatka (%d wezlow)'\n", nNodes);
        fprintf(gp, "set xlabel 'x'\n");
        fprintf(gp, "set yrange [0:1.5]\n");
        fprintf(gp, "unset ylabel\n");
        fprintf(gp, "unset ytics\n");
        fprintf(gp, "unset grid\n");
        fprintf(gp, "set grid xtics\n");
        fprintf(gp, "unset key\n");
        fprintf(gp,
            "plot $dMesh using 1:2 with impulses lc rgb '#2563eb' lw 1.5 notitle\n"
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
        fprintf(gp, "set xrange [-0.5:%d]\n", (int)etaHistory.size() + 1);
        fprintf(gp, "set grid\n");
        fprintf(gp, "set key top right\n");
        fprintf(gp,
            "plot $dEtaHist using 1:2 with linespoints "
                 "pt 7 ps 1.0 lw 2 lc rgb '#2563eb' title 'eta globalny', "
                 "$dTolHist using 1:2 with linespoints "
                 "pt 5 ps 1.0 lw 2 lc rgb '#dc2626' dt 2 title 'TOL * ||u_h||_E'\n"
        );

        // Resetuj modyfikacje przed nastepnym multiplot
        fprintf(gp, "unset logscale\n");
        fprintf(gp, "unset xrange\n");
        fprintf(gp, "unset yrange\n");

        fprintf(gp, "unset multiplot\n");
        fflush(gp);
    }
};
