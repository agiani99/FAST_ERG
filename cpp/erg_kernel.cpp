// ErG v2 raw-histogram kernel: SMILES -> uint8[321] (6 counts + 21 pair types x 15 distance bins).
// Semantics are identical to erg_v2.erg_raw() (python reference); see test_erg_v2.py / test_cpp.py.
//   sites : AR (aromatic 5/6/7 rings), HY (>=3 connected neutral C with only C neighbours),
//           PLUS / MINUS (formal charge), HA (N O F S Cl Br, neutral), HD (N/O with >=1 H, same atom as an HA site)
//   pair  : min bond-count distance over member atoms (a != b), clamped to [1,15];
//           HA/HD pair of one atom and cross-fragment pairs are skipped.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static constexpr int NB = 15, NPT = 21, NCOL = 6 + NPT * NB;
enum Feat { HA = 0, HD = 1, AR = 2, HY = 3, PLUS = 4, MINUS = 5 };

struct PairLut {
    int v[6][6];
    PairLut() {
        int k = 0;
        for (int i = 0; i < 6; ++i)
            for (int j = i; j < 6; ++j) { v[i][j] = v[j][i] = k++; }
    }
};
static const PairLut LUT;

struct Scratch {                       // reused per thread to avoid allocations
    std::vector<int> site_f, site_off, site_atoms;
    std::vector<int> Z, chg, nbr_off, nbr;
    std::vector<uint8_t> arom, ok, seen, dist;
    std::vector<int> row_of, queue, stack, comp;
};

static inline bool is_ha_z(int z) { return z == 7 || z == 8 || z == 9 || z == 16 || z == 17 || z == 35; }

static void add_site(Scratch& s, int f, const int* atoms, int n) {
    s.site_f.push_back(f);
    s.site_atoms.insert(s.site_atoms.end(), atoms, atoms + n);
    s.site_off.push_back((int)s.site_atoms.size());
}

static bool erg_raw(const std::string& smi, uint8_t* out, Scratch& s) {
    std::fill(out, out + NCOL, (uint8_t)0);
    if (smi.empty()) return false;
    std::unique_ptr<RDKit::RWMol> mol;
    try { mol.reset(RDKit::SmilesToMol(smi)); } catch (...) { return false; }
    if (!mol) return false;
    const int n = (int)mol->getNumAtoms();

    s.site_f.clear(); s.site_atoms.clear(); s.site_off.assign(1, 0);
    s.Z.resize(n); s.chg.resize(n); s.arom.resize(n);
    s.nbr_off.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        const RDKit::Atom* a = mol->getAtomWithIdx(i);
        s.Z[i] = a->getAtomicNum(); s.chg[i] = a->getFormalCharge(); s.arom[i] = a->getIsAromatic();
        s.nbr_off[i + 1] = (int)a->getDegree();
    }
    for (int i = 0; i < n; ++i) s.nbr_off[i + 1] += s.nbr_off[i];
    s.nbr.assign(s.nbr_off[n], 0);
    {
        std::vector<int>& fill = s.queue; fill.assign(n, 0);
        for (const auto b : mol->bonds()) {
            int i = (int)b->getBeginAtomIdx(), j = (int)b->getEndAtomIdx();
            s.nbr[s.nbr_off[i] + fill[i]++] = j;
            s.nbr[s.nbr_off[j] + fill[j]++] = i;
        }
    }

    // AR
    for (const auto& r : mol->getRingInfo()->atomRings()) {
        if (r.size() < 5 || r.size() > 7) continue;
        bool all = true;
        for (int idx : r) if (!s.arom[idx]) { all = false; break; }
        if (all) add_site(s, AR, (const int*)r.data(), (int)r.size());
    }
    // HY: neutral non-aromatic C with only C neighbours, connected components >= 3
    s.ok.assign(n, 0);
    int nok = 0;
    for (int i = 0; i < n; ++i) {
        if (s.Z[i] != 6 || s.arom[i] || s.chg[i] != 0) continue;
        bool good = true;
        for (int k = s.nbr_off[i]; k < s.nbr_off[i + 1]; ++k) if (s.Z[s.nbr[k]] != 6) { good = false; break; }
        s.ok[i] = good; nok += good;
    }
    if (nok >= 3) {
        s.seen.assign(n, 0);
        for (int st = 0; st < n; ++st) {
            if (!s.ok[st] || s.seen[st]) continue;
            s.comp.clear(); s.stack.assign(1, st); s.seen[st] = 1;
            while (!s.stack.empty()) {
                int c = s.stack.back(); s.stack.pop_back(); s.comp.push_back(c);
                for (int k = s.nbr_off[c]; k < s.nbr_off[c + 1]; ++k) {
                    int j = s.nbr[k];
                    if (s.ok[j] && !s.seen[j]) { s.seen[j] = 1; s.stack.push_back(j); }
                }
            }
            if (s.comp.size() >= 3) add_site(s, HY, s.comp.data(), (int)s.comp.size());
        }
    }
    // atom-centred
    for (int i = 0; i < n; ++i) {
        if (s.chg[i] > 0) add_site(s, PLUS, &i, 1);
        else if (s.chg[i] < 0) add_site(s, MINUS, &i, 1);
        else if (is_ha_z(s.Z[i])) {
            add_site(s, HA, &i, 1);
            if ((s.Z[i] == 7 || s.Z[i] == 8) && mol->getAtomWithIdx(i)->getTotalNumHs() > 0) add_site(s, HD, &i, 1);
        }
    }

    const int S = (int)s.site_f.size();
    int counts[6] = {0, 0, 0, 0, 0, 0};
    for (int f : s.site_f) counts[f]++;
    int hist[NPT * NB]; std::fill(hist, hist + NPT * NB, 0);

    if (S > 1) {
        // BFS rows for every distinct atom that belongs to a site
        s.row_of.assign(n, -1);
        int nrows = 0;
        for (int a : s.site_atoms) if (s.row_of[a] < 0) s.row_of[a] = nrows++;
        s.dist.assign((size_t)nrows * n, 255);
        for (int a = 0; a < n; ++a) {
            if (s.row_of[a] < 0) continue;
            uint8_t* d = &s.dist[(size_t)s.row_of[a] * n];
            s.queue.assign(1, a); d[a] = 0;
            for (size_t h = 0; h < s.queue.size(); ++h) {
                int c = s.queue[h];
                for (int k = s.nbr_off[c]; k < s.nbr_off[c + 1]; ++k) {
                    int j = s.nbr[k];
                    if (d[j] == 255) { d[j] = (uint8_t)std::min(254, d[c] + 1); s.queue.push_back(j); }
                }
            }
        }
        for (int x = 0; x < S; ++x) {
            for (int y = x + 1; y < S; ++y) {
                int best = 255;
                for (int p = s.site_off[x]; p < s.site_off[x + 1]; ++p) {
                    int a = s.site_atoms[p];
                    const uint8_t* d = &s.dist[(size_t)s.row_of[a] * n];
                    for (int q = s.site_off[y]; q < s.site_off[y + 1]; ++q) {
                        int b = s.site_atoms[q];
                        if (a != b && d[b] < best) best = d[b];
                    }
                }
                if (best == 255) continue;                 // same atom (HA/HD) or disconnected
                int bin = std::min(NB, std::max(1, best));
                hist[LUT.v[s.site_f[x]][s.site_f[y]] * NB + bin - 1]++;
            }
        }
    }
    for (int i = 0; i < 6; ++i) out[i] = (uint8_t)std::min(255, counts[i]);
    for (int i = 0; i < NPT * NB; ++i) out[6 + i] = (uint8_t)std::min(255, hist[i]);
    return true;
}

static py::tuple raw_batch(const std::vector<std::string>& smiles, int threads) {
    const py::ssize_t N = (py::ssize_t)smiles.size();
    py::array_t<uint8_t> raw({N, (py::ssize_t)NCOL});
    py::array_t<bool> okarr(N);
    uint8_t* rp = raw.mutable_data();
    bool* op = okarr.mutable_data();
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        omp_set_num_threads(threads > 0 ? threads : omp_get_num_procs());
#endif
#pragma omp parallel
        {
            Scratch scratch;
#pragma omp for schedule(dynamic, 256)
            for (long long i = 0; i < (long long)N; ++i) {
                bool ok = false;
                try { ok = erg_raw(smiles[i], rp + (size_t)i * NCOL, scratch); }
                catch (...) { ok = false; }
                if (!ok) std::fill(rp + (size_t)i * NCOL, rp + (size_t)(i + 1) * NCOL, (uint8_t)0);
                op[i] = ok;
            }
        }
    }
    return py::make_tuple(raw, okarr);
}

PYBIND11_MODULE(erg_kernel, m) {
    m.doc() = "ErG v2 raw-histogram kernel (C++/RDKit/OpenMP)";
    m.def("raw_batch", &raw_batch, py::arg("smiles"), py::arg("threads") = 0,
          "list[str] -> (uint8[N,321], bool[N] ok). threads=0 uses all logical processors.");
    m.attr("NCOL") = NCOL;
}
