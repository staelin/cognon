// Copyright 2009-2011 Carl Staelin. All Rights Reserved.
// Copyright 2011 Google Inc. All Rights Reserved.
//
// Author: carl.staelin@gmail.com (Carl Staelin)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This table shows information content for simple neurons
// (C=D1=D2=1) subject to the homology contraints that:
// 1) R * pL is approximately one, and 2) D_effective is
// approximately one.
//
// For each H, R pair it does a small grid search through
// the W, G_m, and Q space to find the configuration with
// the highest L (bpn) value that also meets the two
// homology constraints.
//

#include "cognon.h"
#include "monograph.h"

namespace cognon {

void OptimizeConfiguration(int32 H, int32 R, double max_rpl) {
  double optimal_bpn = -1.0;
  NeuronStatistics optimal;
  NeuronStatistics result;
  TrainConfig config;

  for (double G_m = 1.9; 1.0 < G_m ; G_m -= 0.1) {
    double H_m = (double)H * G_m;
    double max_bpn_G = -1.0;
    for (double Q = 0.5; Q < 1.5; Q += 0.1) {
      double max_bpn = -1.0;
      for (int32 w = 1; w <= 100; w += (w < 10 ? 1 : 10)) {
        config.clear();
        result.clear();

        RunTableRow(w, -1, 1, 1, 1, H, Q, R, G_m, H_m, &config, &result);

        double rpl = result.config().config().r() * Mean(result.true_true());
        double bpn = Mean(result.bits_per_neuron());
        double d_eff = Mean(result.d_effective());
        double pF = Mean(result.false_true());

        if (0.4 < rpl && rpl < max_rpl && 0.4 < d_eff && pF < 0.1) {
          if (optimal_bpn < bpn) {
            optimal.CopyFrom(result);
            optimal_bpn = Mean(optimal.bits_per_neuron());
          }
        }
        if (0.0 < max_bpn
            && ((bpn < 0.9 * max_bpn && bpn < optimal_bpn)
                || (bpn < 0.1 * max_bpn && optimal_bpn < 0.0)))
          break;  // Early stop due to declining bpn
        if (max_bpn < 0.0 || max_bpn < bpn) max_bpn = bpn;
      }
      if (max_bpn_G < max_bpn) max_bpn_G = max_bpn;
      if (max_bpn < 0.8 * max_bpn_G) break;
    }
    if (max_bpn_G < 0.8 * optimal_bpn) break;
  }
  if (0.0 < optimal_bpn)
    PrintTableResults(optimal);
}

}  // namespace cognon

int main(int argc, char **argv) {
  cognon::PrintTableHeader();

  for (int32 H = 10; H <= 40; H += 10) {
    for (int32 R = 10; R <= 40; R += 10) {
      if (H < R) continue;
      cognon::OptimizeConfiguration(H, R, 2.0);
    }
  }

  for (int32 H = 10; H <= 40; H += 10) {
    for (int32 R = 10; R <= 40; R += 10) {
      if (H < R) continue;
      cognon::OptimizeConfiguration(H, R, 50.0);
    }
  }
  return 0;
}
