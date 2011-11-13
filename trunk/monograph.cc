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

#include <stdio.h>

#include "cognon.h"
#include "monograph.h"

#include "alice.h"
#include "bob.h"
#include "neuron.h"

namespace cognon {

void PrintTableHeader() {
  printf("\"W\",\"num active\","
         "\"C\",\"D1\",\"D2\",\"H\",\"Q\",\"R\",\"G_m\",\"H_m\",\"spn\","
         "\"pL\",\"pL stddev\",\"pF\",\"pF stddev\","
         "\"bpn\",\"bpn stddev\",\"bps\",\"bps stddev\","
         "\"spn after\", \"spn after stddev\","
         "\"R*pL\",\"R*pL stddev\",\"D_eff\",\"D_eff stddev\""
         "\n");
  fflush(stdout);
}

void PrintTableRow(int32 W, int32 active, int32 C, int32 D1, int32 D2,
                   double H, double Q, int32 R, double G_m, double H_m) {
  TrainConfig config;
  NeuronStatistics result;

  RunTableRow(W, active, C, D1, D2, H, Q, R, G_m, H_m, &config, &result);
  PrintTableResults(result);
}

void RunTableRow(int32 W, int32 active, int32 C, int32 D1, int32 D2,
                 double H, double Q, int32 R, double G_m, double H_m,
                 TrainConfig* config, NeuronStatistics* result) {
  const int32 repetitions = 10;

  // Training parameters
  config->set_w(W);
  if (0 < active) config->set_num_active(active);

  // Neuron configuration parameters
  config->mutable_config()->set_c(C);
  config->mutable_config()->set_d1(D1);
  config->mutable_config()->set_d2(D2);
  config->mutable_config()->set_h(H);
  config->mutable_config()->set_q(Q);
  config->mutable_config()->set_r(R);

  if (0.0 < G_m) {
    if (H_m < 0.0) H_m = H * G_m;
    config->mutable_config()->set_g_m(G_m);
    config->mutable_config()->set_h_m(H_m);
  }

  RunConfiguration(repetitions, *config, result);
}

void PrintTableResults(const NeuronStatistics& result) {
  printf("%d,", result.config().w());

  if (result.config().has_num_active()) {
    printf("%d,", result.config().num_active());
  } else {
    printf("-1,");
  }

  printf("%d,", result.config().config().c());
  printf("%d,", result.config().config().d1());
  printf("%d,", result.config().config().d2());
  printf("%f,", result.config().config().h());
  printf("%f,", result.config().config().q());
  printf("%d,", result.config().config().r());

  if (result.config().config().has_g_m()) {
    printf("%f,", result.config().config().g_m());
  } else {
    printf("-1.0,");
  }

  if (result.config().config().has_h_m()) {
    printf("%f,", result.config().config().h_m());
  } else {
    printf("-1.0,");
  }

  printf("%f,", Mean(result.synapses_per_neuron()));

  printf("%f,", Mean(result.true_true()));
  printf("%f,", Stddev(result.true_true()));

  printf("%f,", Mean(result.false_true()));
  printf("%f,", Stddev(result.false_true()));

  printf("%f,", Mean(result.bits_per_neuron()));
  printf("%f,", Stddev(result.bits_per_neuron()));

  double spn = Mean(result.synapses_per_neuron());
  printf("%f,", Mean(result.bits_per_neuron()) / spn);
  printf("%f,", Stddev(result.bits_per_neuron()) / spn);

  printf("%f,", Mean(result.q_after()) * spn);
  printf("%f,", Stddev(result.q_after()) * spn);

  printf("%f,", result.config().config().r() * Mean(result.true_true()));
  printf("%f,", result.config().config().r() * Stddev(result.true_true()));

  printf("%f,", Mean(result.d_effective()));
  printf("%f", Stddev(result.d_effective()));

  printf("\n");
  fflush(stdout);
}

double OptimizeRow(double H, int32 S, int32 C, int32 D1, int32 D2,
                   double G_max, double G_step) {
  double optimal_bpn = -1.0;
  NeuronStatistics optimal;
  NeuronStatistics result;
  TrainConfig config;
  double optimal_Q = 2.0 * D1;

  for (double G_m = G_max; 1.0 <= G_m; G_m -= G_step) {
    double H_m = (double)H * G_m;
    double best_bpn_G = -1.0;
    double best_pL_G = -1.0;
    double best_pF_G = 100.0;
    int32 last_R = -1;
    bool G_had_optimal_result = false;
    double Q = min(2.0 * D1, optimal_Q + (2.0 * D1) / 10.0);
    for (; 0.5 < Q; Q -= static_cast<double>(D1) / 10.0) {
      bool Q_had_optimal_result = false;
      double max_bpn = -1.0;
      double max_pL = -1.0;
      double min_pF = 100.0;
      int32 R = static_cast<int32>(floor(S / (C * H * Q) + kEpsilon));

      if (R <= 1 || 400 < R || R == last_R)
        continue;
      last_R = R;

      double Q_actual = S / static_cast<double>(C * H * R);
      for (int32 w = 10; w <= 10000;
           w += (w < 100 ? 10 : (w < 1000 ? 100 : 1000))) {
        config.clear();
        result.clear();

        RunTableRow(w, -1, C, D1, D2, H, Q_actual, R, G_m, H_m,
                    &config, &result);

        double rpl = result.config().config().r() * Mean(result.true_true());
        double bpn = Mean(result.bits_per_neuron());
        double d_eff = Mean(result.d_effective());
        double pL = Mean(result.true_true());
        double pF = Mean(result.false_true());

        if (max_pL < pL) max_pL = pL;
        if (pF < min_pF) min_pF = pF;

        if (0.4 < d_eff && pF < pL && pF < 0.03 && optimal_bpn < bpn) {
          optimal.CopyFrom(result);
          optimal_bpn = Mean(optimal.bits_per_neuron());
          printf("# optimal "); PrintTableResults(result);
          Q_had_optimal_result = true;
          G_had_optimal_result = true;
          optimal_Q = Q;
        } else {
          printf("# "); PrintTableResults(result);
        }

        if (0.0 < max_bpn
            && ((bpn < 0.9 * max_bpn && bpn < optimal_bpn)
                || (bpn < 0.1 * max_bpn && optimal_bpn < 0.0)))
          break;  // Early stop due to declining bpn
        if (pL < kEpsilon)
          break;  // Early stop due to not learning
        if (0.1 < pF || pL < pF)
          break;  // Early stop due to too many false positives
        if (bpn <= 0.0)
          break;  // Early stop due to no learned information

        if (max_bpn < bpn) max_bpn = bpn;
      }

      if (best_pL_G < max_pL) best_pL_G = max_pL;
      if (best_bpn_G < max_bpn) best_bpn_G = max_bpn;
      if (min_pF < best_pF_G) best_pF_G = min_pF;

      if (max_pL < kEpsilon) break;  // Early stop on Q due to not learning
      if (0.0 < optimal_bpn
          && max_pL < min_pF) break;  // Early stop on Q due to bad learning
      if (0.0 < optimal_bpn
          && Q < optimal_Q
          && !Q_had_optimal_result) break;  // Early stop on Q due to declining
                                            // learning performance
    }
    if (best_bpn_G < 0.7 * optimal_bpn) break;  // Early stop on G
    if (best_pL_G < kEpsilon) break;  // Early stop on G due to not learning
    if (0.0 < optimal_bpn
        && !G_had_optimal_result) break;  // Early stop for G since performance
                                          // is decreasing, not improving
  }
  if (0.0 < optimal_bpn)
    PrintTableResults(optimal);
  return optimal_bpn;
}

void dump_sum(Neuron& neuron, const Word& word) {
  printf("sum = {");
  int s = neuron.slots();
  for (int32 d = 0; d < s; ++d) {
    for (int32 i = 0; i < neuron.C(); ++i) {
      neuron.set_sum(i, 0.0);
    }
    // Iterate over sparse signals in word
    for (Word::const_iterator it = word.begin(); it != word.end(); ++it) {
      int32 synapse = it->first;
      int32 delay = it->second;
      CHECK(0 <= synapse && synapse < neuron.length());
      CHECK(delay == kDisabled || (0 <= delay && delay < neuron.D1()));

      // If delay and word delay adds up to current delay then
      // increment the container's sum by that synapse's strength
      if (neuron.delays(synapse) + delay == d) {
        CHECK(0 <= neuron.containers(synapse)
              && neuron.containers(synapse) < neuron.C());
        neuron.set_sum(neuron.containers(synapse),
                       neuron.sum(neuron.containers(synapse))
                                  + neuron.strength(synapse));
      }
    }

    // Iterate through containers to see if any fired
    if (0 < s) printf("\t");
    printf("s = %d {", d);
    for (int32 i = 0; i < neuron.C(); ++i) {
      if (0 < i)
        printf(", ");
      printf("%f", neuron.sum(i));
    }
    printf("}\n");
  }
}

void DebugTableRow(int32 W, int32 active, int32 C, int32 D1, int32 D2,
                 double H, double Q, int32 R, double G_m, double H_m,
                 TrainConfig* config, NeuronStatistics* result) {
  config->Clear();

  // Training parameters
  config->set_w(W);
  if (0 < active) config->set_num_active(active);

  // Neuron configuration parameters
  config->mutable_config()->set_c(C);
  config->mutable_config()->set_d1(D1);
  config->mutable_config()->set_d2(D2);
  config->mutable_config()->set_h(H);
  config->mutable_config()->set_q(Q);
  config->mutable_config()->set_r(R);

  if (0.0 < G_m) {
    if (H_m < 0.0) H_m = H * G_m;
    config->mutable_config()->set_g_m(G_m);
    config->mutable_config()->set_h_m(H_m);
  }

  Wordset words;
  Neuron neuron;

  // Must have either both or neither of g_m() and h_m()
  CHECK((config->config().has_g_m() && config->config().has_h_m())
        || (!config->config().has_g_m() && !config->config().has_h_m()));

  result->Clear();
  result->mutable_config()->CopyFrom(*config);

  neuron.Init(config->config());
  if (config->has_num_active()) {
    words.ConfigFixed(config->w(), neuron.length(),
                      config->config().d1(), config->num_active());
  } else {
    words.Config(config->w(), neuron.length(),
                 config->config().d1(), config->config().r());
  }

  vector<int32> synapse_before_delay_histogram;
  neuron.GetSynapseDelayHistogram(&synapse_before_delay_histogram);

  Alice alice;
  alice.Train(&words, &neuron);

  // Now start testing
  Bob bob;
  bob.Test(100, words, neuron, result);

  // Collect statistics
  AddSample(neuron.Q_after(), result->mutable_q_after());
  AddSample(neuron.length(), result->mutable_synapses_per_neuron());
}
}  // namespace
