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
// This program generates the observed probability of false
// positive, given a neuron has S_m strengthened synapses,
// with error bars.
//
// S = 1000
// C = D1 = D2 = 1
// R = {10, 20, 30, 40}
// G = 1.9
// S_m = 1:900:10
//
// Output is {{S_m, Mean(false_true)}, ErrorBar[Stddev(false_true)]}
//

#include <stdio.h>

#include "bob.h"
#include "cognon.h"
#include "monograph.h"
#include "neuron.h"
#include "wordset.h"

class JobRunConfiguration : public cognon::Job {
 public:
  JobRunConfiguration(const cognon::TrainConfig& config, int32 S_m, double G,
                      cognon::NeuronStatistics* result)
      : config_(config), S_m_(S_m), G_(G), result_(result), temp_() { }
  ~JobRunConfiguration() {
      (*result_) += temp_;
  }
  virtual void Run() {
    // Create a neuron with exactly S_m strengthened synapses.
    cognon::Neuron neuron;
    cognon::Wordset selection;
    neuron.Init(config_.config());
    neuron.StartTraining();
    selection.ConfigFixed(1, neuron.length(), config_.config().d1(), S_m_);
    cognon::Word word = selection.get_word(0);
    for (cognon::Word::const_iterator it = word.begin();
         it != word.end(); ++it) {
      neuron.set_strength(it->first, G_);
      neuron.set_frozen(it->first, true);
    }
    neuron.FinishTraining();

    // Test that neuron to find the false positive rate.
    cognon::Wordset words;
    words.Config(1, neuron.length(), config_.config().d1(),
                 config_.config().r());
    cognon::Bob bob;
    bob.Test(100000, words, neuron, &temp_);
    AddSample(neuron.Q_after(), temp_.mutable_q_after());
    AddSample(neuron.length(), temp_.mutable_synapses_per_neuron());
  }
 private:
  const cognon::TrainConfig& config_;
  int32 S_m_;
  double G_;
  cognon::NeuronStatistics* result_;
  cognon::NeuronStatistics temp_;
};

int main(int argc, char **argv) {
  int32 R[] = {10, 20, 30, 40};
  const double G = 1.9;

  printf("{");
  for (int32 r = 0; r < sizeof(R) / sizeof(int32); ++r) {
    if (0 < r) printf(", ");
    printf("{");
    for (int32 S_m = 1; S_m <= 900; S_m += 10) {
      cognon::TrainConfig config;
      cognon::NeuronStatistics result;
      const int32 repetitions = 30;

      // Neuron configuration parameters
      config.mutable_config()->set_c(1);
      config.mutable_config()->set_d1(1);
      config.mutable_config()->set_d2(1);
      config.mutable_config()->set_h(1000.0 / static_cast<double>(R[r]));
      config.mutable_config()->set_q(1.0);
      config.mutable_config()->set_r(R[r]);

      config.mutable_config()->set_g_m(G);
      config.mutable_config()->set_h_m(G * config.config().h());

      vector<cognon::Job*> jobs(repetitions);
      for (int32 i = 0; i < repetitions; ++i) {
        jobs[i] = new JobRunConfiguration(config, S_m, G, &result);
      }
      cognon::RunParallel(&jobs);

      if (1 < S_m) printf(", ");
      printf("{{%d, %f}, ErrorBar[%f]}", S_m,
             cognon::Mean(result.false_true()),
             cognon::Stddev(result.false_true()));
      fflush(stdout);
    }
    printf("}");
  }
  printf("}\n");

  return 0;
}
