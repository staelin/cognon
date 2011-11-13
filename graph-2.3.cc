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
// This program generates the observed false positive probability
// after training with w words, with error bars.
//
// S = 1000
// C = D1 = D2 = 1
// R = {10, 20, 30, 40}
// G = 1.9
// w = 1:60:1
//
// Output is {{w, Mean(false_true)}, ErrorBar[Stddev(false_true)]}
//

#include <stdio.h>

#include "cognon.h"
#include "monograph.h"

int main(int argc, char **argv) {
  int32 R[] = {10, 20, 30, 40};
  const double G = 1.9;

  printf("{");
  for (int32 r = 0; r < sizeof(R) / sizeof(int32); ++r) {
    if (0 < r) printf(", ");
    printf("{");
    for (int32 w = 1; w <= 60; ++w) {
      cognon::TrainConfig config;
      cognon::NeuronStatistics result;
      const int32 repetitions = 10;

      // Training parameters
      config.set_w(w);

      // Neuron configuration parameters
      config.mutable_config()->set_c(1);
      config.mutable_config()->set_d1(1);
      config.mutable_config()->set_d2(1);
      config.mutable_config()->set_h(1000.0 / static_cast<double>(R[r]));
      config.mutable_config()->set_q(1.0);
      config.mutable_config()->set_r(R[r]);

      config.mutable_config()->set_g_m(G);
      double h_m = config.config().h() * config.config().g_m();
      config.mutable_config()->set_h_m(h_m);

      cognon::RunConfiguration(repetitions, config, &result);

      if (1 < w) printf(", ");
      printf("{{%d, %f}, ErrorBar[%f]}", w,
             cognon::Mean(result.false_true()),
             cognon::Stddev(result.false_true()));
      fflush(stdout);
    }
    printf("}");
  }
  printf("}\n");

  return 0;
}
