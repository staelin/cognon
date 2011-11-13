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

#include "cognon.h"
#include "monograph.h"

int main(int argc, char **argv) {
  cognon::PrintTableHeader();

  int32 S[] = {200, 1000, 10000};
  int32 C = 1;
  int32 D1 = 1;
  int32 D2 = 1;
  double G_max[] = {4.0, 1.9};
  double G_step[] = {0.2, 0.1};
  CHECK(sizeof(G_max) == sizeof(G_step));

  for (int32 s = 0; s < sizeof(S) / sizeof(int32); ++s) {
    for (int32 g = 0; g < sizeof(G_max) / sizeof(double); ++g) {
      double optimal_bpn = -1.0;
      double start_H = 5.0;
      double max_H = 0.9 * (S[s] / static_cast<double>(C));
      double step_H = 5;
      for (double H = start_H; H <= max_H + cognon::kEpsilon; H += step_H) {
        printf("# OptimizeRow(%f, %d, %d, %d, %d, %f, %f)\n",
               H, S[s], C, D1, D2, G_max[g], G_step[g]);
        double bpn = cognon::OptimizeRow(H, S[s], C, D1, D2,
                                         G_max[g], G_step[g]);
        if (sqrt(S[s] / (double)C) < H
            && cognon::kEpsilon < optimal_bpn && bpn < 0.3 * optimal_bpn) {
          step_H = (max_H - start_H) / 20.0;
        } else if (optimal_bpn * 0.8 < bpn && step_H != 5.0) {
          step_H = 5.0;
        }
        if (optimal_bpn < bpn) optimal_bpn = bpn;
      }
    }
  }
  return 0;
}
