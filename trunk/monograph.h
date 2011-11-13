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
#ifndef COGNON_MONOGRAPH_H_
#define COGNON_MONOGRAPH_H_

#include "cognon.h"

namespace cognon {

void PrintTableHeader();

void PrintTableRow(int32 W, int32 active, int32 C, int32 D1, int32 D2,
                   double H, double Q, int32 R,
                   double G_m = -1.0, double H_m = -1.0);

void RunTableRow(int32 W, int32 active, int32 C, int32 D1, int32 D2,
                 double H, double Q, int32 R, double G_m, double H_m,
                 TrainConfig* config, NeuronStatistics* result);

void PrintTableResults(const NeuronStatistics& result);

double OptimizeRow(double H, int32 S, int32 C, int32 D1, int32 D2,
                   double G_max, double G_step);

void DebugTableRow(int32 W, int32 active, int32 C, int32 D1, int32 D2,
                 double H, double Q, int32 R, double G_m, double H_m,
                 TrainConfig* config, NeuronStatistics* result);
}  // namespace

#endif  // COGNON_MONOGRAPH_H_
