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

void TableRow(int32 N, int32 H, int32 S, int32 w, double G) {
  // R is implicitly 1 for this table.
  // Q = S / (H * R) = S / (H * 1) = S / H
  //
  cognon::PrintTableRow(w, N, 1, 1, 1, (double)H, S / (double)H, 1, G);
}

int main(int argc, char **argv) {
  cognon::PrintTableHeader();
  TableRow(4,   4,    10,  1, 100.0);
  TableRow(5,   4,    10,  1, 100.0);
  TableRow(4,   4,    10,  2, 100.0);
  TableRow(10, 10,   100,  4, 100.0);
  TableRow(11, 10,   100,  4, 100.0);
  TableRow(11, 10,   100,  5, 100.0);
  TableRow(11, 10,  1000, 15, 100.0);
  TableRow(11, 10,  1000, 60, 100.0);
  TableRow(11, 10, 10000, 160, 100.0);
  TableRow(11, 10, 10000, 600, 100.0);
  TableRow(22, 20, 10000, 450, 100.0);

  TableRow(10, 10,   100,  6, 1.5);
  // TableRow(12, 10,  1000,  20, 1.5);
  TableRow(11, 10,  1000, 15, 1.5);
  TableRow(12, 10,  1000,  15, 1.5);
  TableRow(11, 10, 10000, 160, 1.5);
  TableRow(12, 10, 10000, 160, 1.5);
  TableRow(14, 10, 10000,  10, 1.5);

  return 0;
}
