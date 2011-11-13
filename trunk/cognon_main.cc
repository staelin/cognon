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
// Reads a CSV (comma-separated-value) file that specifies simulator
// configurations to be run.
//
// Input line is in the form of (default):
//
//    "W","num active","C","D1","D2","H","Q","R","G_m","H_m"
//
// Or (when optimize option is selected):
//
//    "H","S","C","D1","D2","G_max","G_step"
//
// A value of -1 means "unspecified".
// A numeric value means that single value.
// A string value with a list of values, e.g. "10,20,30", means
// multiple configurations, one with each value.
//

#include <stdlib.h>

#include <fstream>      // fstream
#include <iostream>     // cout, endl
#include <map>
#include <string>
#include <utility>
#include <vector>
using namespace std;

#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
using namespace boost;

#include "cognon.h"
#include "monograph.h"
using namespace cognon;

typedef tokenizer< escaped_list_separator<char> > Tokenizer;

namespace cognon {

void ParseList(string s, vector<string>* values) {
    Tokenizer tokens(s);
    values->assign(tokens.begin(), tokens.end());
}

void ParseInt(string s, vector<int32>* values) {
  if (s.find_first_of(',') != string::npos) {
    vector<string> list;
    ParseList(s, &list);
    for (int32 i = 0; i < list.size(); ++i) {
      ParseInt(list[i], values);
    }
  } else {
    int32 v;
    if (sscanf(s.c_str(), "%d", &v) == 1) {
      values->push_back(v);
    }
  }
}

void ParseDouble(string s, vector<double>* values) {
  if (s.find_first_of(',') != string::npos) {
    vector<string> list;
    ParseList(s, &list);
    for (int32 i = 0; i < list.size(); ++i) {
      ParseDouble(list[i], values);
    }
  } else {
    double v;
    if (sscanf(s.c_str(), "%lf", &v) == 1) {
      values->push_back(v);
    }
  }
}

void ParseFile(const char* fname) {
  ifstream in(fname);
  if (!in.is_open()) return;

  PrintTableHeader();

  string line;
  while (getline(in, line)) {
    if (line.find_first_of('#') != string::npos) continue;  // comment line

    Tokenizer tokens(line);
    vector<string> values;
    values.assign(tokens.begin(), tokens.end());

    // "W", "num active","C","D1","D2","H","Q","R","G_m","H_m","S"
    vector<int32> W;
    vector<int32> num_active;
    vector<int32> C;
    vector<int32> D1;
    vector<int32> D2;
    vector<double> H;
    vector<double> Q;
    vector<double> R;
    vector<double> G_m;
    vector<double> H_m;
    vector<int32> S;

    if (values.size() < 2) continue;
    if (values.size() < 10) {
      printf("Not enough values in input line: %s\n", line.c_str());
      exit(1);
    }

    if (0 < values.size()) ParseInt(values[0], &W);
    if (1 < values.size()) ParseInt(values[1], &num_active);
    if (2 < values.size()) ParseInt(values[2], &C);
    if (3 < values.size()) ParseInt(values[3], &D1);
    if (4 < values.size()) ParseInt(values[4], &D2);
    if (5 < values.size()) ParseDouble(values[5], &H);
    if (6 < values.size()) ParseDouble(values[6], &Q);
    if (7 < values.size()) ParseDouble(values[7], &R);
    if (8 < values.size()) ParseDouble(values[8], &G_m);
    if (9 < values.size()) ParseDouble(values[9], &H_m);
    if (10 < values.size()) ParseInt(values[10], &S);

    for (int32 w = 0; w < W.size(); ++w) {
      for (int32 a = 0; a < num_active.size(); ++a) {
	for (int32 c = 0; c < C.size(); ++c) {
	  for (int32 d1 = 0; d1 < D1.size(); ++d1) {
	    for (int32 d2 = 0; d2 < D2.size(); ++d2) {
	      for (int32 h = 0; h < H.size(); ++h) {
		for (int32 q = 0; q < Q.size(); ++q) {
		  for (int32 r = 0; r < R.size(); ++r) {
		    for (int32 g_m = 0; g_m < 1 || g_m < G_m.size(); ++g_m) {
		      for (int32 h_m = 0; h_m < 1 || h_m < H_m.size(); ++h_m) {
                        for (int32 s = 0; s < S.size(); s++) {
                          if (Q[q] <= 0.0 && 0 < S[s]) {
                            double q_ = (S[s] + kEpsilon);
                            q_ /= (C[c] * H[h] * R[r]);
                            PrintTableRow(W[w], num_active[a], C[c],
                                          D1[d1], D2[d2],
                                          H[h], q_, R[r],
                                          (G_m.size() == 0 ? -1.0 : G_m[g_m]),
                                          (H_m.size() == 0 ? -1.0 : H_m[h_m]));
                          } else if (0.0 < Q[q] && S[s] <= 0) {
                            PrintTableRow(W[w], num_active[a], C[c],
                                          D1[d1], D2[d2],
                                          H[h], Q[q], R[r],
                                          (G_m.size() == 0 ? -1.0 : G_m[g_m]),
                                          (H_m.size() == 0 ? -1.0 : H_m[h_m]));
                          } else {
                            printf("Specify either Q or S, and the other as -1\n");
                            exit(1);
                          }
                        }
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
}

void ParseOptimal(const char* fname) {
  ifstream in(fname);
  if (!in.is_open()) return;

  string line;

  while (getline(in, line)) {
    if (line.find_first_of('#') != string::npos) continue;  // comment line

    Tokenizer tokens(line);
    vector<string> values;
    values.assign(tokens.begin(), tokens.end());

    // "H","S","C","D1","D2","G_max","G_step"
    vector<double> H;
    vector<int32> S;
    vector<int32> C;
    vector<int32> D1;
    vector<int32> D2;
    vector<double> G_max;
    vector<double> G_step;

    if (values.size() < 2) continue;
    if (values.size() < 5) {
      printf("Not enough values in input line: %s\n", line.c_str());
      exit(1);
    }

    if (0 < values.size()) ParseDouble(values[0], &H);
    if (1 < values.size()) ParseInt(values[1], &S);
    if (2 < values.size()) ParseInt(values[2], &C);
    if (3 < values.size()) ParseInt(values[3], &D1);
    if (4 < values.size()) ParseInt(values[4], &D2);
    if (5 < values.size()) ParseDouble(values[5], &G_max);
    if (6 < values.size()) ParseDouble(values[6], &G_step);

    for (int32 h = 0; h < H.size(); ++h) {
      for(int32 s = 0; s < S.size(); ++s) {
	for (int32 c = 0; c < C.size(); ++c) {
	  for (int32 d1 = 0; d1 < D1.size(); ++d1) {
	    for (int32 d2 = 0; d2 < D2.size(); ++d2) {
	      for (int32 g_max = 0;
		   g_max < 1 || g_max < G_max.size(); ++g_max) {
		for (int32 g_step = 0;
		     g_step < 1 || g_step < G_step.size(); ++g_step) {
		  OptimizeRow(H[h], S[s], C[c], D1[d1], D2[d2],
			      (G_max.size()  == 0 ? 1.9 : G_max[g_max]),
			      (G_step.size() == 0 ? 0.1 : G_step[g_step]));
		}
	      }
	    }
	  }
	}
      }
    }
  }
}

}  // namespace cognon

int main(int argc, char* argv[]) {
  bool optimize = false;
  ::scoped_ptr<RandomBase> r(cognon::CreateRandom());

  int c;
  while((c = getopt(argc, argv, "c")) != EOF) {
    switch (c) {
    case 'c':
      optimize = true;
      break;
    default:
      fprintf(stderr, "Unknown option %c\n", c);
      exit(1);
      break;
    }
  }

  // Any remaining arguments are the input filenames (or patterns);
  // Parse them to generate the various configurations that will be run.
  for (int i = optind; i < argc; i++) {
    if (optimize) {
      ParseOptimal(argv[i]);
    } else {
      ParseFile(argv[i]);
    }
  }
  return 0;
}
