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
// Numerical analysis of various aspects of Cognon performance.  It outputs
// data in the form of Mathematica input.
//

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

const double kEpsilon = 1.0e-6;
const double kFraction = 1.0;

double lchoose(double n, double k);
double lbinomial(double n, double k, double p);
double L(double prob_learn, double prob_false_positive, int w);
double prob_fire(int nSynapses, int nStrongSynapses, double rate,
                 double G, double H);
double prob_false_positive(int nSynapses, int nStrongSynapses, double rate,
                           double G, double H, int w);
double** prob_strengthen_synapses(int nSynapses,
                                  double rate, double G, double H, int w);
double* expected_strengthen_synapses(int nSynapses,
                                     double rate, double G, double H, int w);
void generate_strengthen_synapses(int nSynapses,
                                  double rate, double G, double H, int W);
void generate_false_positive(int nSynapses,
                             double rate, double G, double H, int W);
double mean(int n, double* v);
double** malloc2D(int n, int m);
void free2D(int n, int m, double** ptr);

double lchoose(double n, double k) {
  assert(0.0 <= k && k <= n && !isnan(n) && !isnan(k));

  return lgamma(n + 1) - lgamma(n - k + 1) - lgamma(k + 1);
}

double lbinomial(double n, double k, double p) {
  assert(!isnan(p) && !isnan(n) && !isnan(k));
  assert(0.0 <= k && k <= n && 0.0 <= p && p <= 1.0);

  return (lchoose(n, k) + k * log(p) + (n - k) * log(1.0 - p));
}

double L(double prob_learn, double prob_false_positive, int w) {
  assert(!isnan(prob_learn) && !isnan(prob_false_positive));
  assert(0.0 <= prob_learn && prob_learn <= 1.0);
  assert(0.0 <= prob_false_positive && prob_false_positive <= 1.0);
  assert(0 <= w);

  if (prob_learn <= prob_false_positive)
    return 0.0;

  return ((w / log(2.0)) * (log(1.0 - prob_learn)
                            - log(1.0 - prob_false_positive)
                            + prob_learn * (-log(1.0 - prob_learn)
                                            + log(1.0 - prob_false_positive)
                                            + log(prob_learn)
                                            - log(prob_false_positive))));
}

double prob_fire(int nSynapses, int nStrongSynapses, double rate,
                 double G, double H) {
  double result = 0.0;
  double synapse_rate = nStrongSynapses / (double)nSynapses;
  double* probs = (double*)malloc((nSynapses - nStrongSynapses + 1) * sizeof(double));

  {
    double sum = 0.0;
    for (int i = 0; i <= nSynapses - nStrongSynapses; ++i) {
      probs[i] = exp(lbinomial(nSynapses - nStrongSynapses, i, rate));
      sum += probs[i];
      assert(!isnan(probs[i]) && 0.0 <= probs[i] && probs[i] <= 1.0);
    }
    assert(!isnan(sum) && 0.0 <= sum && sum <= 1.0 + kEpsilon);
  }

  // Iterate over input/active strengthened synapses
  for (int i = 0; i <= nStrongSynapses; ++i) {
    double sum = 0.0;
    // Iterate over input/active original synapses
    for (int j = 0; j <= nSynapses - nStrongSynapses; ++j) {
      if (G * i + j + kEpsilon >= H) {
        sum += probs[j];
        assert(!isnan(sum) && 0.0 <= sum && sum <= 1.0 + kEpsilon);
      }
    }
    if (1.0 < sum) sum = 1.0;
    result += exp(lbinomial(nStrongSynapses, i, rate)) * sum;
    assert(!isnan(result) && 0.0 <= result && result <= 1.0 + kEpsilon);
  }
  assert(!isnan(result) && 0.0 <= result && result <= 1.0 + kEpsilon);
  free(probs);
  if (1.0 < result) result = 1.0;
  return result;
}

double prob_false_positive(int nSynapses, int nStrongSynapses, double rate,
                           double G, double H, int w) {
  double result = prob_fire(nSynapses, nStrongSynapses, rate, G, H);
  result -= w / pow(2.0, nSynapses * rate);
  return (result < 0.0 ? 0.0 : result);
}

double** prob_strengthen_synapses(int nSynapses,
                                  double rate, double G, double H, int w) {
  double** result = malloc2D(w + 1, nSynapses + 1);
  result[0][0] = 1.0;

  double** prob = malloc2D(nSynapses + 1, nSynapses + 1);
  for (int i = 0; i <= nSynapses; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= i; ++j) {
      prob[i][j] = exp(lbinomial(i, j, rate));
      sum += prob[i][j];
      assert(!isnan(prob[i][j]) && 0.0 <= prob[i][j] && prob[i][j] <= 1.0);
    }
    assert(1.0 - kEpsilon < sum && sum < 1.0 + kEpsilon);
  }

  // Iterate over words.
  for (int i = 1; i <= w; ++i) {
    // Iterate over number of previously strengthened synapses.
    for (int j = 0; j <= nSynapses; ++j) {
      // Iterate over input/active strengthened synapses.
      for (int k = 0; k <= j; ++k) {
        double weight = result[i - 1][j] * prob[j][k];
        double* p = prob[nSynapses - j];
        double sum_not_fired = 0.0;
        double sum_fired = 0.0;

        assert(!isnan(weight));

        // Iterate over input/active original synapses.
        // These synapses will be strengthened if the neuron fires.
        //
        for (int l = 0; l <= nSynapses - j; ++l) {
          if (G * k + l + kEpsilon < H) {
            sum_not_fired += p[l];
          } else {
            result[i][j + l] += weight * p[l];
            sum_fired += p[l];
          }
        }
        result[i][j] += weight * sum_not_fired;
        assert(1.0 - kEpsilon <= sum_not_fired + sum_fired
               && sum_not_fired + sum_fired <= 1.0 + kEpsilon);
      }
    }

    // Adjust for rounding errors; re-normalize so sum == 1
    double sum = 0.0;
    for (int j = 0; j <= nSynapses; ++j) {
      sum += result[i][j];
    }
    assert(!isnan(sum) && 0.0 <= sum && sum <= 1.001);
    for (int j = 0; j <= nSynapses; ++j) {
      result[i][j] /= sum;
    }
  }
  free2D(w + 1, nSynapses + 1, prob);
  return result;
}

double* expected_strengthen_synapses(int nSynapses,
                                     double rate, double G, double H, int w) {
  double** probs = prob_strengthen_synapses(nSynapses, rate, G, H, w);
  double* result = (double*)malloc((w + 1) * sizeof(double));

  for (int i = 0; i <= w; ++i) {
    result[i] = mean(nSynapses + 1, probs[i]);
  }
  free2D(w + 1, nSynapses + 1, probs);
  return result;
}

void generate_strengthen_synapses(int nSynapses,
                                  double rate, double G, double H, int W) {
  double* expected = expected_strengthen_synapses(nSynapses, rate, G, H, W);

  printf("{");
  for (int i = 0; i <= W; ++i) {
    if (0 < i) printf(", ");
    printf("{%d,%lf}", i, expected[i]);
  }
  printf("}");
  fflush(stdout);

  free(expected);
}

void generate_false_positive(int nSynapses,
                             double rate, double G, double H, int W) {
  // False probability
  printf("{");
  double** probs = prob_strengthen_synapses(nSynapses, rate, G, H, W);
  for (int i = 1; i <= W; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= nSynapses; ++j) {
      sum += probs[i][j] * prob_false_positive(nSynapses, j, rate, G, G * H, W);
      assert(!isnan(sum) && 0.0 <= sum && sum <= 1.0 + kEpsilon);
    }
    if (1 < i) printf(", ");
    printf("{%d,%lf}", i, sum);
    fflush(stdout);
  }
  printf("}");
  fflush(stdout);

  free2D(W + 1, nSynapses + 1, probs);
}

double mean(int n, double* prob) {
  double result = 0.0;
  double prob_sum = 0.0;

  for (int i = 0; i <= n; ++i) {
    assert(!isnan(prob[i]) && 0.0 <= prob[i] && prob[i] <= 1.0);
    prob_sum += prob[i];

    result += (double)i * prob[i];
  }
  assert(1.0 - kEpsilon <= prob_sum && prob_sum <= 1.0 + kEpsilon);
  return result;
}

double** malloc2D(int n, int m) {
  double** result = (double**)malloc(n * sizeof(double*));
  for (int i = 0; i < n; ++i) {
    result[i] = (double*)malloc(m * sizeof(double));
    for (int j = 0; j < m; ++j) {
      result[i][j] = 0.0;
    }
  }
  return result;
}

void free2D(int n, int m, double** ptr) {
  for (int i = 0; i < n; ++i) {
    free(ptr[i]);
  }
  free(ptr);
}

int main(int argc, char* argv[]) {
  bool bDoSynapseProbabilityDistribution = false;
  bool bDoSynapseStrength = false;
  int doMathematica = -1;
  double G = -1.0;
  double H = -1.0;
  double rate;
  int R = -1;
  int W = -1;
  int c;
  int nSynapses = -1;

  while ((c = getopt(argc, argv, "dsm:G:H:R:S:W:")) != -1) {
    switch (c) {
      case 'd':
        bDoSynapseProbabilityDistribution = true;
        break;
      case 's':
        bDoSynapseStrength = true;
        break;
      case 'm':
        // Generate Mathematica data
        doMathematica = atoi(optarg);
        break;
      case 'G':
        G = atof(optarg);
        break;
      case 'H':
        H = atof(optarg);
        break;
      case 'R':
        R = atoi(optarg);
        break;
      case 'S':
        nSynapses = atoi(optarg);
        break;
      case 'W':
        W = atoi(optarg);
        break;
      default:
        break;
    }
  }
  if (nSynapses < 0) nSynapses = 1000;
  if (R < 0) R = 10;
  if (H < 0.0) H = nSynapses / (double)R;
  if (G < 0.0) G = 1.9;
  if (W < 0) W = 60;
  rate = 1 / (double)R;

  if (0 < doMathematica) {
    switch (doMathematica) {
      case 1:
        // Create a mathematica-friendy dataset for expected strengthened
        // synapses versus W.
        //
        printf("{");
        generate_strengthen_synapses(nSynapses, 1.0 / 10.0, G,
				     kFraction * (nSynapses / 10.0), W);
        printf(",");
        generate_strengthen_synapses(nSynapses, 1.0 / 20.0, G,
				     kFraction * (nSynapses / 20.0), W);
        printf(",");
        generate_strengthen_synapses(nSynapses, 1.0 / 30.0, G,
				     kFraction * (nSynapses / 30.0), W);
        printf(",");
        generate_strengthen_synapses(nSynapses, 1.0 / 40.0, G,
				     kFraction * (nSynapses / 40.0), W);
        printf("}\n");
        break;
      case 3:
        // Create a Mathematica-friendly dataset for probability of
        // false-positive versus W.
        //
        printf("{");
        generate_false_positive(nSynapses, 1.0 / 10.0, G,
				kFraction * (nSynapses / 10.0), W);
        printf(",");
        generate_false_positive(nSynapses, 1.0 / 20.0, G,
				kFraction * (nSynapses / 20.0), W);
        printf(",");
        generate_false_positive(nSynapses, 1.0 / 30.0, G,
				kFraction * (nSynapses / 30.0), W);
        printf(",");
        generate_false_positive(nSynapses, 1.0 / 40.0, G,
				kFraction * (nSynapses / 40.0), W);
        printf("}\n");
        break;
      default:
        break;
    }
    return 0;
  }

  if (bDoSynapseStrength) {
    double* expected = expected_strengthen_synapses(nSynapses, rate, G, H, W);
    for (int i = 0; i <= W; ++i) {
      printf("%d,%lf\n", i, expected[i]);
    }
    free(expected);
    return 0;
  }

  if (bDoSynapseProbabilityDistribution) {
    double** probs = prob_strengthen_synapses(nSynapses, rate, G, H, W);
    for (int i = 0; i <= nSynapses; ++i) {
      printf("%d,%lf\n", i, probs[W][i]);
    }
    free2D(W + 1, nSynapses + 1, probs);
    return 0;
  }

  // False probability
  double** probs = prob_strengthen_synapses(nSynapses, rate, G, H, W);
  for (int i = 1; i <= W; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= nSynapses; ++j) {
      sum += probs[i][j] * prob_false_positive(nSynapses, j, rate, G, G * H, W);
      // if (i == W) {
      //   printf("sum = %lf, probs[%d][%d] = %lf, prob_false_positive = %lf\n",
      //          sum, i, j,
      //          probs[i][j], prob_false_positive(nSynapses, j, rate, G, G * H, W));
      // }
    }
    printf("%d,%lf\n", i, sum);
  }
  free2D(W + 1, nSynapses + 1, probs);
}
