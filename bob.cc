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

#include <math.h>

#include <set>

#include "bob.h"
#include "cognon.h"
#include "neuron.h"
#include "wordset.h"

namespace cognon {

Bob::Bob() { }

void Bob::Test(int32 num_test_words,
               Wordset& words, Neuron& neuron, NeuronStatistics* stats) {
  // Confusion matrix: <ground_truth_value>_<neuron_result_value>
  //
  int32 true_false = 0;
  int32 true_true = 0;
  int32 false_false = 0;
  int32 false_true = 0;

  TestTrainingSet(words, neuron, &true_true, &true_false);
  double total = static_cast<double>(true_true + true_false);
  AddSample(true_true / total, stats->mutable_true_true());
  AddSample(true_false / total, stats->mutable_true_false());
  AddSample(total, stats->mutable_true_count());

  TestTestSet(words, neuron, num_test_words, &false_true, &false_false);
  total = static_cast<double>(false_true + false_false);
  AddSample(false_true / total, stats->mutable_false_true());
  AddSample(false_false / total, stats->mutable_false_false());
  AddSample(total, stats->mutable_false_count());

  AddSample(BitsPerNeuron(words.size(), true_true, true_false,
                          false_true, false_false),
            stats->mutable_bits_per_neuron());
  AddSample(BitsPerNeuron(words.size(), true_true, true_false,
                          false_true, false_false)
            / static_cast<double>(neuron.config().r()),
            stats->mutable_bits_per_neuron_per_refractory_period());
  AddSample(MutualInformation(neuron, true_true, true_false,
                              false_true, false_false),
            stats->mutable_mutual_information());
}

void Bob::TestTrainingSet(const Wordset& words, Neuron& neuron,
                          int32* true_true, int32* true_false) {
  for (int32 i = 0; i < words.size(); ++i) {
    int32 slot = neuron.Expose(words.get_word(i));
    if (0 <= slot && slot == words.delay(i) && slot < neuron.slots()) {
      ++*true_true;
    } else {
      ++*true_false;
    }
  }
}

// TODO(staelin): currently this does not check that the
// randomly generated test words are not also trained
// words (words in the training set).
//
void Bob::TestTestSet(Wordset& words, Neuron& neuron,
                      int32 num_test_words,
                      int32* false_true, int32* false_false) {
  set<Word> training;
  Wordset test;

  test.CopyFrom(1, words);

  // Remember what words the neuron was trained on
  for (int32 i = 0; i < words.size(); ++i) {
    training.insert(words.get_word(i));
  }

  for (int32 i = 0; i < num_test_words; ++i) {
    test.Init();
    while (training.find(test.get_word(0)) != training.end()) {
      test.Init();
    }
    int32 slot = neuron.Expose(test.get_word(0));
    if (0 <= slot && slot < neuron.slots()) {
      ++*false_true;
    } else {
      ++*false_false;
    }
  }
}

double Bob::BitsPerNeuron(int32 num_words,
                          int32 true_true, int32 true_false,
                          int32 false_true, int32 false_false) {
  // Calculate the false alarm and learning probabilities
  double prob_false = (1.0 / 360.0)
      + false_true / static_cast<double>(false_false + false_true);
  double prob_learn = true_true / static_cast<double>(true_true + true_false);

  if (prob_learn < prob_false) return 0.0;

  if (0.999999 <= prob_false) prob_false = 0.999999;
  if (0.999999 <= prob_learn) prob_learn = 0.999999;
  if (prob_false < 0.0000001) prob_false = 0.0000001;
  if (prob_learn < 0.0000001) prob_learn = 0.0000001;

  // Calculate bits
  double infoValue = log(1.0 - prob_learn)
      - log(1.0 - prob_false)
      - prob_learn * log(1.0 - prob_learn)
      + prob_learn * log(1.0 - prob_false)
      + prob_learn * log(prob_learn)
      - prob_learn * log(prob_false);

  // Return the calculated bits
  return (num_words * infoValue) / log(2.0);
}

// Return the (natural) log of the choose function
inline double lchoose(double n, double k) {
    return lgamma(n + 1) - lgamma(n - k + 1) - lgamma(k + 1);
}

double Bob::MutualInformation(const Neuron& neuron,
                              int32 true_true, int32 true_false,
                              int32 false_true, int32 false_false) {
  if (true_true == 0) return 0.0;

  double Z = pow(2.0, static_cast<double>(neuron.length()));
                                      // Number of possible words
  double num_words = true_true + true_false;
                                      // Number of words Alice tried to teach
  double num_words_learned = true_true;
                                      // Number of words learned by neuron
  double prob_learn = true_true / static_cast<double>(true_true + true_false);
                                      // Probability of successfully training
                                      // a word (Wl / W)
  double prob_false = false_true / static_cast<double>(false_true
                                                       + false_false);
                                      // Probability of a false positive

  // There is always some probability of a false positive
  if (prob_false < 0.0000001) prob_false = 0.000001;

  double result = lchoose(Z * prob_learn, num_words_learned);
  result -= lchoose((Z - num_words) * prob_false + num_words_learned,
                    (Z - num_words) * prob_false);
  result /= log(2.0);

  if (1 < neuron.D1()) {
    double Ild = 0.0;
    result += num_words_learned * log2(neuron.D1()) - Ild;
  }
  return (result < 0.0 ? 0.0 : result);
}

}  // namespace cognon
