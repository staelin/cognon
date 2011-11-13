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
// Bob is given a trained neuron and the training wordset, and
// it tests the neuron to evaluate and measure its learning
// performance.
//
#ifndef COGNON_BOB_H_
#define COGNON_BOB_H_


#include "cognon.h"

namespace cognon {

class Neuron;
class Wordset;
class NeuronStatistics;

// Only in bob_test.cc for accessing private functions
class TestBob;

// Bob is used to test a neuron to see how well it learned the
// wordset.  It also checks the neuron's responses to words that
// weren't learned during training.  It generates the full
// confusion matrix and collected the neuron statistics.
//
// The caller retains ownership of config, neuron, and stats,
// but must ensure that they all outlive Bob.
//
class Bob {
 public:
  Bob();
  virtual ~Bob() { }

  // Take a neuron trained on words and evaluate
  void Test(int num_test_words,
            Wordset& words, Neuron& neuron, NeuronStatistics* stats);

 private:
  // Given a neuron and a set of (hopefully) learned words,
  // collect the confusion matrix statistics vis-a-vis the
  // learned words.
  //
  void TestTrainingSet(const Wordset& words, Neuron& neuron,
                       int32* true_true, int32* true_false);

  // Given a neuron and a set of (hopefully) learned words,
  // collect the confusion matrix statistics for words which
  // were not learned and should not be recognized.
  //
  void TestTestSet(Wordset& words, Neuron& neuron,
                   int32 num_test_words,
                   int32* false_true, int32* false_false);

  // This function calculates the information stored by a single neuron
  double BitsPerNeuron(int32 num_words,
                       int32 true_true, int32 true_false,
                       int32 false_true, int32 false_false);
  double MutualInformation(const Neuron& neuron,
                           int32 true_true, int32 true_false,
                           int32 false_true, int32 false_false);

  friend class TestBob;
};
}  // namespace cognon

#endif  // BOB_H_
