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
// Alice trains a given neuron to (hopefully) recognize
// a given set of words.  It iterates over the Wordset
// training the neuron to recognize a word at a time.
// It stores the learned response (the delay at which the
// neuron fired) in the wordset.
//
#ifndef COGNON_ALICE_H_
#define COGNON_ALICE_H_

#include <vector>

#include "cognon.h"

namespace cognon {

class Neuron;
class Wordset;

// Alice is used to train a neuron on a given set of words.
// The caller retains ownership of both neuron and words, but
// must ensure that they both outlive Alice.
//
// Alice is not thread safe.
//
class Alice {
 public:
  Alice();
  virtual ~Alice() { }

  void Train(Wordset* words, Neuron* neuron);

  void TrainHistogram(Wordset* words,
                      Neuron* neuron,
                      vector<int32>* delay_histogram,
                      vector<int32>* input_delay_histogram,
                      vector<int32>* input_max_sum_delay_histogram,
                      vector<int32>* H_histogram);
};

}  // namespace cognon

#endif  // COGNON_ALICE_H_
