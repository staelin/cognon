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
// A Wordset is essentially a sparse vector containing
// the delays for each input signal for each synapse.
// Synapses which do not receive a signal do not have
// delay values, although such non-existent signals may
// be represented by the value "disabled()".
//
// Words could be implemented using STL map<>, except
// that each time a word needed to be randomized it
// generated a large number of malloc()/free() operations
// which killed performance.  Instead it is implemented
// as a vector of <offset, delay> pairs.  In the
// current implementation the pairs are sorted by <offset>,
// but this need not be the case.
//
// Example usage when testing a trained neuron might look
// like:
//
// TrainConfig config;  // This is initialized externally.
// Neuron neuron;       // This was trained externally.
// Wordset wordset;
//
// wordset.Config(num_words,
//                neuron.length(),
//                neuron.D1(),
//                neuron.R());
// for (int32 i = 0; i < wordset.size(); ++i) {
//   int32 delay = neuron.Expose(wordset.get_word(i));
// }
//
#ifndef COGNON_WORDSET_H_
#define COGNON_WORDSET_H_


#include <utility>
#include <vector>

#include "cognon.h"

namespace cognon {

typedef vector<pair<int32, int32> > Word;
bool operator<(const Word& a, const Word& b);
bool operator<(const pair<int32, int32>& a, const pair<int32, int32>& b);

class Wordset {
 public:
  Wordset();
  ~Wordset() { }

  // Configure the wordset
  void Config(int32 num_words, int32 word_length,
              int32 num_delays, int32 refractory_period);
  void ConfigFixed(int32 num_words, int32 word_length,
                   int32 num_delays, int32 num_active);

  // Copy the configuration from anther wordset, except for num_words
  void CopyFrom(int32 num_words, const Wordset& other);

  // Randomize word vector according to configuration
  void Init();

  const int32 size() const { return words_.size(); }
  void set_size(int32 num_words);

  int32 word_length() const { return word_length_; }
  int32 num_delays() const { return num_delays_; }

  const Word& get_word(int32 i) const { return words_[i]; }
  void set_word(int32 i, const Word& word) { words_[i] = word; }

  // get/set the trained delay slot for a given word
  int32 delay(int32 word) const;
  int32 set_delay(int32 word, int32 delay);

  // get/set the refractory period
  int32 refractory_period() const { return refractory_period_; }
  void set_refractory_period(int32 refractory_period) {
    refractory_period_ = refractory_period;
  }

  // get/set the number of active signals for a given word
  int32 num_active() const { return num_active_; }
  void set_num_active(int32 num_active) { num_active_ = num_active; }

 protected:
  int num_words_;  // Number of words
  int word_length_;  // Length of words (#synapses in neuron)
  int num_delays_;  // Number of delays
  int refractory_period_;  // Refractory period
  int num_active_;  // Number of active signals
  vector<Word> words_;
  vector<int32> delays_;
  scoped_ptr<RandomBase> random_;  // Pointer to random number generator

  void InitOrig();
  void InitFixed();
};

}  // namespace cognon

#endif  // COGNON_WORDSET_H_
