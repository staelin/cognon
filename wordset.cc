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

#include "wordset.h"

#include <algorithm>
#include <set>

#include "cognon.h"

namespace cognon {

// Assumes that pairs are sorted by synapse
bool operator<(const Word& a, const Word& b) {
  vector<pair<int32, int32> >::const_iterator it_a = a.begin();
  vector<pair<int32, int32> >::const_iterator it_b = b.begin();

  for (; it_a != a.end() && it_b != b.end(); ++it_a, ++it_b) {
    if (*it_a < *it_b) return true;
    if (*it_b < *it_a) return false;
    if ((*it_a).first < (*it_b).first) return true;
    if ((*it_b).first < (*it_a).first) return false;
    if ((*it_a).second < (*it_b).second) return true;
    if ((*it_b).second < (*it_a).second) return false;
  }
  if (it_a == a.end() && it_b != b.end()) return true;
  return false;
}

bool operator<(const pair<int32, int32>& a, const pair<int32, int32>& b) {
  if (a.first < b.first) return true;
  if (a.first == b.first && a.second < b.second) return true;
  return false;
}

Wordset::Wordset()
    : num_words_(-1), word_length_(-1), num_delays_(-1), refractory_period_(-1),
      num_active_(-1), words_(), delays_(), random_(CreateRandom()) {
}

void Wordset::Config(int32 num_words, int32 word_length,
                     int32 num_delays, int32 refractory_period) {
  num_words_ = num_words;
  word_length_ = word_length;
  num_delays_ = num_delays;
  refractory_period_ = refractory_period;

  Init();
}

void Wordset::ConfigFixed(int32 num_words, int32 word_length,
                          int32 num_delays, int32 num_active) {
  num_active_ = num_active;
  Config(num_words, word_length, num_delays, -1);
}

void Wordset::CopyFrom(int32 num_words, const Wordset& other) {
  if (0 < other.refractory_period()) {
    Config(num_words,
           other.word_length(), other.num_delays(), other.refractory_period());
  } else {
    ConfigFixed(num_words,
                other.word_length(), other.num_delays(), other.num_active());
  }
}

void Wordset::Init() {
  CHECK(0 < refractory_period_ || 0 < num_active_);

  if (0 < refractory_period_) InitOrig();
  else InitFixed();
}

void Wordset::InitOrig() {
  // Calls to random_->Rand64() in this routine consumed the majority
  // of CPU time in the whole program.  Try to reduce the number of
  // calls to random_ by getting a big number, and then using a few
  // bits at a time from it.
  //
  int32 nbits = 0;
  int32 valid_bits = 64;
  uint64 scratch = random_->Rand64();

#define RANDOM_UPDATE                                           \
  valid_bits -= nbits;                                          \
  scratch = (scratch >> nbits) ^ (scratch << (61 - nbits));     \
  if (valid_bits <= nbits + 3) {                                \
      scratch = random_->Rand64();                              \
      valid_bits = 64;                                          \
  }

  // Count how many bits are in refractory_period_ and num_delays_
  for (uint32 tmp = max(refractory_period_, num_delays_); tmp; tmp >>= 1)
    ++nbits;

  if (words_.size() != num_words_) words_.resize(num_words_);
  if (delays_.size() != num_words_) delays_.resize(num_words_);

  // TODO(staelin): Need to change how this works so it does
  // less work.  Right now it is looping over every synapse
  // in every word, doing a random check to see if the synapse
  // should be enabled and if so randomly choosing the delay.
  // Since the refractory period is generally so large, we
  // should simply randomly choose the number of synapses to
  // enable, and then randomly choose that many synapses.
  //

  for (int32 i = 0; i < num_words_; ++i) {
    Word& word = words_[i];
    delays_[i] = kDisabled;
    word.clear();
    for (int32 j = 0; j < word_length_; ++j) {
      RANDOM_UPDATE;
      if ((scratch % refractory_period_) == 0) {
        RANDOM_UPDATE;
        pair<int32, int32> value(j, scratch % num_delays_);
        word.push_back(value);
      }
    }
  }
#undef RANDOM_UPDATE
}

void Wordset::InitFixed() {
  set<int32> active;
  if (words_.size() != num_words_) words_.resize(num_words_);
  if (delays_.size() != num_words_) delays_.resize(num_words_);
  CHECK(0 < num_active_ && num_active_ < word_length_);

  for (int32 i = 0; i < num_words_; ++i) {
    Word& word = words_[i];
    delays_[i] = kDisabled;
    word.clear();
    active.clear();
    for (int32 j = 0; j < num_active_; ++j) {
      int32 k;
      while (active.find(k = random_->Rand64() % word_length_) != active.end());
      pair<int32, int32> value(k, random_->Rand64() % num_delays_);
      word.push_back(value);
      active.insert(k);
    }
    sort(word.begin(), word.end());
  }
}

void Wordset::set_size(int32 num_words) {
  num_words_ = num_words;
  Init();
}

// Get the trained delay slot for a given word
int32 Wordset::delay(int32 word) const {
  if (0 <= word && word < delays_.size())
    return delays_[word];
  return kDisabled;
}

// Set the trained delay slot for a given word
int32 Wordset::set_delay(int32 word, int32 delay) {
  if (0 <= word && word < delays_.size())
    return (delays_[word] = delay);
  return kDisabled;
}

}  // namespace cognon
