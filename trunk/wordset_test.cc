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

#include <sys/param.h>

#include <set>

#include "cognon.h"

#define ABS(a) ((a) < 0.0 ? -(a) : (a))
#define FLOAT_EQ(a, b) \
  (ABS((a) - (b)) / (MIN(ABS(a), ABS(b)) + 1.0e-6) < 1.0e-6)
#define EXPECT_FEQ(a, b) EXPECT_TRUE(FLOAT_EQ(a, b))

namespace cognon {

class WordsetTest : public testing::Test {
};

// Check that each synapse appears in the word at most once
void check_synapses_are_unique(const Word& word) {
  set<int32> synapses;

  for (Word::const_iterator it = word.begin(); it != word.end(); ++it) {
    int synapse = it->first;

    EXPECT_TRUE(synapses.find(synapse) == synapses.end());
    synapses.insert(synapse);
  }
}

TEST_F(WordsetTest, CheckWordset) {
  Wordset w;
  const int32 nwords = 1000;
  const int32 nsynapses = 1000;
  const int32 ndelays = 10;
  const int32 R = 20;
  vector<int32> delay_count;

  delay_count.resize(ndelays + 1);

  EXPECT_EQ(0, w.size()) << "w: Expect w.size() 0: " << w.size();

  w.Config(nwords, nsynapses, ndelays, R);
  EXPECT_EQ(nwords, w.size())
      << "w: Expect w.size " << nwords << ": " << w.size();
  EXPECT_TRUE(kDisabled < 0 || ndelays <= kDisabled)
      << "w: Expected invalid delay value for kDisabled: " << kDisabled;
  EXPECT_TRUE(kDisabled + kDisabled < 0
              || ndelays <= kDisabled + kDisabled)
      << "w: Expected invalid delay value for sum of kDisabled: "
      << kDisabled;

  // Check that we get the right number of active synapses per word
  int64 sum = 0;
  for (int32 i = 0; i < w.size(); ++i) {
    const Word& word = w.get_word(i);
    check_synapses_are_unique(word);
    EXPECT_EQ(kDisabled, w.delay(i))
        << "w: Expect delay disabled: " << w.delay(i);
    for (Word::const_iterator it = word.begin(); it != word.end(); ++it) {
      int synapse = it->first;
      int32 v = it->second;
      EXPECT_TRUE((0 <= v && v < ndelays) || v == kDisabled)
          << "w: Expect w.get_word(" << i << ")(" << synapse << ") value: "
          << v;
      if (0 <= v && v < ndelays) sum++;
      if (v != kDisabled) delay_count[v]++;
    }
  }
  double R_actual = (nwords * nsynapses) / static_cast<double>(sum);
  EXPECT_LE(R_actual, 1.2 * R)
      << "w: Expect R_actual <= 1.2 * " << R << ": " << R_actual;
  EXPECT_LE(0.8 * R, R_actual)
      << "w: Expect 0.8 * " << R << " <= R_actual: " << R_actual;

  // Check that the delays are uniformly distributed (not skewed)
  for (int32 i = 0; i < ndelays; ++i) {
    EXPECT_LE(delay_count[i], 1.2 * (nwords * nsynapses) / (ndelays * R))
        << "Expected even distribution of delays";
    EXPECT_LE(0.8 * (nwords * nsynapses) / (ndelays * R), delay_count[i])
        << "Expected even distribution of delays";
  }

  for (int32 i = 0; i < w.size(); ++i) {
    w.set_delay(i, i % ndelays);
  }
  for (int32 i = 0; i < w.size(); ++i) {
    EXPECT_EQ(i % ndelays, w.delay(i))
        << "w: Expect delay(" << i << ") " << i % ndelays << ": " << w.delay(i);
  }
}

TEST_F(WordsetTest, CheckWordsetFixed) {
  Wordset w;
  const int32 nwords = 1000;
  const int32 nsynapses = 1000;
  const int32 ndelays = 10;
  const int32 R = 20;
  vector<int32> delay_count;

  delay_count.resize(ndelays + 1);

  EXPECT_EQ(0, w.size()) << "w: Expect w.size() 0: " << w.size();

  w.ConfigFixed(nwords, nsynapses, ndelays, 50);
  EXPECT_EQ(nwords, w.size())
      << "w: Expect w.size " << nwords << ": " << w.size();
  EXPECT_TRUE(kDisabled < 0 || ndelays <= kDisabled)
      << "w: Expected invalid delay value for kDisabled: " << kDisabled;
  EXPECT_TRUE(kDisabled + kDisabled < 0
              || ndelays <= kDisabled + kDisabled)
      << "w: Expected invalid delay value for sum of kDisabled: "
      << kDisabled;

  // Check that we get the right number of active synapses per word
  int64 sum = 0;
  for (int32 i = 0; i < w.size(); ++i) {
    int32 active = 0;
    const Word& word = w.get_word(i);
    check_synapses_are_unique(word);
    EXPECT_EQ(kDisabled, w.delay(i))
        << "w: Expect delay disabled: " << w.delay(i);
    for (Word::const_iterator it = word.begin(); it != word.end(); ++it) {
      int synapse = it->first;
      int32 v = it->second;
      EXPECT_TRUE((0 <= v && v < ndelays) || v == kDisabled)
          << "w: Expect w.get_word(" << i << ")(" << synapse << ") value: "
          << v;
      if (0 <= v && v < ndelays) {
        sum++;
        active++;
      }
      if (v != kDisabled) delay_count[v]++;
    }
    EXPECT_EQ(active, w.num_active())
        << "w: Expect number of active synapses to be " << w.num_active()
        << ", actual value is " << active;
  }

  // Check that the delays are uniformly distributed (not skewed)
  for (int32 i = 0; i < ndelays; ++i) {
    EXPECT_LE(delay_count[i], 1.2 * (nwords * nsynapses) / (ndelays * R))
        << "Expected even distribution of delays";
    EXPECT_LE(0.8 * (nwords * nsynapses) / (ndelays * R), delay_count[i])
        << "Expected even distribution of delays";
  }

  for (int32 i = 0; i < w.size(); ++i) {
    w.set_delay(i, i % ndelays);
  }
  for (int32 i = 0; i < w.size(); ++i) {
    EXPECT_EQ(i % ndelays, w.delay(i))
        << "w: Expect delay(" << i << ") " << i % ndelays << ": " << w.delay(i);
  }
}
}  // namespace

int main(int argc, char **argv) {
  CALL_TEST(cognon::CheckWordset);
  CALL_TEST(cognon::CheckWordsetFixed);
  return 0;
}
