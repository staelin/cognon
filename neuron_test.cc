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
// Tests of the cognon implementation.  Validates that the results from this
// simulator agree with the results of the original simulator, as contained
// in the book "Models for Neural Spike Computation and Cognition" by David
// H. Staelin and Carl H. Staelin, October 2011.
//

#include "neuron.h"

#include "cognon.h"
#include "cognon-orig.h"
#include "wordset.h"

namespace cognon {

class NeuronTest : public testing::Test {
};

static void CheckInitialState(const NeuronConfig& config,
                              const Neuron& neuron) {
  for (int32 i = 0; i < neuron.length(); ++i) {
    EXPECT_TRUE(0 <= neuron.delays(i) && neuron.delays(i) < config.d2());
    EXPECT_TRUE(0 <= neuron.containers(i) && neuron.containers(i) < config.c());
    EXPECT_EQ(neuron.frozen(i), false);
    EXPECT_EQ(neuron.strength(i), 1.0);
  }

  // Check that synapses are roughly evenly distributed across delays
  vector<int32> counts;
  counts.resize(config.d2());
  for (int32 i = 0; i < neuron.length(); ++i) {
    counts[neuron.delays(i)]++;
  }
  for (int32 i = 0; i < config.d2(); ++i) {
    EXPECT_LE(counts[i],
              (1.3 * neuron.length()) / static_cast<double>(config.d2()))
        << "Delays should be roughly equally distributed";
    EXPECT_LE((0.7 * neuron.length()) / static_cast<double>(config.d2()),
              counts[i])
        << "Delays should be roughly equally distributed";
  }

  // Check that synapses are roughly evenly distributed across containers
  counts.clear();
  counts.resize(config.c());
  for (int32 i = 0; i < neuron.length(); ++i) {
    counts[neuron.containers(i)]++;
  }
  for (int32 i = 0; i < config.c(); ++i) {
    EXPECT_LE(counts[i],
              (1.45 * neuron.length()) / static_cast<double>(config.c()))
        << "Synapses should be roughly equally distributed across containers";
    EXPECT_LE((0.6 * neuron.length()) / static_cast<double>(config.c()),
              counts[i])
        << "Synapses should be roughly equally distributed across containers";
  }
}

static void CheckFinalState(const NeuronConfig& config, const Neuron& neuron) {
  // After training completes, the neuron should match this profile
  for (int32 i = 0; i < neuron.length(); ++i) {
    if (config.has_g_m()) {
      EXPECT_TRUE(0 <= neuron.delays(i) && neuron.delays(i) < config.d2());
      EXPECT_TRUE(neuron.strength(i) == config.g_m()
                  || neuron.strength(i) == 1.0);
    } else {
      EXPECT_TRUE((neuron.frozen(i)
                   && 0 <= neuron.delays(i)
                   && neuron.delays(i) < config.d2())
                  || (!neuron.frozen(i) && neuron.delays(i) == kDisabled));
      EXPECT_TRUE(neuron.strength(i) == 1.0
                  || neuron.strength(i) == 0.0);
    }
    EXPECT_TRUE(0 <= neuron.containers(i) && neuron.containers(i) < config.c());
  }
}

static void copy_word(int max_delay, int N, const int* values, Word* word) {
  word->clear();
  for (int32 i = 0; i < N; ++i) {
    if (0 <= values[i] && values[i] < max_delay) {
      pair<int32, int32> value(i, values[i]);
      word->push_back(value);
    }
  }
}

static void compare_networks(const cognon_orig::Neuron* neuron_orig,
                             Neuron* neuron) {
  EXPECT_EQ(neuron_orig->l, neuron->length());
  EXPECT_EQ(neuron_orig->C, neuron->C());
  EXPECT_EQ(neuron_orig->D1, neuron->D1());
  EXPECT_EQ(neuron_orig->D2, neuron->D2());
  EXPECT_EQ(neuron_orig->Q, neuron->Q());
  EXPECT_EQ(neuron_orig->R, neuron->R());

  for (int32 i = 0; i < neuron_orig->l; ++i) {
    EXPECT_EQ(neuron_orig->delays[i], neuron->delays(i))
        << "compare_networks: Expect "
        << "neuron_orig->delays[i] == neuron->delays(i)\n";
    EXPECT_EQ(neuron_orig->containers[i], neuron->containers(i))
        << "compare_networks: Expect "
        << "neuron_orig->containers[i] == neuron->containers(i)\n";
  }

  if (!neuron->config().has_g_m() || neuron->config().g_m() < 0) {
    EXPECT_TRUE(!neuron->config().has_h_m()
                || neuron->config().h_m() <= 0)
        << "compare_networks: Expect "
        << "!neuron->config().has_h_m() || neuron->config().h_m() <= 0\n";
    EXPECT_EQ(neuron_orig->H, neuron->H())
        << "compare_networks: Expect "
        << "neuron_orig->H == neuron->H()\n";
    for (int32 i = 0; i < neuron_orig->l; ++i) {
      EXPECT_TRUE((neuron_orig->frozen[i] && neuron->frozen(i))
                  || (!neuron_orig->frozen[i] && !neuron->frozen(i)))
        << "compare_networks: Expect "
        << "(neuron_orig->frozen[i] && neuron->frozen(i)) "
        << "|| (!neuron_orig->frozen[i] && !neuron->frozen(i)\n";
      EXPECT_TRUE(neuron->strength(i) == 1.0
                  || neuron->strength(i) == 0.0)
        << "compare_networks: Expect "
        << "neuron->strength(i) == 1.0 || neuron->strength == 0.0\n";
    }
  } else {
    const cognon_orig::NeuronStrength* ns =
        static_cast<const cognon_orig::NeuronStrength*>(neuron_orig);
    EXPECT_TRUE(neuron->config().has_h_m()
                && 0 < neuron->config().h_m())
        << "compare_network: Expect neuron->config().has_h_m() "
        << "&& 0 < neuron->config().h_m()\n";
    EXPECT_TRUE(neuron_orig->H == neuron->H() || ns->H_m == neuron->H())
        << "compare_networks: Expect neuron_orig->H == neuron->H() "
        << "|| ns->H_m == neuron->H()\n";
    EXPECT_EQ(ns->G_m, neuron->G_m())
        << "compare_networks: Expect ns->G_m " << ns->G_m
        << " == neuron->G_m() " << neuron->G_m() << "\n";
    EXPECT_EQ(ns->H_m, neuron->H_m());
    for (int32 i = 0; i < neuron_orig->l; ++i) {
      EXPECT_EQ(ns->strength[i], neuron->strength(i))
          << "compare_networks: Expect ns->strength[" << i
          << "] " << ns->strength[i] << " == neuron->strength(" << i
          << ") " << neuron->strength(i) << "\n";
    }
  }
}

static void test_replay_single(int32 seed, int32 W,
                               const NeuronConfig& config) {
  Neuron neuron;
  cognon_orig::Neuron* neuron_orig;
  cognon_orig::Wordset words_orig;

  if (!config.has_g_m() || config.g_m() <= 0.0) {
    CHECK(!config.has_h_m() || config.h_m() <= 0.0);
    neuron_orig = new cognon_orig::Neuron();
  } else {
    CHECK(config.has_g_m() && 0 < config.g_m()
          && config.has_h_m() && 0 < config.h_m());
    cognon_orig::NeuronStrength* ns = new cognon_orig::NeuronStrength();
    ns->SetG_m(config.g_m());
    ns->SetH_m(config.h_m());
    neuron_orig = ns;
  }
  scoped_ptr<cognon_orig::Neuron> neuron_orig_ptr(neuron_orig);

  // Initialize the original neuron and traing wordset
  neuron_orig->initialize(config.c(), config.d1(), config.d2(),
                          config.h(), config.q(), config.r());
  neuron.Init(config);
  CheckInitialState(config, neuron);
  EXPECT_EQ(neuron_orig->l, neuron.length())
      << "Neurons should be the same size";

  // Copy random state from original neuron to new neuron
  for (int32 i = 0; i < neuron_orig->l; ++i) {
    neuron.set_delays(i, neuron_orig->delays[i]);
    neuron.set_containers(i, neuron_orig->containers[i]);
  }

  // Train both neurons.  Expect identical results for each word.
  words_orig.seed(seed);
  words_orig.initialize(1, neuron_orig->length(), config.d1(), config.r());
  neuron.StartTraining();
  for (int32 i = 0; i < W; ++i) {
    Word word;
    words_orig.initialize();  // Randomize the word
    copy_word(config.d1(), neuron_orig->length(), words_orig.words[0], &word);
    EXPECT_EQ(neuron_orig->train(words_orig.words[0], 1),
              neuron.Train(word))
        << "Neurons should train to the same delay slot";
  }
  neuron_orig->clean();
  neuron.FinishTraining();
  compare_networks(neuron_orig, &neuron);

  // Test trained neurons on training words.  Expect identical results.
  words_orig.seed(seed);
  words_orig.initialize(1, neuron_orig->length(), config.d1(), config.r());
  for (int32 i = 0; i < W; ++i) {
    Word word;
    words_orig.initialize();  // Randomize the word
    copy_word(config.d1(), neuron_orig->length(), words_orig.words[0], &word);
    EXPECT_EQ(neuron_orig->train(words_orig.words[0]), neuron.Expose(word))
        << "Neurons should recognize trained words in the same delay slot";
  }

  // Test trained neurons on random words.  Expect identical results.
  words_orig.initialize(1, neuron_orig->length(), config.d1(), config.r());
  for (int32 i = 0; i < 10000; ++i) {
    Word word;
    words_orig.initialize();  // Randomize the word
    copy_word(config.d1(), neuron_orig->length(), words_orig.words[0], &word);
    EXPECT_EQ(neuron_orig->train(words_orig.words[0]), neuron.Expose(word))
        << "Neurons should recognize random words in the same delay slot";
  }
}

class JobNeuronTestSingle : public Job {
 public:
  JobNeuronTestSingle(int32 seed, int32 W, NeuronConfig* config) :
      seed_(seed), W_(W), config_(config) { }
  ~JobNeuronTestSingle() {
  }
  virtual void Run() {
    test_replay_single(seed_, W_, *config_);
  }
 private:
  int32 seed_;
  int32 W_;
  NeuronConfig* config_;
};

void test_replay(int32 W, int32 C, int32 D1, int32 D2, double H,
                 double Q, int32 R, double G_m, double H_m) {
  const int32 repetitions = 10;
  scoped_ptr<RandomBase> random(CreateRandom());
  NeuronConfig config;

  config.set_c(C);
  config.set_d1(D1);
  config.set_d2(D2);
  config.set_h(H);
  config.set_q(Q);
  config.set_r(R);
  if (0 < G_m && 0 < H_m) {
    config.set_g_m(G_m);
    config.set_h_m(H_m);
  }

  vector<Job*> jobs(repetitions);
  for (int32 i = 0; i < repetitions; ++i) {
    jobs[i] = new JobNeuronTestSingle(random->Rand32(), W, &config);
  }
  RunParallel(&jobs);
}

TEST_F(NeuronTest, CheckNeuronLength) {
  NeuronConfig config;

  config.set_c(1);
  config.set_d1(1);
  config.set_d2(1);
  config.set_h(10);
  config.set_q(64.001 / (1.0 * 10.0 * 10.0));
  config.set_r(10);
  Neuron neuron;

  neuron.Init(config);
  CheckInitialState(config, neuron);
  EXPECT_EQ(64, neuron.length()) << "Expected 64: " << neuron.length();
}

TEST_F(NeuronTest, CheckSynapseAtrophy) {
  NeuronConfig config;

  config.set_c(1);
  config.set_d1(4);
  config.set_d2(7);
  config.set_h(100);
  config.set_q(5.448);
  config.set_r(30);

  Neuron neuron;
  neuron.Init(config);
  neuron.StartTraining();

  // Initialize the synapse vector to a set of known values
  for (int32 i = 0; i < neuron.length(); ++i) {
    neuron.set_delays(i, 1);
  }

  // Initialize the wordset to a known word
  Word word;
  for (int32 i = 0; i < config.h(); ++i) {
    pair<int32, int32> value(i, 1);
    word.push_back(value);
  }

  // Train the neuron to recognize this word
  EXPECT_EQ(2, neuron.Train(word))
      << "Should train to recognize this word in delay slot 2";

  neuron.FinishTraining();
  EXPECT_EQ(2, neuron.Expose(word))
      << "Neuron should recognize this learned word in slot 2";
  CheckFinalState(config, neuron);

  int32 count = 0;
  for (int32 i = 0; i < neuron.length(); ++i) {
    if (neuron.frozen(i)) count++;
  }
  EXPECT_EQ(count, word.size())
      << "Frozen synapse count should equal the number of inputs";
}

TEST_F(NeuronTest, CheckSynapseStrength) {
  NeuronConfig config;

  config.set_c(10);
  config.set_d1(1);
  config.set_d2(1);
  config.set_h(10);
  config.set_q(0.362000);
  config.set_r(30);
  config.set_g_m(1.3);
  config.set_h_m(config.h() * config.g_m());

  Neuron neuron;
  neuron.Init(config);

  // Initialize the synapse vector to a set of known values
  for (int32 i = 0; i < neuron.length(); ++i) {
    neuron.set_delays(i, 0);
  }

  // Initialize the wordset to a known word
  Word word;
  int32 container;
  // Find a container with enough synapses
  for (container = 0; container < config.c(); ++container) {
    int32 count = 0;
    for (int32 i = 0; i < neuron.length(); ++i) {
      if (neuron.containers(i) == container) count++;
    }
    if (config.h() <= count) break;
  }
  EXPECT_TRUE(container < config.c());
  // Setup the word to use that container
  int32 j = 0;
  for (int32 i = 0; i < config.h(); ++i) {
    while (j < neuron.length() && neuron.containers(j) != container) {
      j++;
    }
    pair<int32, int32> value(j++, 0);
    word.push_back(value);
  }

  // Train the neuron to recognize this word
  neuron.StartTraining();
  EXPECT_EQ(0, neuron.Train(word))
      << "Should train to recognize this word in delay slot 0";

  neuron.FinishTraining();
  EXPECT_EQ(config.h_m(), neuron.H());
  EXPECT_EQ(0, neuron.Expose(word))
      << "Neuron should recognize this learned word in slot 0";
  CheckFinalState(config, neuron);

  int32 count = 0;
  for (int32 i = 0; i < neuron.length(); ++i) {
    if (neuron.strength(i) == config.g_m()) count++;
  }
  EXPECT_EQ(count, word.size())
      << "Strengthened synapse count should equal the number of inputs";
}

#define NAME_TEST_REPLAY_SA(W, C, D1, D2, H, Q, R)                      \
  TestReplaySA_##W##_##C##_##D1##_##D2##_##H##_##R

#define TEST_REPLAY_SA(W, C, D1, D2, H, Q, R)                           \
TEST_F(NeuronTest, NAME_TEST_REPLAY_SA(W, C, D1, D2, H, Q, R)) {        \
    test_replay(W, C, D1, D2, H, Q, R, -1.0, -1.0);                     \
}

#define NAME_TEST_REPLAY_SS(W, C, D1, D2, H, Q, R, G_m, H_m)            \
  TestReplaySS_##W##_##C##_##D1##_##D2##_##H##_##R

#define TEST_REPLAY_SS(W, C, D1, D2, H, Q, R, G_m, H_m)                 \
TEST_F(NeuronTest, NAME_TEST_REPLAY_SS(W, C, D1, D2, H, Q, R, G_m, H_m))\
{                                                                       \
    test_replay(W, C, D1, D2, H, Q, R, G_m, H_m);                       \
}

TEST_REPLAY_SA(40, 1, 1, 1, 10, 0.64, 10);
TEST_REPLAY_SA(925, 1, 1, 1, 30, 0.69556666, 10);
TEST_REPLAY_SA(175, 10, 1, 1, 10, 0.69556666, 30);
TEST_REPLAY_SA(1900, 10, 1, 1, 10, 0.352, 30);
TEST_REPLAY_SA(4000, 10, 1, 1, 30, 0.576, 30);
TEST_REPLAY_SA(4750, 1, 4, 7, 30, 4.32, 30);
TEST_REPLAY_SA(9200, 1, 4, 7, 100, 5.448, 30);
TEST_REPLAY_SA(10000, 4, 8, 14, 20, 6.58875, 20);

TEST_REPLAY_SS(150, 1, 1, 1, 30, 0.726667, 30, 1.2, 36.0);
TEST_REPLAY_SS(100, 1, 1, 1, 30, 0.69, 30, 1.6, 48.0);
TEST_REPLAY_SS(101, 1, 1, 1, 30, 0.695556, 30, 2.0, 60.0);
TEST_REPLAY_SS(40, 10, 1, 1, 10, 0.52, 10, 1.6, 16.0);
TEST_REPLAY_SS(1100, 10, 1, 1, 10, 0.362, 30, 1.3, 39.0);
TEST_REPLAY_SS(1400, 10, 1, 1, 30, 0.58, 30, 1.2, 36.0);
TEST_REPLAY_SS(1300, 1, 4, 7, 30, 4.32, 30, 1.2, 36.0);
TEST_REPLAY_SS(900, 10, 4, 7, 30, 3.67, 30, 1.2, 36.0);

static void test_WordsetFixed(int32 W, int32 active, NeuronConfig* config,
                              double* prob_learn, double* prob_false) {
  Neuron neuron;
  neuron.Init(*config);
  CheckInitialState(*config, neuron);

  *prob_learn = 0.0;
  *prob_false = 0.0;

  Wordset words;
  words.ConfigFixed(W, neuron.length(), config->d1(), active);
  neuron.StartTraining();
  for (int32 i = 0; i < words.size(); ++i) {
    const Word& word = words.get_word(i);
    words.set_delay(i, neuron.Train(word));
  }
  neuron.FinishTraining();

  // Test trained neurons on training words.  Expect identical results.
  for (int32 i = 0; i < words.size(); ++i) {
    const Word& word = words.get_word(i);
    CHECK(words.delay(i) == neuron.Expose(word));
    EXPECT_EQ(words.delay(i), neuron.Expose(word))
        << "Neurons should recognize trained words in the same delay slot";
    if (words.delay(i) != kDisabled) (*prob_learn)++;
  }
  *prob_learn /= static_cast<double>(W);

  // Test trained neurons on random words.  Expect identical results.
  for (int32 i = 0; i < 10000; ++i) {
    int j = i % words.size();
    if (j == 0) words.Init();
    const Word& word = words.get_word(j);
    if (neuron.Expose(word) != kDisabled) (*prob_false)++;
  }
  *prob_false /= 10000.0;
}

TEST_F(NeuronTest, TestWordsetFixed) {
  double prob_learn;
  double prob_false;
  NeuronConfig config;

  config.set_c(1);
  config.set_d1(1);
  config.set_d2(1);
  config.set_h(10);
  config.set_q(1.0);
  config.set_r(10);
  config.set_g_m(1.1);
  config.set_h_m(11.0);

  test_WordsetFixed(1, 10, &config, &prob_learn, &prob_false);
  EXPECT_TRUE(0.9 < prob_learn)
      << "Expect this configuration to learn; prob_learn = "
      << prob_learn << "\n";
  EXPECT_TRUE(prob_false < 0.1)
      << "Expect this configuration to have few false positives: prob_false = "
      << prob_false << "\n";
}

}  // namespace cognon


int main(int argc, char **argv) {
  CALL_TEST(cognon::CheckNeuronLength);
  CALL_TEST(cognon::CheckSynapseAtrophy);
  CALL_TEST(cognon::CheckSynapseStrength);
  CALL_TEST(cognon::TestWordsetFixed);

  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(40,1,1,1,10,0.64,10));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(925,1,1,1,30,0.69556666,10));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(175,10,1,1,10,0.69556666,30));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(1900,10,1,1,10,0.352,30));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(4000,10,1,1,30,0.576,30));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(4750,1,4,7,30,4.32,30));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(9200,1,4,7,100,5.448,30));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SA(10000,4,8,14,20,6.58875,20));

  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(150,1,1,1,30,0.726667,30,1.2,36.0));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(100,1,1,1,30,0.69,30,1.6,48.0));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(101,1,1,1,30,0.695556,30,2.0,60.0));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(40,10,1,1,10,0.52,10,1.6,16.0));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(1100,10,1,1,10,0.362,30,1.3,39.0));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(1400,10,1,1,30,0.58,30,1.2,36.0));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(1300,1,4,7,30,4.32,30,1.2,36.0));
  CALL_TEST(cognon::NAME_TEST_REPLAY_SS(900,10,4,7,30,3.67,30,1.2,36.0));
  return 0;
}
