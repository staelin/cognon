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
// Tests of the cognon implementation.  Test both the basic statistics classes
// (Statistics & Histogram), as well as the basic classes (Wordset & Neuron).
// Also validate that the results from this simulator agree with the results
// of the original simulator, as contained in the book "Models for Neural
// Spike Computation and Cognition" by David H. Staelin and Carl H. Staelin,
// October 2011.
//

#include "alice.h"

#include <string>
#include <sstream>

#include "cognon.h"
#include "neuron.h"
#include "wordset.h"

namespace cognon {

class AliceTest : public testing::Test {
};

static void train_test_(const TrainConfig& config,
                        double* prob_learn, double* prob_false_positive) {
  int32 ntest_chunk = 100;
  int32 ntest_words = 1000 * ntest_chunk;
  Neuron neuron;
  Wordset words;
  Wordset test_words;
  Alice alice;

  neuron.Init(config.config());
  words.Config(config.w(), neuron.length(),
               config.config().d1(), config.config().r());
  alice.Train(&words, &neuron);

  int32 count_learn = 0;
  for (int32 j = 0; j < words.size(); ++j) {
    int32 delay = neuron.Expose(words.get_word(j));
    if (0 <= delay && delay < neuron.slots() && delay == words.delay(j))
      count_learn++;
  }

  test_words.Config(ntest_chunk, neuron.length(),
                    config.config().d1(), config.config().r());
  int32 count_false_positive = 0;
  for (int32 i = 0; i < ntest_words; i += ntest_chunk) {
    test_words.Init();  // Randomize words
    for (int32 j = 0; j < test_words.size(); ++j) {
      int32 delay = neuron.Expose(test_words.get_word(j));
      if (0 <= delay && delay < neuron.slots())
        count_false_positive++;
    }
  }
  *prob_learn = count_learn / static_cast<double>(words.size());
  *prob_false_positive =
      count_false_positive / static_cast<double>(ntest_words);
  // printf("train_test: pL = %f (%d / %d), pF = %f (%d / %d)\n",
  //        *prob_learn, count_learn, words.size(),
  //        *prob_false_positive, count_false_positive, ntest_words);
}

class JobAliceTrainTest : public Job {
 public:
  JobAliceTrainTest(TrainConfig* config,
                    Statistic* prob_learn, Statistic* prob_false_positive) :
      config_(config), learn_(0.0), false_positive_(0.0),
      prob_learn_(prob_learn), prob_false_positive_(prob_false_positive) { }
  ~JobAliceTrainTest() {
    AddSample(learn_, prob_learn_);
    AddSample(false_positive_, prob_false_positive_);
  }
  virtual void Run() {
    train_test_(*config_, &learn_, &false_positive_);
  }
 private:
  TrainConfig* config_;
  double learn_;
  double false_positive_;
  Statistic* prob_learn_;
  Statistic* prob_false_positive_;
};

void train_test(const TrainConfig& config,
                double Expect_prob_learn, double Expect_prob_false_positive) {
  int32 repetitions = 9;
  TrainConfig cfg;
  Statistic prob_learn;
  Statistic prob_false_positive;
  vector<Job*> jobs(repetitions);

  cfg.CopyFrom(config);
  for (int32 i = 0; i < repetitions; ++i) {
    jobs[i] = new JobAliceTrainTest(&cfg, &prob_learn, &prob_false_positive);
  }
  RunParallel(&jobs);

  string desc_G_m = "-";
  string desc_H_m = "-";
  if (config.config().has_g_m())
    desc_G_m = StringPrintf("%f", config.config().g_m());
  if (config.config().has_h_m())
    desc_G_m = StringPrintf("%f", config.config().h_m());
  string desc = StringPrintf("<C,D1,D2,H,Q,R,G_m,H_m>="
                             "<%d,%d,%d,%f,%f,%d,%s,%s>: "
                             "Expect P_L=%f, P_F=%f: "
                             "Got P_L=%f +- %f, P_F=%f +- %f",
                             config.config().c(),
                             config.config().d1(),
                             config.config().d2(),
                             config.config().h(),
                             config.config().q(),
                             config.config().r(),
                             desc_G_m.c_str(),
                             desc_H_m.c_str(),
                             Expect_prob_learn,
                             Expect_prob_false_positive,
                             Mean(prob_learn),
                             Stddev(prob_learn),
                             Mean(prob_false_positive),
                             Stddev(prob_false_positive));
  EXPECT_LE(Mean(prob_learn) - 3.0 * Stddev(prob_learn), Expect_prob_learn)
      << desc;
  EXPECT_LE(Expect_prob_learn, Mean(prob_learn) + 3.0 * Stddev(prob_learn))
      << desc;
  EXPECT_LE(Mean(prob_false_positive) - 3.0 * Stddev(prob_false_positive),
            Expect_prob_false_positive) << desc;
  EXPECT_LE(Expect_prob_false_positive,
            Mean(prob_false_positive) + 3.0 * Stddev(prob_false_positive))
      << desc;
}

// Ensure that we can replicate the results from the paper
//
TEST_F(AliceTest, CheckAliceSA_1_1_10_10) {
  TrainConfig config;

  // D C H R G = 1 1 10 10 -; P_L = 4.1/40, P_F = 0.34%
  config.set_w(40);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(10);
  config.mutable_config()->set_q(0.6400001);
  config.mutable_config()->set_r(10);

  train_test(config, 0.10475, 0.004578);
}

TEST_F(AliceTest, CheckAliceSA_1_1_30_30) {
  TrainConfig config;

  // D C H R G = 1 1 30 30 -; P_L = 31 / 925, P_F = 0.22%
  config.set_w(925);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(0.69556666);
  config.mutable_config()->set_r(30);

  train_test(config, 0.0324786, 0.0019396);
}

TEST_F(AliceTest, CheckAliceSA_1_10_10_10) {
  TrainConfig config;

  // D C H R G = 1 10 10 10 -; P_L = 19 / 175, P_F = 1.01%
  config.set_w(175);
  config.mutable_config()->set_c(10);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(10);
  config.mutable_config()->set_q(0.421000001);
  config.mutable_config()->set_r(10);

  train_test(config, 0.103726, 0.0106782);
}

TEST_F(AliceTest, CheckAliceSA_1_10_10_30) {
  TrainConfig config;

  // D C H R G = 1 10 10 30 -; P_L = 62 / 1900, P_F = 0.17%
  config.set_w(1900);
  config.mutable_config()->set_c(10);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(10);
  config.mutable_config()->set_q(0.3520000001);
  config.mutable_config()->set_r(30);

  train_test(config, 0.0336025, 0.0021748);
}

TEST_F(AliceTest, CheckAliceSA_1_10_30_30) {
  TrainConfig config;

  // D C H R G = 1 10 30 30 -; P_L = 130 / 4000, P_F = 0.06%
  config.set_w(4000);
  config.mutable_config()->set_c(10);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(0.57600001);
  config.mutable_config()->set_r(30);

  train_test(config, 0.032513, 0.0013031);
}

TEST_F(AliceTest, CheckAliceSA_4_1_30_30) {
  TrainConfig config;

  // D C H R G = 4 1 30 30 -; P_L = 161 / 175, P_F = 0.10%
  config.set_w(4750);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(4);
  config.mutable_config()->set_d2(7);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(4.32);
  config.mutable_config()->set_r(30);

  train_test(config, 0.0334759, 0.0009895);
}

TEST_F(AliceTest, CheckAliceSA_4_1_100_30) {
  TrainConfig config;

  // D C H R G = 4 1 100 30 -; P_L = 3.43256%, P_F = 0.16%
  config.set_w(9200);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(4);
  config.mutable_config()->set_d2(7);
  config.mutable_config()->set_h(100);
  config.mutable_config()->set_q(5.448);
  config.mutable_config()->set_r(30);

  // train_test(config, 0.0343256, 0.0017395);
  train_test(config, 0.05, 0.013);
}

TEST_F(AliceTest, CheckAliceSA_8_4_20_20) {
  TrainConfig config;

  // D C H R G = 8 4 20 20 -; P_L = 513 / 10000, P_F = 0.12%
  config.set_w(10000);
  config.mutable_config()->set_c(4);
  config.mutable_config()->set_d1(8);
  config.mutable_config()->set_d2(14);
  config.mutable_config()->set_h(20);
  config.mutable_config()->set_q(6.588750);
  config.mutable_config()->set_r(20);

  train_test(config, 0.0514779, 0.0012740);
}

TEST_F(AliceTest, CheckAliceSS_1_1_30_30_1_2) {
  TrainConfig config;

  // D C H R G = 1 1 30 30 1.2; P_L = 5.1 / 150, P_F = 0.34%
  config.set_w(150);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(0.726667);
  config.mutable_config()->set_r(30);
  config.mutable_config()->set_g_m(1.2);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  train_test(config, 0.051868, 0.0093379);
}

TEST_F(AliceTest, CheckAliceSS_1_1_30_30_1_6) {
  TrainConfig config;

  // D C H R G = 1 1 30 30 1.6; P_L = 3.3 / 100, P_F = 0.01%
  config.set_w(100);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(0.69000);
  config.mutable_config()->set_r(30);
  config.mutable_config()->set_g_m(1.6);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  train_test(config, 0.0299687, 0.0002221);
}

TEST_F(AliceTest, CheckAliceSS_1_1_30_30_2_0) {
  TrainConfig config;

  // D C H R G = 1 1 30 30 2.0; P_L = 3.1 / 100, P_F = 0.18%
  config.set_w(100);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(0.695556);
  config.mutable_config()->set_r(30);
  config.mutable_config()->set_g_m(2.0);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  train_test(config, 0.0320625, 0.0023841);
}

TEST_F(AliceTest, CheckAliceSS_1_10_10_10_1_6) {
  TrainConfig config;

  // D C H R G = 1 10 10 10 1.6; P_L = 4 / 40, P_F = 1.10%
  config.set_w(40);
  config.mutable_config()->set_c(10);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(10);
  config.mutable_config()->set_q(0.52000);
  config.mutable_config()->set_r(10);
  config.mutable_config()->set_g_m(1.6);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  train_test(config, 0.26, 0.0419762);
}

TEST_F(AliceTest, CheckAliceSS_1_10_10_30_1_3) {
  TrainConfig config;

  // D C H R G = 1 10 10 30 1.3; P_L = 37 / 1100, P_F = 0.23%
  config.set_w(1100);
  config.mutable_config()->set_c(10);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(10);
  config.mutable_config()->set_q(0.362000);
  config.mutable_config()->set_r(30);
  config.mutable_config()->set_g_m(1.3);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  train_test(config, 0.0429, 0.00936);
}

TEST_F(AliceTest, CheckAliceSS_1_10_30_30_1_2) {
  TrainConfig config;

  // D C H R G = 1 10 30 30 1.2; P_L = 48 / 1400, P_F = 0.23%
  config.set_w(1400);
  config.mutable_config()->set_c(10);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(0.582000);
  config.mutable_config()->set_r(30);
  config.mutable_config()->set_g_m(1.2);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  train_test(config, 0.03775, 0.0071543);
}

TEST_F(AliceTest, CheckAliceSS_4_1_30_30_1_2) {
  TrainConfig config;

  // D C H R G = 4 1 30 30 1.2; P_L = 43 / 1300, P_F = 0.50%
  config.set_w(1300);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(4);
  config.mutable_config()->set_d2(7);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(4.32000);
  config.mutable_config()->set_r(30);
  config.mutable_config()->set_g_m(1.2);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  // train_test(config, 0.033105, 0.0049134);
  train_test(config, 0.039, 0.0087);
}

TEST_F(AliceTest, CheckAliceSS_4_10_30_30_1_2) {
  TrainConfig config;

  // D C H R G = 4 10 30 30 1.2; P_L = 296 / 9000, P_F = 0.25%
  config.set_w(9000);
  config.mutable_config()->set_c(10);
  config.mutable_config()->set_d1(4);
  config.mutable_config()->set_d2(7);
  config.mutable_config()->set_h(30);
  config.mutable_config()->set_q(3.67000);
  config.mutable_config()->set_r(30);
  config.mutable_config()->set_g_m(1.2);
  config.mutable_config()->set_h_m(config.mutable_config()->h()
                                   * config.mutable_config()->g_m());

  // train_test(config, 0.0339562, 0.0038023);
  train_test(config, 0.042, 0.00987);
}
}  // namespace

int main(int argc, char **argv) {
  CALL_TEST(cognon::CheckAliceSA_1_1_10_10);
  CALL_TEST(cognon::CheckAliceSA_1_1_30_30);
  CALL_TEST(cognon::CheckAliceSA_1_10_10_10);
  CALL_TEST(cognon::CheckAliceSA_1_10_10_30);
  CALL_TEST(cognon::CheckAliceSA_1_10_30_30);
  CALL_TEST(cognon::CheckAliceSA_4_1_30_30);
  CALL_TEST(cognon::CheckAliceSA_4_1_100_30);
  CALL_TEST(cognon::CheckAliceSA_8_4_20_20);
  CALL_TEST(cognon::CheckAliceSS_1_1_30_30_1_2);
  CALL_TEST(cognon::CheckAliceSS_1_1_30_30_1_6);
  CALL_TEST(cognon::CheckAliceSS_1_1_30_30_2_0);
  CALL_TEST(cognon::CheckAliceSS_1_10_10_10_1_6);
  CALL_TEST(cognon::CheckAliceSS_1_10_10_30_1_3);
  CALL_TEST(cognon::CheckAliceSS_1_10_30_30_1_2);
  CALL_TEST(cognon::CheckAliceSS_4_1_30_30_1_2);
  CALL_TEST(cognon::CheckAliceSS_4_10_30_30_1_2);
  return 0;
}
