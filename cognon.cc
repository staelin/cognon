// Copyright 2009-2011 Carl Staelin. All Rights Reserved.
// Copyright 2011 Google Inc. All Rights Reserved.
//
// Author: carl.staelin@gmail.com (Carl Staelin)
//
// Cognon neuron simulator implementation.
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

#include "cognon.h"

#include <math.h>

#include "alice.h"
#include "bob.h"
#include "neuron.h"
#include "wordset.h"


namespace cognon {

void AddSample(double v, Statistic* stat) {
  CHECK((!stat->has_count() && stat->values_size() == 0) ||
        (stat->has_count() &&
         (stat->values_size() == 0 || stat->count() == stat->values_size())));
  CHECK(!stat->has_count()
        || stat->count() == 0
        || (0 < stat->count() && stat->has_sum() && stat->has_ssum()));

  if (!stat->has_count()
      || stat->count() == 0
      || (stat->count() == stat->values_size()))
    stat->add_values(v);

  stat->set_count(stat->count() + 1);
  stat->set_sum(stat->sum() + v);
  stat->set_ssum(stat->ssum() + v * v);
}

double Mean(const Statistic& stat) {
  CHECK((!stat.has_count() && stat.values_size() == 0) ||
        (stat.has_count() &&
         (stat.values_size() == 0 || stat.count() == stat.values_size())));
  CHECK(!stat.has_count()
        || stat.count() == 0
        || (0 < stat.count() && stat.has_sum() && stat.has_ssum()));

  if (stat.has_count() && stat.has_sum() && 0 < stat.count())
    return stat.sum() / static_cast<double>(stat.count());
  return -1.0;
}

double Stddev(const Statistic& stat) {
  CHECK((!stat.has_count() && stat.values_size() == 0) ||
        (stat.has_count() &&
         (stat.values_size() == 0 || stat.count() == stat.values_size())));
  CHECK(!stat.has_count()
        || stat.count() == 0
        || (0 < stat.count() && stat.has_sum() && stat.has_ssum()));

  if (stat.has_count() && 1 < stat.count()
      && stat.has_sum() && stat.has_ssum()) {
    double v = stat.count() * stat.ssum() - stat.sum() * stat.sum();
    v /= static_cast<double>(stat.count() * (stat.count() - 1));
    return (0.0 <= v ? sqrt(v) : 0.0);
  }
  return -1.0;
}

Statistic operator+(const Statistic& a, const Statistic& b) {
  Statistic result(a);
  result += b;
  return result;
}

Statistic& operator+=(Statistic& a, const Statistic& b) {
  CHECK((!a.has_count() && a.values_size() == 0) ||
        (a.has_count() &&
         (a.values_size() == 0 || a.count() == a.values_size())));
  CHECK((!b.has_count() && b.values_size() == 0) ||
        (b.has_count() &&
         (b.values_size() == 0 || b.count() == b.values_size())));
  CHECK(!a.has_count()
        || a.count() == 0
        || (0 < a.count() && a.has_sum() && a.has_ssum()));
  CHECK(!b.has_count()
        || b.count() == 0
        || (0 < b.count() && b.has_sum() && b.has_ssum()));

  if (a.has_count() && 0 < a.count()
      && b.has_count() && 0 < b.count()
      && (a.count() != a.values_size() || b.count() != b.values_size())) {
    // Only one of a and b has values(), so result may not have values()
    a.clear_values();
  } else {
    for (int i = 0; i < b.values_size(); ++i) {
      a.add_values(b.values(i));
    }
  }

#define PB_OPERATOR_PLUS(param) \
  if (a.has_##param() && b.has_##param()) { \
    a.set_##param(a.param() + b.param()); \
  } else if (b.has_##param()) { \
    a.set_##param(b.param()); \
  }
  PB_OPERATOR_PLUS(count);
  PB_OPERATOR_PLUS(sum);
  PB_OPERATOR_PLUS(ssum);
#undef PB_OPERATOR_PLUS
  return a;
}

void AddSample(int32 bucket, double v, Histogram* hist) {
  // Add empty buckets, if necessary
  for (int32 i = hist->values_size(); i <= bucket; ++i) {
    hist->add_values();
  }
  AddSample(v, hist->mutable_values(bucket));
}

void SetHistogram(const vector<int32>& data, Histogram* result) {
  for (int32 i = data.size() - 1; 0 <= i; --i) {
    AddSample(i, static_cast<double>(data[i]), result);
  }
}

Histogram operator+(const Histogram& a, const Histogram& b) {
  Histogram result(a);
  result += b;
  return result;
}

Histogram& operator+=(Histogram& a, const Histogram& b) {
  int32 i;
  int32 a_count = (a.values_size() > 0 ? a.values(0).count() : 0);
  int32 b_count = (b.values_size() > 0 ? b.values(0).count() : 0);

  for (i = 0; i < b.values_size(); ++i) {
    if (i < a.values_size()) {
      *a.mutable_values(i) += b.values(i);
    } else {
      Statistic* s = a.add_values();
      for (int32 j = 0; j < a_count; ++j) {
        AddSample(0.0, s);
      }
      *a.mutable_values(i) += b.values(i);
    }
  }
  for (; i < a.values_size(); ++i) {
    for (int32 j = 0; j < b_count; ++j) {
      AddSample(0.0, a.mutable_values(i));
    }
  }
  return a;
}

NeuronStatistics operator+(const NeuronStatistics& a,
                           const NeuronStatistics& b) {
  NeuronStatistics result(a);
  result += b;
  return result;
}

NeuronStatistics& operator+=(NeuronStatistics& a,
                            const NeuronStatistics& b) {
#define PB_OPERATOR_PLUS(param) \
  if (a.has_##param() && b.has_##param()) { \
    *a.mutable_##param() += b.param();  \
  } else if (b.has_##param()) { \
    a.mutable_##param()->CopyFrom(b.param());      \
  }
  PB_OPERATOR_PLUS(false_false);
  PB_OPERATOR_PLUS(false_true);
  PB_OPERATOR_PLUS(false_count);
  PB_OPERATOR_PLUS(true_false);
  PB_OPERATOR_PLUS(true_true);
  PB_OPERATOR_PLUS(true_count);
  PB_OPERATOR_PLUS(q_after);
  PB_OPERATOR_PLUS(synapses_per_neuron);
  PB_OPERATOR_PLUS(bits_per_neuron);
  PB_OPERATOR_PLUS(bits_per_neuron_per_refractory_period);
  PB_OPERATOR_PLUS(mutual_information);
  PB_OPERATOR_PLUS(d_effective);
  PB_OPERATOR_PLUS(delay_histogram);
  PB_OPERATOR_PLUS(input_delay_histogram);
  PB_OPERATOR_PLUS(input_max_sum_delay_histogram);
  PB_OPERATOR_PLUS(h_histogram);
  PB_OPERATOR_PLUS(word_delay_histogram);
  PB_OPERATOR_PLUS(synapse_before_delay_histogram);
  PB_OPERATOR_PLUS(synapse_after_delay_histogram);
#undef PB_OPERATOR_PLUS

  return a;
}

void StatisticStripValues(Statistic* stat) {
  stat->clear_values();
}

void HistogramStripValues(Histogram* hist) {
  for (int32 i = 0; i < hist->values_size(); ++i) {
    StatisticStripValues(hist->mutable_values(i));
  }
}

void NeuronStatisticsStripValues(NeuronStatistics* stats) {
  StatisticStripValues(stats->mutable_false_false());
  StatisticStripValues(stats->mutable_false_true());
  StatisticStripValues(stats->mutable_false_count());
  StatisticStripValues(stats->mutable_true_false());
  StatisticStripValues(stats->mutable_true_true());
  StatisticStripValues(stats->mutable_true_count());
  StatisticStripValues(stats->mutable_q_after());
  StatisticStripValues(stats->mutable_synapses_per_neuron());
  StatisticStripValues(stats->mutable_bits_per_neuron());
  StatisticStripValues(stats->mutable_bits_per_neuron_per_refractory_period());
  StatisticStripValues(stats->mutable_mutual_information());
  StatisticStripValues(stats->mutable_d_effective());
  HistogramStripValues(stats->mutable_delay_histogram());
  HistogramStripValues(stats->mutable_input_delay_histogram());
  HistogramStripValues(stats->mutable_input_max_sum_delay_histogram());
  HistogramStripValues(stats->mutable_h_histogram());
  HistogramStripValues(stats->mutable_word_delay_histogram());
  HistogramStripValues(stats->mutable_synapse_before_delay_histogram());
  HistogramStripValues(stats->mutable_synapse_after_delay_histogram());
}

// Return "true" if a is "less" than b.
bool operator<(const TrainConfig&a, const TrainConfig& b) {
  const NeuronConfig& neuron_a = a.config();
  const NeuronConfig& neuron_b = b.config();

#define VALUE_COMPARE(a, b, v)                       \
  if (!a.has_##v() && b.has_##v()) return true;      \
  if (a.has_##v() && !b.has_##v()) return false;     \
  if (a.has_##v() && b.has_##v()) {                  \
    if (a.v() < b.v()) return true;                  \
    if (a.v() > b.v()) return false;                 \
  }

  VALUE_COMPARE(neuron_a, neuron_b, c);
  VALUE_COMPARE(neuron_a, neuron_b, d1);
  VALUE_COMPARE(neuron_a, neuron_b, d2);
  VALUE_COMPARE(neuron_a, neuron_b, h);
  VALUE_COMPARE(neuron_a, neuron_b, q);
  VALUE_COMPARE(neuron_a, neuron_b, r);
  VALUE_COMPARE(neuron_a, neuron_b, g_m);
  VALUE_COMPARE(neuron_a, neuron_b, h_m);

  VALUE_COMPARE(a, b, w);
  VALUE_COMPARE(a, b, num_active);
  VALUE_COMPARE(a, b, num_test_words);
#undef VALUE_COMPARE

  return false;
}

static double HistogramEntropy(const vector<int32>& hist) {
  // First calculate the entropy of the delay histogram
  double sum = 0.0;
  for (int32 i = 0; i < hist.size(); ++i) {
    sum += hist[i];
  }
  if (sum == 0.0) return 0.0;

  double entropy = 0.0;
  for (int32 i = 0; i < hist.size(); ++i) {
    if (0 < hist[i]) {
      double prob = hist[i] / sum;
      entropy -= prob * log(prob) / log(2.0);
    }
  }
  return pow(2.0, entropy);
}

void RunExperiment(const TrainConfig& config, NeuronStatistics* result) {
  Wordset words;
  Neuron neuron;

  // Must have either both or neither of g_m() and h_m()
  CHECK((config.config().has_g_m() && config.config().has_h_m())
        || (!config.config().has_g_m() && !config.config().has_h_m()));

  result->Clear();
  result->mutable_config()->CopyFrom(config);

  neuron.Init(config.config());
  if (config.has_num_active()) {
    words.ConfigFixed(config.w(), neuron.length(),
                      config.config().d1(), config.num_active());
  } else {
    words.Config(config.w(), neuron.length(),
                 config.config().d1(), config.config().r());
  }

  vector<int32> synapse_before_delay_histogram;
  neuron.GetSynapseDelayHistogram(&synapse_before_delay_histogram);

  Alice alice;
  vector<int32> delay_histogram;
  vector<int32> input_delay_histogram;
  vector<int32> input_max_sum_delay_histogram;
  vector<int32> H_histogram;
  alice.TrainHistogram(&words, &neuron,
                       &delay_histogram, &input_delay_histogram,
                       &input_max_sum_delay_histogram, &H_histogram);

  vector<int32> synapse_after_delay_histogram;
  neuron.GetSynapseDelayHistogram(&synapse_after_delay_histogram);

  // Collect various training-related statistics
  vector<int32> word_delay_histogram;
  SetHistogram(delay_histogram,
               result->mutable_delay_histogram());
  SetHistogram(input_delay_histogram,
               result->mutable_input_delay_histogram());
  SetHistogram(input_max_sum_delay_histogram,
               result->mutable_input_max_sum_delay_histogram());
  SetHistogram(H_histogram,
               result->mutable_h_histogram());
  SetHistogram(word_delay_histogram,
               result->mutable_word_delay_histogram());
  SetHistogram(synapse_before_delay_histogram,
               result->mutable_synapse_before_delay_histogram());
  SetHistogram(synapse_after_delay_histogram,
               result->mutable_synapse_after_delay_histogram());
  // Only add d_effective() if the neuron learned any words
  for (vector<int32>::const_iterator it = delay_histogram.begin();
       it != delay_histogram.end(); ++it) {
    if (0 < *it) {
      AddSample(HistogramEntropy(delay_histogram),
                result->mutable_d_effective());
      break;
    }
  }

  // Now start testing
  Bob bob;
  bob.Test((config.has_num_test_words() ? config.num_test_words() : 100000),
           words, neuron,
           result);

  // Collect statistics
  AddSample(neuron.Q_after(), result->mutable_q_after());
  AddSample(neuron.length(), result->mutable_synapses_per_neuron());
}

class JobRunConfiguration : public Job {
 public:
  JobRunConfiguration(TrainConfig* config, NeuronStatistics* result)
      : config_(config), result_(result), temp_() { }
  ~JobRunConfiguration() {
      (*result_) += temp_;
  }
  virtual void Run() {
    RunExperiment(*config_, &temp_);
  }
 private:
  TrainConfig* config_;
  NeuronStatistics* result_;
  NeuronStatistics temp_;
};

void RunConfiguration(int32 repetitions,
                      const TrainConfig& config, NeuronStatistics* result) {
  int32 N = repetitions;

  result->Clear();
  result->mutable_config()->CopyFrom(config);

  // Ensure that we try to learn at least 10,000 words in aggregate
  if (N * result->config().w() < 10000) {
    N = 10000 / result->config().w();
    if (10000 % result->config().w()) N++;
  }
  CHECK(repetitions <= N);
  CHECK_LT(0, N);

  if (config.has_num_test_words()) {
    result->mutable_config()->set_num_test_words(config.num_test_words());
  } else {
    // By default, ensure that we test on at least 1,000,000 random words
    const int32 kNumTestWords = 1000000;
    int32 num_test_words = kNumTestWords / N;
    if (kNumTestWords % N) num_test_words++;
    if (num_test_words < 1000) num_test_words = 1000;
    result->mutable_config()->set_num_test_words(num_test_words);
  }

  // printf("N = %d, num_test_words = %d\n",
  //        N, result->config().num_test_words());
  // fflush(stdout);
  vector<Job*> jobs(N);
  for (int32 i = 0; i < N; ++i) {
    jobs[i] = new JobRunConfiguration(result->mutable_config(), result);
  }
  RunParallel(&jobs);
}
}  // namespace cognon
