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
#ifndef COGNON_COMPAT_H_
#define COGNON_COMPAT_H_

#include <iostream>
#include <vector>

#include <assert.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <stdint.h>

#include "mtrand.h"

using namespace std;

typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

#define CHECK(condition) assert(condition)
#define CHECK_LT(a, b) assert((a) < (b))
#define CHECK_NOTNULL(condition) assert((condition) != NULL)

// For testing
#define CALL_TEST(name) {                       \
  string test_name(#name);                      \
  test_fail = false;                            \
  name();                                       \
  if (test_fail) {                              \
    cerr << "\nFAIL: " << test_name << "\n";      \
  } else {                                      \
    cerr << "PASS: " << test_name << "\n";      \
  }                                             \
}

#define EXPECT_TRUE(condition)                          \
  /* assert(condition); */                              \
  ((condition) ? cnull : (test_fail = true, cerr))
#define EXPECT_FALSE(condition)                         \
  /* assert(!(condition)); */                           \
  (!(condition) ? cnull : (test_fail = true, cerr))
#define EXPECT_EQ(a,b)                                  \
  /* assert((a) == (b)); */                             \
  (((a) == (b)) ? cnull : (test_fail = true, cerr))
#define EXPECT_LE(a,b)                                  \
  /* assert((a) <= (b)); */                             \
  (((a) <= (b)) ? cnull : (test_fail = true, cerr))
extern bool test_fail;

template<class T> class scoped_ptr {
public:
  scoped_ptr(T* ptr = NULL) : ptr_(ptr) { }
  ~scoped_ptr() { delete ptr_; }
  void reset(T* ptr) { delete ptr_; ptr_ = ptr; }
  T* operator->() { return ptr_; }
  T* get() { return ptr_; }
private:
  T* ptr_;
};

class RandomBase {
 public:
  RandomBase();

  uint32 Rand32();
  uint64 Rand64();
 private:
  int32 threadId;
  static MTRand** random;
};

class nullstream : public std::ostream {
 public:
  struct nullbuf : std::streambuf {
    int overflow(int c) { return traits_type::not_eof(c); }
  } m_sbuf;
  nullstream() : std::ios(&m_sbuf), std::ostream(&m_sbuf) {}
};
extern nullstream cnull;

string StringPrintf(const char* format, ...);
string vStringPrintf(const char* format, va_list args);

namespace testing {
#define TEST_F(testManager, funcName) static void funcName()
class Test {
 public:
  Test() { }
  ~Test() { }
};
}  // namespace

namespace cognon {

#define VALUE_PARAMETER_COMMON(type,name)                               \
 protected:                                                             \
  bool has_##name##_;                                                   \
  type name##_;                                                         \
 public:                                                                \
 bool has_##name() const { return has_##name##_; }                      \
 const type& name() const { return name##_; }                           \
 void set_##name(type v) { name##_ = v; has_##name##_ = true; }         \
 type* mutable_##name() { has_##name##_ = true; return &name##_; }

#define VALUE_PARAMETER(type,name)                                      \
  VALUE_PARAMETER_COMMON(type,name);                                    \
  void clear_##name() { has_##name##_ = false; name##_ = 0; }

#define VALUE_PARAMETER_CLASS(type,name)                                \
  VALUE_PARAMETER_COMMON(type,name);                                    \
  void clear_##name() { has_##name##_ = false; name##_.clear(); }

#define VALUE_COPY(type,name)                                           \
  has_##name##_ = other.has_##name##_;                                  \
  name##_ = other.name##_;

#define VALUE_COPY_CLASS(type,name)                                     \
  has_##name##_ = other.has_##name##_;                                  \
  if (other.has_##name##_) name##_.CopyFrom(other.name##_);

#define VALUE_SERIALIZE(type,name)                                      \
  /* ar & has_##name##_ */;                                             \
  /* ar & name##_ */;

#define VALUE_COMPARE_LESS_THAN(type,name)                              \
  if (!has_##name() && other.has_##name()) return true;                 \
  if (has_##name() && !other.has_##name()) return false;                \
  if (has_##name() && other.has_##name()) {                             \
    if (name() < other.name()) return true;                             \
    if (other.name() < name()) return false;                            \
  }

#define VECTOR_PARAMETER(type,name)                                     \
  protected:                                                            \
  vector<type> name##_;                                            \
 public:                                                                \
 size_t name##_size() const { return name##_.size(); }                  \
 type* add_##name(const type& v) {                                      \
   name##_.push_back(v);                                                \
   return &name##_.back();                                              \
 }                                                                      \
 type* add_##name() {                                                   \
   type v;                                                              \
   name##_.push_back(v);                                                \
   return &name##_.back();                                              \
 }                                                                      \
 void clear_##name() { name##_.clear(); }                               \
 const type& name(int i) const { return name##_[i]; }                   \
 type* mutable_##name(int32 i) { return &name##_[i]; }

#define VECTOR_COPY(type,name)                                          \
  name##_.clear();                                                      \
  name##_.resize(other.name##_.size());                                 \
  for (int32 i = 0; i < other.name##_.size(); ++i) {                    \
    name##_[i] = other.name##_[i];                                      \
  }

#define VECTOR_COPY_CLASS(type,name)                                    \
  name##_.clear();                                                      \
  name##_.resize(other.name##_.size());                                 \
  for (int32 i = 0; i < other.name##_.size(); ++i) {                    \
    name##_[i].CopyFrom(other.name##_[i]);                              \
  }

#define VECTOR_SERIALIZE(type,name)                                     \
  ar & name##_;

#define VECTOR_COMPARE_LESS_THAN(type,name)                             \
  for (int32 i = 0; i < min(name##_size(), other.name##_size()); ++i) { \
    if (name(i) < other.name(i)) return true;                           \
    if (other.name(i) < name(i)) return false;                          \
  }                                                                     \
  if (name##_size() < other.name##_size()) return true;                 \
  if (other.name##_size() < name##_size()) return false;

// NeuronConfig contains the neuron configuration information.
// Currently, all neurons need C, D1, D2, H, Q, and R.
// Only "synapse strength", SS, neurons need G_m and H_m.
//
class NeuronConfig {
  // Number of containers (dendrites with independent summation).
  VALUE_PARAMETER(int32,c);

  // Number of delays in the input words (when the signal arrives
  // at the synapse).
  //
  VALUE_PARAMETER(int32,d1);

  // Number of delays in the synapses (additional delay for the signal
  // to traverse the synapse and arrive at the neuron summation point).
  //
  VALUE_PARAMETER(int32,d2);

  // Summation threshold.
  VALUE_PARAMETER(double,h);

  // Oversampling synapse rate.
  VALUE_PARAMETER(double,q);

  // Recovery period; a synapse will fire at most once every R cycles.
  VALUE_PARAMETER(int32,r);

  // During "synapse strength" training, set synapse strength to
  // this value for synapses that participating in firing during training.
  //
  VALUE_PARAMETER(double,g_m);

  // Threshold used during recognition for "synapse strength"
  // variant of the model.
  //
  VALUE_PARAMETER(double,h_m);

 public:
  NeuronConfig() { clear(); }

  void Clear() { clear(); }

  void clear() {
    clear_c();
    clear_d1();
    clear_d2();
    clear_h();
    clear_q();
    clear_r();
    clear_g_m();
    clear_h_m();
  }

  void CopyFrom(const NeuronConfig& other) {
    clear();
    VALUE_COPY(int32,c);
    VALUE_COPY(int32,d1);
    VALUE_COPY(int32,d2);
    VALUE_COPY(double,h);
    VALUE_COPY(double,q);
    VALUE_COPY(int32,r);
    VALUE_COPY(double,g_m);
    VALUE_COPY(double,h_m);
  }

  bool operator<(const NeuronConfig& other) const {
    VALUE_COMPARE_LESS_THAN(int32,c);
    VALUE_COMPARE_LESS_THAN(int32,d1);
    VALUE_COMPARE_LESS_THAN(int32,d2);
    VALUE_COMPARE_LESS_THAN(double,h);
    VALUE_COMPARE_LESS_THAN(double,q);
    VALUE_COMPARE_LESS_THAN(int32,r);
    VALUE_COMPARE_LESS_THAN(double,g_m);
    VALUE_COMPARE_LESS_THAN(double,h_m);
    return false;
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    VALUE_SERIALIZE(int32,c);
    VALUE_SERIALIZE(int32,d1);
    VALUE_SERIALIZE(int32,d2);
    VALUE_SERIALIZE(double,h);
    VALUE_SERIALIZE(double,q);
    VALUE_SERIALIZE(int32,r);
    VALUE_SERIALIZE(double,g_m);
    VALUE_SERIALIZE(double,h_m);
  }
};

  // TrainConfig contains configuration for the neuron to be trained, as
  // well as the information about how it is to be trained.  Currently
  // this is simply the number of words to try and teach the neuron.
  //
class TrainConfig {
  // configuration of the neuron to be trained.
  VALUE_PARAMETER_CLASS(NeuronConfig,config);

  // Number of words to try to learn.
  VALUE_PARAMETER(int32,w);

  // Number of active inputs per word.
  VALUE_PARAMETER(int32,num_active);

  // Number of random words to test neuron with.
  VALUE_PARAMETER(int32,num_test_words);

 public:
  TrainConfig() { clear(); }

  void Clear() { clear(); }

  void clear() {
    clear_config();
    clear_w();
    clear_num_active();
    clear_num_test_words();
  }

  void CopyFrom(const TrainConfig& other) {
    clear();
    VALUE_COPY_CLASS(NeuronConfig,config);
    VALUE_COPY(int32,w);
    VALUE_COPY(int32,num_active);
    VALUE_COPY(int32,num_test_words);
  }

  bool operator<(const TrainConfig& other) const {
    VALUE_COMPARE_LESS_THAN(NeuronConfig,config);
    VALUE_COMPARE_LESS_THAN(int32,w);
    VALUE_COMPARE_LESS_THAN(int32,num_active);
    VALUE_COMPARE_LESS_THAN(int32,num_test_words);
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    VALUE_SERIALIZE(NeuronConfig,config);
    VALUE_SERIALIZE(int32,w);
    VALUE_SERIALIZE(int32,num_active);
    VALUE_SERIALIZE(int32,num_test_words);
  }
};

// Statistic is a basic statistical class used to keep track of
// experimental results.  In cognon.h there are functions to report
// the mean() and stddev() of statistical results, as well as a
// function to add a new experimental value to statistics (add_value()).
//
class Statistic {
  // Number of values included in the statistic.
  VALUE_PARAMETER(int32,count);

  // Summation of the values.
  VALUE_PARAMETER(double,sum);

  // Summation of squared values.  This is used to compute the
  // standard deviation efficiently.
  //
  VALUE_PARAMETER(double,ssum);

  // Array of the actual values. (Optional)
  VECTOR_PARAMETER(double,values);

 public:
  Statistic() { clear(); }

  void clear() {
    clear_count();
    clear_sum();
    clear_ssum();
    clear_values();
  }

  void CopyFrom(const Statistic& other) {
    clear();
    VALUE_COPY(int32,count);
    VALUE_COPY(double,sum);
    VALUE_COPY(double,ssum);
    VECTOR_COPY(double,values);
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    VALUE_SERIALIZE(int32,count);
    VALUE_SERIALIZE(double,sum);
    VALUE_SERIALIZE(double,ssum);
    VECTOR_SERIALIZE(double,values);
  }
};

// Histogram holds a histogram.  Note that it stores the statistics
// for the values in each bin when multiple Histogram results are
// combined.  So mean(histogram[i]) would report the mean value
// for all the histogram values in bucket i.
//
class Histogram {
  // Array of the statistics for each bucket.
  VECTOR_PARAMETER(Statistic,values);

 public:
  Histogram() { clear(); }

  void Clear() { clear(); }

  void clear() {
    clear_values();
  }

  void CopyFrom(const Histogram& other) {
    clear();
    VECTOR_COPY_CLASS(Statistic,values);
  }
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    VECTOR_SERIALIZE(Statistic,values);
  }
};

// NeuronStatistics is used to hold the results of running simulations.
// Note that a single run reports a set of NeuronStatistics, but
// multiple results can be combined to accumulate statistics for
// each value across runs (for a given configuration).
//
class NeuronStatistics {
  // Configuration used for training the neuron(s).
  VALUE_PARAMETER_CLASS(TrainConfig,config);

  // Statistics of the probability that the neuron will not fire,
  // given that the input word was not learned.
  //
  VALUE_PARAMETER_CLASS(Statistic,false_false);

  // Statistics of the probability that the neuron will fire,
  // given that the input word was not learned.
  //
  VALUE_PARAMETER_CLASS(Statistic,false_true);

  // Statistics of the count of number of test words used per
  // neuron to check false__true and false_false.
  //
  VALUE_PARAMETER_CLASS(Statistic,false_count);

  // Statistics of the probability that the neuron will not fire,
  // given that the word was supposed to have been learned by the neuron.
  //
  VALUE_PARAMETER_CLASS(Statistic,true_false);

  // Statistics of the probability that the neuron will fire,
  // given that the word was supposed to have been learned by
  // the neuron.
  //
  VALUE_PARAMETER_CLASS(Statistic,true_true);

  // Statistics of the count of number of training words used
  // per neuron to check true_true and true_false.
  //
  VALUE_PARAMETER_CLASS(Statistic,true_count);

  // Synapse oversampling rate, after synapse atrophy at end of learning.
  VALUE_PARAMETER_CLASS(Statistic,q_after);

  // Synapses per neuron.
  VALUE_PARAMETER_CLASS(Statistic,synapses_per_neuron);

  // Measure of learning information rate, in bits per neuron.
  VALUE_PARAMETER_CLASS(Statistic,bits_per_neuron);

  // Learned bits per neuron per recovery period.
  VALUE_PARAMETER_CLASS(Statistic,bits_per_neuron_per_refractory_period);

  // Mutual information measure of learning.
  VALUE_PARAMETER_CLASS(Statistic,mutual_information);

  // Effective homology of output delay histogram (based on entropy)
  VALUE_PARAMETER_CLASS(Statistic,d_effective);

  // Histogram of delays for trained words.
  VALUE_PARAMETER_CLASS(Histogram,delay_histogram);

  // Histogram of delays for which neuron could have fired during
  // training.  This is usually right-shifted compared to delay_histogram
  // because the neuron fires in the the *first* slot it is able to fire.
  //
  VALUE_PARAMETER_CLASS(Histogram,input_delay_histogram);

  // Histogram of the delay which had the maximal sum value.
  VALUE_PARAMETER_CLASS(Histogram,input_max_sum_delay_histogram);

  // Histogram of the various summation values, regardless of delay.
  VALUE_PARAMETER_CLASS(Histogram,h_histogram);

  // Histogram delay values for the input words.
  VALUE_PARAMETER_CLASS(Histogram,word_delay_histogram);

  // Histogram of the synapse delay values before training.
  VALUE_PARAMETER_CLASS(Histogram,synapse_before_delay_histogram);

  // Histogram of the synapse delay values after training.
  VALUE_PARAMETER_CLASS(Histogram,synapse_after_delay_histogram);

 public:
  NeuronStatistics() { clear(); }

  void Clear() { clear(); }

  void clear() {
    clear_config();
    clear_false_false();
    clear_false_true();
    clear_false_count();
    clear_true_false();
    clear_true_true();
    clear_true_count();
    clear_q_after();
    clear_synapses_per_neuron();
    clear_bits_per_neuron();
    clear_bits_per_neuron_per_refractory_period();
    clear_mutual_information();
    clear_d_effective();
    clear_delay_histogram();
    clear_input_delay_histogram();
    clear_input_max_sum_delay_histogram();
    clear_h_histogram();
    clear_word_delay_histogram();
    clear_synapse_before_delay_histogram();
    clear_synapse_after_delay_histogram();
  }

  void CopyFrom(const NeuronStatistics& other) {
    clear();
    VALUE_COPY_CLASS(TrainConfig,config);
    VALUE_COPY_CLASS(Statistic,false_false);
    VALUE_COPY_CLASS(Statistic,false_true);
    VALUE_COPY_CLASS(Statistic,false_count);
    VALUE_COPY_CLASS(Statistic,true_false);
    VALUE_COPY_CLASS(Statistic,true_true);
    VALUE_COPY_CLASS(Statistic,true_count);
    VALUE_COPY_CLASS(Statistic,q_after);
    VALUE_COPY_CLASS(Statistic,synapses_per_neuron);
    VALUE_COPY_CLASS(Statistic,bits_per_neuron);
    VALUE_COPY_CLASS(Statistic,bits_per_neuron_per_refractory_period);
    VALUE_COPY_CLASS(Statistic,mutual_information);
    VALUE_COPY_CLASS(Statistic,d_effective);
    VALUE_COPY_CLASS(Histogram,delay_histogram);
    VALUE_COPY_CLASS(Histogram,input_delay_histogram);
    VALUE_COPY_CLASS(Histogram,input_max_sum_delay_histogram);
    VALUE_COPY_CLASS(Histogram,h_histogram);
    VALUE_COPY_CLASS(Histogram,word_delay_histogram);
    VALUE_COPY_CLASS(Histogram,synapse_before_delay_histogram);
    VALUE_COPY_CLASS(Histogram,synapse_after_delay_histogram);
  }
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    VALUE_SERIALIZE(TrainConfig,config);
    VALUE_SERIALIZE(Statistic,false_false);
    VALUE_SERIALIZE(Statistic,false_true);
    VALUE_SERIALIZE(Statistic,false_count);
    VALUE_SERIALIZE(Statistic,true_false);
    VALUE_SERIALIZE(Statistic,true_true);
    VALUE_SERIALIZE(Statistic,true_count);
    VALUE_SERIALIZE(Statistic,q_after);
    VALUE_SERIALIZE(Statistic,synapses_per_neuron);
    VALUE_SERIALIZE(Statistic,bits_per_neuron);
    VALUE_SERIALIZE(Statistic,bits_per_neuron_per_refractory_period);
    VALUE_SERIALIZE(Statistic,mutual_information);
    VALUE_SERIALIZE(Statistic,d_effective);
    VALUE_SERIALIZE(Histogram,delay_histogram);
    VALUE_SERIALIZE(Histogram,input_delay_histogram);
    VALUE_SERIALIZE(Histogram,input_max_sum_delay_histogram);
    VALUE_SERIALIZE(Histogram,h_histogram);
    VALUE_SERIALIZE(Histogram,word_delay_histogram);
    VALUE_SERIALIZE(Histogram,synapse_before_delay_histogram);
    VALUE_SERIALIZE(Histogram,synapse_after_delay_histogram);
  }
};

// Create a random number generator.
RandomBase* CreateRandom();

class Job {
 public:
  Job();
  virtual ~Job();

  virtual void Run() = 0;
};

void RunParallel(vector<Job*>* jobs);

}  // namespace

#endif  // COGNON_COMPAT_H_
