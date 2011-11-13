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
// The Neuron class is the neuron model.  Since neural training only impacts
// synapses, either immediately during each training signal exposure, or later
// at the end of training, the model contains hooks to change these behaviors
// in the form of permanent callbacks update_synapse and atrophy_synapse.
//
#ifndef COGNON_NEURON_H_
#define COGNON_NEURON_H_


#include <vector>
#include "cognon.h"
#include "wordset.h"

namespace cognon {

class Neuron;

class Learn {
 public:
  explicit Learn(Neuron* neuron);
  virtual ~Learn();

  virtual void StartTraining() = 0;
  virtual void UpdateSynapse(int synapse) = 0;
  virtual void FinishTraining() = 0;
 protected:
  Neuron* neuron_;
};

class LearnSynapseAtrophy : public Learn {
 public:
  explicit LearnSynapseAtrophy(Neuron* neuron);
  virtual ~LearnSynapseAtrophy();

  virtual void StartTraining();
  virtual void UpdateSynapse(int synapse);
  virtual void FinishTraining();
};

class LearnSynapseStrength : public Learn {
 public:
  explicit LearnSynapseStrength(Neuron* neuron);
  virtual ~LearnSynapseStrength();

  virtual void StartTraining();
  virtual void UpdateSynapse(int synapse);
  virtual void FinishTraining();
};

// Not thread safe
// training: because of updates to delays_, frozen_, and sum_
// testing: because of updates to sum_
//
class Neuron {
 public:
  Neuron();
  virtual ~Neuron() { }

  // Initializes a neuron
  virtual void Init(const NeuronConfig& config);

  // Expose a neuron to a word.
  //
  // A word is a random vector of [0, ..., D1-1, kDisabled] values, with an
  // input signal (non-disabled value) roughly every R slots.
  //
  int32 Expose(const Word& word);

  // Train a neuron to recognize a word.
  //
  // A word is a random vector of [0, ..., d1-1, kDisabled] values,
  // with non-disabled values on average every R slots.
  //
  int32 Train(const Word& word);

  // Start a new training cycle.
  void StartTraining();

  // Finished a training cycle, so update synapses and statistics
  // as appropriate.
  //
  void FinishTraining();

  // Accumulate histograms
  //
  // histogram: histogram of delays that could fire
  // max_histogram: histogram of delay with maximum firing sum
  // H_histogram: histogram of container summation values
  //
  void GetInputDelayHistogram(const Word& word,
                             vector<int32>* histogram,
                             vector<int32>* max_histogram,
                             vector<int32>* H_histogram);

  void GetSynapseDelayHistogram(vector<int32>* histogram);

  // Various accessor functions to report the Neuron's configuration
  const NeuronConfig& config() { return config_; }
  const int32 C() const { return C_; }
  const int32 D1() const { return D1_; }
  const int32 D2() const { return D2_; }
  const int32 slots() const { return (D1_ + D2_); }  // return delay spread
  const double H() const { return H_; }
  const double Q() const { return Q_; }
  const double Q_after() const { return Q_after_; }
  const int32 R() const { return R_; }
  const double G_m() const { return G_m_; }
  const double H_m() const { return H_m_; }
  const int32 length() const { return length_; }

  void set_H(double value) { H_ = value; }

  const int32 delays(int32 i) const { return delays_[i]; }
  void set_delays(int32 i, int32 value) { delays_[i] = value; }

  const int32 containers(int32 i) const { return containers_[i]; }
  void set_containers(int32 i, int32 value) { containers_[i] = value; }

  const bool frozen(int32 i) const { return frozen_[i]; }
  void set_frozen(int32 i, bool value) { frozen_[i] = value; }

  const double strength(int32 i) const { return strength_[i]; }
  void set_strength(int32 i, double value) { strength_[i] = value; }

  const double sum(int32 i) const { return sum_[i]; }
  void set_sum(int32 i, double value) { sum_[i] = value; }

 private:
  NeuronConfig config_;  // Configuration data
  int32 C_;         // Number of containers
  int32 D1_;        // Number of input delays
  int32 D2_;        // Number of axon delays
  double H_;        // Firing threshold
  double Q_;        // Oversampling rates
  double Q_after_;  // Q after training
  int32 R_;         // Refractory period
  double G_m_;      // Increment synapse strength by this amount
  double H_m_;      // Synapse-strength threshold value

  scoped_ptr<RandomBase> random_;  // Pointer to random number generator
  int32 length_;                   // The number of synapses
  vector<int32> delays_;      // The delay for each synapse
  vector<int32> containers_;  // The container id for each synapse
  vector<bool> frozen_;       // Is the synapse frozen?
  vector<double> strength_;   // Strength of the synapse
  vector<double> sum_;        // Per container summation values
  scoped_ptr<Learn> learn_;        // Modifies neuron during learning
};

}  // namespace cognon

#endif  // COGNON_NEURON_H_
