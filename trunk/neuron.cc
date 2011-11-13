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
// Cognon neuron simulator implementation.
//

#include "neuron.h"

#include <math.h>
#include <utility>

#include "cognon.h"
#include "wordset.h"

namespace cognon {

Learn::Learn(Neuron* neuron) : neuron_(neuron) { }

Learn::~Learn() { }

LearnSynapseAtrophy::LearnSynapseAtrophy(Neuron* neuron) : Learn(neuron) { }

LearnSynapseAtrophy::~LearnSynapseAtrophy() { }

void LearnSynapseAtrophy::StartTraining() { }

void LearnSynapseAtrophy::UpdateSynapse(int synapse) {
  neuron_->set_frozen(synapse, true);
}

void LearnSynapseAtrophy::FinishTraining() {
  for (int32 i = 0; i < neuron_->length(); ++i) {
    if (!neuron_->frozen(i)) {
      neuron_->set_strength(i, 0.0);
      neuron_->set_delays(i, kDisabled);
    }
  }
}

LearnSynapseStrength::LearnSynapseStrength(Neuron* neuron) : Learn(neuron) { }

LearnSynapseStrength::~LearnSynapseStrength() { }

void LearnSynapseStrength::StartTraining() {
  CHECK(neuron_->config().has_h());
  neuron_->set_H(neuron_->config().h());
}

void LearnSynapseStrength::UpdateSynapse(int synapse) {
  CHECK(neuron_->config().has_g_m());
  neuron_->set_strength(synapse, neuron_->config().g_m());
  neuron_->set_frozen(synapse, true);
}

void LearnSynapseStrength::FinishTraining() {
  CHECK(neuron_->config().has_h_m());
  neuron_->set_H(neuron_->config().h_m());
}

Neuron::Neuron()
    : C_(1), D1_(1), D2_(1), H_(1.0), Q_(1.0), Q_after_(-1.0), R_(1),
      G_m_(-1.0), H_m_(-1.0), random_(NULL), length_(-1),
      delays_(), containers_(), frozen_(), sum_(), learn_(NULL) {
  random_.reset(CreateRandom());
}

void Neuron::Init(const NeuronConfig& config) {
  CHECK(config.has_c());
  CHECK(config.has_d1());
  CHECK(config.has_d2());
  CHECK(config.has_h());
  CHECK(config.has_q());
  CHECK(config.has_r());
  CHECK(1 <= config.c());
  CHECK(config.d1() <= config.d2());
  CHECK(1 <= config.h());
  //  CHECK(1 < config.r());  // Ensure sparsity constraint.

  // Intialize neuron values
  config_ = config;
  C_ = config.c();
  D1_ = config.d1();
  D2_ = config.d2();
  H_ = config.h();
  Q_ = config.q();
  R_ = config.r();
  Q_after_ = -1.0;
  if (config.has_g_m()) {
    G_m_ = config.g_m();
  } else {
    G_m_ = -1.0;
  }
  if (config.has_h_m()) {
    H_m_ = config.h_m();
  } else {
    H_m_ = -1.0;
  }
  length_ = static_cast<int32>(floor(C_ * H_ * Q_ * R_ + kEpsilon));

  // Initialize each synapses' delay, container, and frozen status
  if (delays_.size() != length_) {
    delays_.resize(length_);
    containers_.resize(length_);
    frozen_.resize(length_);
    strength_.resize(length_);
    sum_.resize(C_);
  }

  // Randomly assign delays and containers to each synapse
  for (int32 i = 0; i < length_; ++i) {
    // Randomly assign delays to synapses
    delays_[i] = random_->Rand32() % D2_;
    CHECK(0 <= delays_[i] && delays_[i] < D2_);

    // containers_[i] = i % C_;             // Assign synapses to containers

    // Randomly assign synapses to containers
    containers_[i] = random_->Rand32() % C_;
    CHECK(0 <= containers_[i] && containers_[i] < C_);

    frozen_[i] = false;
    strength_[i] = 1.0;
  }

  if (config.has_g_m() && config.has_h_m()) {
    learn_.reset(new LearnSynapseStrength(this));
  } else {
    learn_.reset(new LearnSynapseAtrophy(this));
  }
}

int32 Neuron::Expose(const Word& word) {
  CHECK(sum_.size() == C_);
  CHECK(delays_.size() == length_);
  CHECK(strength_.size() == length_);
  CHECK(containers_.size() == length_);

  // Iterate over delays until neuron fires occurs
  int s = slots();
  for (int32 d = 0; d < s; ++d) {
    for (int32 i = 0; i < C_; ++i) {
      sum_[i] = 0.0;
    }
    // Iterate over sparse signals in word
    for (Word::const_iterator it = word.begin(); it != word.end(); ++it) {
      int32 synapse = it->first;
      int32 delay = it->second;
      CHECK(0 <= synapse && synapse < length_);
      CHECK(delay == kDisabled || (0 <= delay && delay < D1_));

      // If delay and word delay adds up to current delay then
      // increment the container's sum by that synapse's strength
      if (delays_[synapse] + delay == d) {
        CHECK(0 <= containers_[synapse] && containers_[synapse] < C_);
        sum_[containers_[synapse]] += strength_[synapse];
      }
    }

    // Iterate through containers to see if any fired
    for (int32 i = 0; i < C_; ++i) {
      // If at least H firings, then denote that container as fired
      if (H_ <= sum_[i] + kEpsilon) return d;
    }
  }
  return kDisabled;
}

int32 Neuron::Train(const Word& word) {
  CHECK(sum_.size() == C_);
  CHECK(delays_.size() == length_);
  CHECK(containers_.size() == length_);
  CHECK_NOTNULL(learn_.get());

  int32 d = Expose(word);

  if (d == kDisabled) return d;

  // Iterate through containers, updating synapses in those that fired
  for (int32 i = 0; i < C_; ++i) {
    if (sum_[i] + kEpsilon < H_) continue;
    // Update those synapses that contributed to the neuron firing
    for (Word::const_iterator it = word.begin(); it != word.end(); ++it) {
      int32 synapse = it->first;
      int32 delay = it->second;

      if (delays_[synapse] + delay == d && containers_[synapse] == i) {
        learn_->UpdateSynapse(synapse);
      }
    }
  }
  return d;
}

void Neuron::StartTraining() {
  CHECK_NOTNULL(learn_.get());

  learn_->StartTraining();
}

void Neuron::FinishTraining() {
  CHECK_NOTNULL(learn_.get());
  CHECK(frozen_.size() == length_);

  learn_->FinishTraining();

  int count = 0;
  for (int32 i = 0; i < length_; ++i) {
    if (frozen_[i]) count++;
  }
  Q_after_ = count / static_cast<double>(length_);
}

void Neuron::GetInputDelayHistogram(const Word& word,
                                   vector<int32>* histogram,
                                   vector<int32>* max_histogram,
                                   vector<int32>* H_histogram) {
  int32 m = -1;  // delay for maximum sum within a delay-container pair
  int32 s = slots();
  double max_sum = -1.0;  // current maximum sum

  if (histogram->size() < s + 1) histogram->resize(s + 1);
  if (max_histogram->size() < s + 1) max_histogram->resize(s + 1);

  for (int32 d = 0; d < s; ++d) {
    for (int32 i = 0; i < C_; ++i) {
      sum_[i] = 0.0;
    }
    // Iterate over sparse signals in word
    for (Word::const_iterator it = word.begin(); it != word.end(); ++it) {
      int32 synapse = it->first;
      int32 delay = it->second;

      // If delay and word delay adds up to current delay then
      // increment the container's sum by that synapse's strength
      if (delays_[synapse] + delay == d) {
        sum_[containers_[synapse]] += strength_[synapse];
      }
    }

    // Find the maximal summation value (over all delays)
    for (int32 i = 0; i < C_; ++i) {
      if (m < 0 || max_sum < sum_[i]) {
        m = d;
        max_sum = sum_[i];
      }
    }

    for (int32 i = 0; i < C_; ++i) {
      // This delay & container would fire, so add it to the histogram
      if (H_ <= sum_[i] + kEpsilon) (*histogram)[d]++;

      // Record all delay & container summations
      int bucket = floor(sum_[i] + kEpsilon);
      if (H_histogram->size() <= bucket)
        H_histogram->resize(bucket + 1);
      (*H_histogram)[bucket]++;
    }
  }

  // Record the maximal summation over all delays and containers
  if (0 <= m) (*max_histogram)[m]++;
}

void Neuron::GetSynapseDelayHistogram(vector<int32>* histogram) {
  CHECK_NOTNULL(histogram);

  if (histogram->size() < D2_ + 1)
    histogram->resize(D2_ + 1);
  for (int32 i = 0; i < length_; ++i) {
    if (0 <= delays_[i] && delays_[i] < D2_)
      (*histogram)[delays_[i]]++;
  }
}

}  // namespace cognon
