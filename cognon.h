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
// Simulator for the Cognon model of neural computation.
// The model consists of Neuron and its input word, contained
// in Wordset.
//
// First, a brief description of neuron function in the brain.
// A neuron has a set of synapses, which in the brain are connections
// between the neuron's input dendrites and the output axons of other
// neurons.  Signals must travel from a neuron, down the axon,
// cross a synapse to the dendrite of another neuron, where they
// travel the dendrite to the neuron's cell body, or soma.
// A neuron fires if "enough" signals reach the soma within
// a small time span, perhaps 30ms.  It is known that synapses
// change as a result of learning, and that new synapses are
// continually created.  Once a neuron has "fired", it generally
// needs some time to rebuild its energy stores before it can
// fire again, known as the "refractory period".  In addition,
// there is a variant where neural summation occurs in the
// branching input dendrites, and a neuron fires if any
// dendrite fires.
//
// The Cognon neural model is a digital simplification of the
// above description.  Time is quantized, and a neuron fires
// if the number of signals reaching the soma in a single
// "clock tick" crosses a threshold.  A neuron has a number
// of input synapses, and there is a random time delay in the
// range [0, ..., D2-1] that models the time for the signal
// to travel from the synapse to the soma.  Synapses may be
// randomly assigned to "compartments", which correspond to
// the root dendrite branches that may independently sum the
// inputs.  The number of compartments is governed by "C".
// C=1 implies summation in the soma, whereas C>1 implies
// summation in the dendrites.
//
// An input word represents the input signals to the neuron
// for a time period.  Assuming that all input neuron cells
// fire at the same time period, then the input values
// represent the time the signal takes to traverse the axon
// to reach the synapse, and are in the range 0...D1.  To
// represent those input neurons that did not fire at all
// we use the value "disabled", which is larger than D1+D2.
// The fraction of active inputs is assumed to be roughly
// 1 / R, where R is the refractory period.  Wordset is just
// that, a set of words which are presented to the Neuron.
//
// When presented with an input word, the neuron computes
// the vector sum of the input word vector (axon delay value)
// with the neuron's synapse vector (dendrite delay value).
// For each compartment, it then finds the smallest value in
// the range 0...(D1+D1-2) whose signal count crosses the
// threshold, if any, and fires with that delay.
//
// During training, a neuron is exposed to a series of
// words.  If the neuron fires, then all the synapses
// which participated in the summation that caused the
// firing are noted and either "frozen" (immortalized)
// according to the synapse atrophy model, or strengthened
// according to the synapse strength model.
//
// During operation and testing, synapses do not change
// strength or value, and the neuron either fires or
// doesn't, as the case may be.
//
// A word is deemed to have been learned if the neuron
// fires at the correct delay slot when the word is
// shown to the neuron.
//
// Statistic, Histogram, and NeuronStatistics are used to collect and
// pool various statistics about neuron learning and operation.
//
#ifndef COGNON_COGNON_H_
#define COGNON_COGNON_H_

#include <vector>

#include "compat.h"

namespace cognon {

// Represents that a synapse or input value is disabled.
// Chosen so neither kDisabled nor kDisabled + kDisabled
// result in valid delay slots or arithmetic overflow.
//
const int32 kDisabled = (1 << 29);

// Represents a small value used to compensate for various
// inaccuracies in, and limitations of, floating point
// arithmetic
//
const double kEpsilon = 1.0e-6;

// Add a new sample value to the statistic sample set.
void AddSample(double v, Statistic* stat);

// Compute the mean of the statistic sample set.
double Mean(const Statistic& stat);

// Compute the standard deviation of the statistic sample set.
double Stddev(const Statistic& stat);

// Summation creates the union of the two statistical sample sets.
// The internal values are summed, e.g. result.count = a.count + b.count,
// and the sample values are concatenated.
//
Statistic operator+(const Statistic& a, const Statistic& b);
Statistic& operator+=(Statistic& a, const Statistic& b);

// Add a new sample value to the statistic sample set in the bucket.
void AddSample(int32 bucket, double v, Histogram* stat);

// Given a vector of values, add each value to the statistic sample
// set in the appropriate bucket in the histogram.
//
void SetHistogram(const vector<int32>& data, Histogram* result);

// Summation of histograms operates on a per-bucket basis.
// It creates the Statistic summation (see above) for the
// samples in each bucket.
//
Histogram operator+(const Histogram& a, const Histogram& b);
Histogram& operator+=(Histogram& a, const Histogram& b);

// Summation of NeuronStatistics does the per-statistic and
// per-histogram summation (as above) of the two NeuronStatistics
// result sets.
//
NeuronStatistics operator+(const NeuronStatistics& a,
                           const NeuronStatistics& b);
NeuronStatistics& operator+=(NeuronStatistics& a,
                             const NeuronStatistics& b);

// Strip out the detailed result values, keeping just the
// summary statistics
//
void StatisticStripValues(Statistic* stats);
void HistogramStripValues(Histogram* stats);
void NeuronStatisticsStripValues(NeuronStatistics* stats);

// Needed to create a set<> of TrainConfig configurations
bool operator<(const TrainConfig&a, const TrainConfig& b);

// Train and test a single neuron using the given configuration
void RunExperiment(const TrainConfig& config, NeuronStatistics* result);

// Train and test repetitions neurons using the given configuration
void RunConfiguration(int32 repetitions,
                      const TrainConfig& config, NeuronStatistics* result);

}  // namespace cognon

#endif  // COGNON_COGNON_H_
