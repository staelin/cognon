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

#include "alice.h"
#include "neuron.h"
#include "wordset.h"

namespace cognon {

Alice::Alice() { }

void Alice::Train(Wordset* words, Neuron* neuron) {
  CHECK_NOTNULL(words);
  CHECK_NOTNULL(neuron);

  neuron->StartTraining();
  for (int32 i = 0; i < words->size(); ++i) {
    int32 delay = neuron->Train(words->get_word(i));
    if (delay < 0 || neuron->slots() <= delay)
      continue;
    words->set_delay(i, delay);
  }
  neuron->FinishTraining();
}

void Alice::TrainHistogram(Wordset* words,
                           Neuron* neuron,
                           vector<int32>* delay_histogram,
                           vector<int32>* input_delay_histogram,
                           vector<int32>* input_max_sum_delay_histogram,
                           vector<int32>* H_histogram) {
  CHECK_NOTNULL(words);
  CHECK_NOTNULL(neuron);
  CHECK_NOTNULL(delay_histogram);
  CHECK_NOTNULL(input_delay_histogram);
  CHECK_NOTNULL(input_max_sum_delay_histogram);
  CHECK_NOTNULL(H_histogram);

  neuron->StartTraining();
  for (int32 i = 0; i < words->size(); ++i) {
    int32 delay = neuron->Train(words->get_word(i));
    if (delay < 0 || neuron->slots() <= delay)
      continue;
    words->set_delay(i, delay);
    if (delay_histogram->size() < delay + 1)
      delay_histogram->resize(delay + 1);
    (*delay_histogram)[delay]++;
    neuron->GetInputDelayHistogram(words->get_word(i),
                                   input_delay_histogram,
                                   input_max_sum_delay_histogram,
                                   H_histogram);
  }
  neuron->FinishTraining();
}

}  // namespace cognon
