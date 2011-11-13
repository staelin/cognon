#ifndef COGNON_ORIG_H
#define COGNON_ORIG_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "compat.h"

namespace cognon_orig {

#ifdef WIN32
#include <float.h>
inline double log2(double v) { return log(v) / 1.44269504088896340736; }
double lgamma(double v);
double fmax (double _x, double _y) { return ( !_isnan(_x) && !_isnan(_y) ? _x >= _y : (_isnan (_y) ?  _x : _y )); }
#endif

#define COGNON_H_VERSION "$Rev: 82 $"

const int kDisabled = (1 << 29);

class Statistics
{
	int		n;
	int*	count;
	double*	sum;
	double* ssum;

public:

	// Fill new statistic data structure with zeros, e.g. counts, sums, squared sums
	Statistics(int _n) : n(_n), count(new int[_n]), sum(new double[_n]), ssum(new double[_n])
	{
		reset();
	}

	~Statistics()
	{
		delete [] count;
		delete [] sum;
		delete [] ssum;
	}

	void reset()
	{
		int i;

		for (i = 0; i < n; ++i) {
			count[i] = 0;
			sum[i] = 0.0;
			ssum[i] = 0.0;
		}
	}

	// Adds a new sample (v) to the i'th experiment
	void sample(int i, double v)
	{
		if (n <= i) {
			return;
		}
		count[i]++;
		sum[i] += v;
		ssum[i] += v * v;
	}

	// Returns the mean of the i'th experiment
	double mean(int i)
	{
		if (n <= i) {
			return -1.0;
		}
		return sum[i] / count[i];
	}

	// returns standard deviation of i'th experiment
	double stddev(int i)
	{
		if (n <= i) {
			return -1.0;
		}
		double v = (count[i] * ssum[i] - sum[i] * sum[i]) / (count[i] * (count[i] - 1));
		if (v < 0.0)
			return 0.0;
		return sqrt(v);

		/**********************************************************
		// compute the standard error for the sample mean estimator
		**********************************************************/
		double s2;  // standard error squared of trials
		double mu; // sample mean of trials
		double mu_std; // standard error for sample mean
		if (count[i] > 1){
			mu = sum[i]/count[i];
			s2 = (count[i] * ssum[i] - sum[i] * sum[i]) / (count[i] * (count[i] - 1));
			mu_std = (1/count[i]) * (s2 + pow(mu,2)) - pow(mu,2);
			return mu_std;
		}
		else
			return 0;
	}
};




class Neuron
{
 public:
	int l; // So;
	int* delays; // the delay for each of the CQHR synapse connections
	int* containers; // the container id for each of the CQHR synapse connections
	char* frozen; // is the synapse frozen?
	double* sum;

	int C; // number of containers
	int D1; // number of input delays
	int D2; // number of axon delays
	double H; // firing threshold
	double Q; // oversampling rates
	int R; // refractory period
	double Qafter; // Q after training
	MTRand r;

	Neuron() :
		l(-1), delays(NULL), containers(NULL), frozen(NULL), sum(NULL),
                C(1), D1(1), D2(1), H(1.0), Q(1), R(1), Qafter(-1.0),
                r(rand())
	{
	}

	// Initializes the neuron parameters
	Neuron(int CC, int DD1, int DD2, double HH, double QQ, int RR) :
		l(-1), delays(NULL), containers(NULL), frozen(NULL), sum(NULL),
                C(CC), D1(DD1), D2(DD2), H(HH), Q(QQ), R(RR), Qafter(-1.0),
                r(rand())
	{
	}

	virtual ~Neuron()
	{
		delete [] containers;
		delete [] frozen;
		delete [] sum;
		delete [] delays;
	}

	// return So
	int length() { return l; }

	// return delay spread
	int slots() { return (D1 + D2); }

	// initializes a neuron
	virtual void initialize(int CC = -1, int DD1 = -1, int DD2 = -1, double HH = -1.0, double QQ = -1.0, int RR = -1)
	{
		int i;

		// intialize neuron values
		if (0 < CC) {
			C = CC;
			D1 = DD1;
			D2 = DD2;
			H = HH;
			Q = QQ;
			R = RR;
			Qafter = -1.0;
			l = (int)floor(C * H * Q * R + 1.0e-6);
		}

		// initialize what each synapses delay, container, frozen status is
		if (!delays) {
			delays = new int[l];
			containers = new int[l];
			frozen = new char[l];
			sum = new double[C];
		}

		// randomly assign delays and containers to each synapse
		for (i = 0; i < l; ++i) {
			delays[i] = r.randInt() % D2; // randomly assign synapses to delays
			// containers[i] = i % C; // r.randInt() % C; // randomly assign synapses to containers
			containers[i] = r.randInt() % C; // randomly assign synapses to containers
		}

		// set froze status to zero for all synapses
		memset(frozen, 0, l * sizeof(char));
	}

	/*
	 * A word is a random vector of [0,1] values,
	 * with a 1 on average every R slots
	 */

	virtual int train(int* word, int do_freeze = 0)
	{
		int _d, d, i, j, s = slots(), fired = 0;
		// this appears to always be zero
		int offset = (0 && do_freeze ? r.randInt() % s : 0);

		// iterate through each delay until done or a firing occurs
		for (_d = 0; _d < s && !fired; ++_d) {
			// this doesn't do anything
			d = (_d + offset) % s;

			// set number of firings in each container to zero
			for (i = 0; i < C; ++i)
				sum[i] = 0.0;
			// iterate each synapse
			for (i = 0; i < l; ++i) {
				// if delay and word delay adds up to current delay then add an alignment to its container
				if (delays[i] + word[i] == d) {
					sum[containers[i]]++;
				}
			}

			/// iterate through containers
			for (i = 0; i < C; ++i) {

				// if atleat H firings then denote that container as fired
				if (H <= sum[i] + 1.0e-6) {
				// if ((do_freeze ? H : H - 1) <= sum[i]) {
					// FIRED!
					fired = 1;

					// freeze those synapses that contributed
					if (do_freeze) {
						// Freeze those synapses that fired
						for (j = 0; j < l; ++j) {
							if (delays[j] + word[j] == d && containers[j] == i)
								frozen[j] = 1;
						}
					}
					break;
				}
			}
		}

		return (fired ? d : kDisabled);
	}

	// all unfrozen synapses are disabled
	virtual void clean()
	{
		int i;
		int count = 0;

		for (i = 0; i < l; ++i) {
			if (frozen[i]) {
				count++;
			} else {
				delays[i] = kDisabled; // never fires with this value; disables synapse
			}
		}
		Qafter = count / (double)(C * H * R);
	}

	void train_input_delay_histogram(int* word, int* histogram, int* max_histogram, int* H_histogram)
	{
		int d, i, s = slots();

		//		for (i = 0; i < l; ++i) {
		//			histogram[delays[i] + word[i]]++;
		//		}
		//		return;

		// iterate through the delays, until at least one
		// container at one delay fires
		//

		// CHS XXX:
		// Keep a histogram of the maximal sum for any
		// container (to control the number of buckets
		// limit the histogram to [H - 5, ..., H + 5])
		//

		int m = -1;
		int max_sum = -1;
		for (d = 0; d < s; ++d) {
			memset(sum, 0, C * sizeof(int));
			for (i = 0; i < l; ++i) {
				if (delays[i] + word[i] == d) {
					sum[containers[i]]++;
				}
			}
			for (i = 0; i < C; ++i) {
				if (m < 0 || max_sum < sum[i]) {
					m = d;
					max_sum = sum[i];
				}
				if (H <= sum[i] + 1.0e-6) {
					histogram[d]++;
				}
				if (H_histogram) {
					int bucket = floor(sum[i] + 1.0e-6) - H + 5;
                                        if (bucket < 0) bucket = 0;
                                        else if (10 < bucket) bucket = 10;
                                        H_histogram[bucket]++;
				}
			}
			if (0 <= m && max_histogram)
				max_histogram[m]++;
		}
	}
};

class NeuronStrength : public Neuron
{
public:
	double G_m;
	double H_m;
	double* strength;

	NeuronStrength() : Neuron(), G_m(-1.0), H_m(-1.0), strength(NULL)
	{
	}

	virtual ~NeuronStrength()
	{
		delete [] strength;
	}

	void SetG_m(double v) { G_m = v; }
	void SetH_m(double v) { H_m = v; }

	// initializes a neuron
	virtual void initialize(int CC = -1, int DD1 = -1, int DD2 = -1, double HH = -1.0, double QQ = -1.0, int RR = -1)
	{
		int i;

		Neuron::initialize(CC, DD1, DD2, HH, QQ, RR);

		// initialize what each synapses delay, container, frozen status is
		if (!strength) {
			strength = new double[l];
		}

		// randomly assign delays and containers to each synapse
		for (i = 0; i < l; ++i) {
			strength[i] = 1.0;
		}
	}

	virtual int train(int* word, int do_freeze = 0)
	{
		int _d, d, i, j, s = slots(), fired = 0;
		// this appears to always be zero
		int offset = (0 && do_freeze ? r.randInt() % s : 0);

		// iterate through each delay until done or a firing occurs
		for (_d = 0; _d < s && !fired; ++_d) {
			// this doesn't do anything
			d = (_d + offset) % s;

			// set number of firings in each container to zero
			for (i = 0; i < C; ++i)
				sum[i] = 0.0;
			// iterate each synapse
			for (i = 0; i < l; ++i) {
				// if delay and word delay adds up to current delay then add an alignment to its container
				if (delays[i] + word[i] == d) {
					sum[containers[i]] += strength[i];
				}
			}

			/// iterate through containers
			for (i = 0; i < C; ++i) {

				// if atleat H firings then denote that container as fired
				if ((do_freeze && H <= sum[i] + 1.0e-6)
                                    || (!do_freeze && H_m <= sum[i] + 1.0e-6)) {
				// if ((do_freeze ? H : H - 1) <= sum[i]) {
					// FIRED!
					fired = 1;

					// freeze those synapses that contributed
					if (do_freeze) {
						// Freeze those synapses that fired
						for (j = 0; j < l; ++j) {
							if (delays[j] + word[j] == d && containers[j] == i) {
								strength[j] = G_m;
								frozen[j] = 1;
							}
						}
					}
					break;
				}
			}
		}

		return (fired ? d : kDisabled);
	}

	// all unfrozen synapses are disabled
	virtual void clean()
	{
		int i;
		int count = 0;

		for (i = 0; i < l; ++i) {
			if (frozen[i]) {
				count++;
			}
		}
		Qafter = count / (double)(C * H * R);
	}

};

class Wordset
{
public:
	int n; // number of words
	int l; // length of words (So)
	int D; // number of delays
	int R; // refractory period;
	int** words;
	int* delays;
	MTRand r;

	Wordset() : n(-1), l(-1), D(-1), R(-1), words(NULL), delays(NULL), r(rand())
	{
	}

	Wordset(int _n, int _l, int _D, int _R) : n(_n), l(_l), D(_D), R(_R), words(NULL), delays(NULL), r(rand())
	{
		initialize(_n, _l, _D, _R);
	}

	~Wordset()
	{
		if (words)
			free(words);
		words = NULL;
		if (delays)
			free(delays);
		delays = NULL;
	}

	void seed(int s) { r.seed(s); }

	// randomly set one of every R synapses
	void initialize(int _n = -1, int _l = -1, int _D = -1, int _R = -1)
	{
		int i, N;
		int* w;

		if (0 < _n) {
			if (n != _n) {
				if (delays)
					free(delays);
				delays = NULL;
				if (words)
					free(words);
				words = NULL;
			}
			n = _n;
		}
		if (0 < _l) l = _l;
		if (0 < _D) D = _D;
		if (0 < _R) R = _R;

		if (!words) {
			int* p;

			words = (int**)malloc((n + 1) * sizeof(int*) + n * l * sizeof(int));

			for (i = 0, p = (int*)&words[n]; i < n; ++i) {
				words[i] = p;
				p += l;
			}
		}

		N = n * l;
		w = words[0];

		for (i = 0; i < n * l; ++i) {
			w[i] = kDisabled;
			if ((r.randInt() % R) == 0)
				w[i] = r.randInt() % D;
			//			printf("\tw[%d]=%d", i, w[i]);
		}

		if (delays) {
			for (i = 0; i < n; ++i) {
				delays[i] = kDisabled;
			}
		}
	}

	// get/set the delay slot for a given word
	int get_delay(int w) {
		return (delays ? delays[w] : kDisabled);
	}
	int set_delay(int w, int v)
	{
		int i;
		if (!delays) {
	  		delays = (int*)malloc(n * sizeof(int));
			for (i = 0; i < n; ++i) {
				delays[i] = kDisabled;
			}
		}
	 	return (delays[w] = v);
	}
};


class Alice
{
public:
	Neuron* neuron;
	Wordset* words;

	Alice(Neuron* _neuron, Wordset* _words, int* input_delay_histogram = NULL, int* input_max_sum_delay_histogram = NULL, int* H_histogram = NULL)
	  : neuron(_neuron), words(_words)
	{
		int i;

		for (i = 0; i < words->n; ++i) {
			int delay = neuron->train(words->words[i], 1);
			if (input_delay_histogram)
				neuron->train_input_delay_histogram(words->words[i], input_delay_histogram, input_max_sum_delay_histogram, H_histogram);
			words->set_delay(i, delay);
		}
	}
};

class Bob
{
public:
	// <ground truth, neuron's answer>
	int false_false;
	int false_true; // false positive
	int true_false; // false negative
	int true_true;
	int count;
	double Qafter_sum;
	double Qafter;
	int* delay_histogram;
	int* input_delay_histogram;
	int* input_max_sum_delay_histogram;
	int* H_histogram;
	int* word_delay_histogram;
	int* synapse_before_delay_histogram;
	int* synapse_after_delay_histogram;
	int  delay_n;
	int  C;

	Bob() : false_false(0),
		false_true(0) /* false positive */,
		true_false(0) /* false negative */,
		true_true(0),
		count(0),
		Qafter_sum(0),
		Qafter(0),
		delay_histogram(NULL),
		input_delay_histogram(NULL),
		input_max_sum_delay_histogram(NULL),
		H_histogram(NULL),
		word_delay_histogram(NULL),
		synapse_before_delay_histogram(NULL),
		synapse_after_delay_histogram(NULL),
		delay_n(0),
		C(0)
	{
	}

	~Bob()
	{
		delete [] delay_histogram;
		delete [] input_delay_histogram;
		delete [] input_max_sum_delay_histogram;
		delete [] H_histogram;
		delete [] word_delay_histogram;
		delete [] synapse_before_delay_histogram;
		delete [] synapse_after_delay_histogram;
	}

	void initialize(Neuron* neuron)
	{
		false_false = 0;
		false_true = 0; // false positive
		true_false = 0; // false negative
		true_true = 0;
		count = 0;
		Qafter_sum = 0.0;
		Qafter = 0.0;

		if (neuron && delay_n != neuron->slots()) {
			delete [] delay_histogram;
			delete [] input_delay_histogram;
			delete [] input_max_sum_delay_histogram;
			delete [] H_histogram;
			delete [] word_delay_histogram;
			delete [] synapse_before_delay_histogram;
			delete [] synapse_after_delay_histogram;

			delay_histogram = NULL;
			input_delay_histogram = NULL;
			input_max_sum_delay_histogram = NULL;
			H_histogram = NULL;
			word_delay_histogram = NULL;
			synapse_before_delay_histogram = NULL;
			synapse_after_delay_histogram = NULL;

			delay_n = neuron->slots();
			C = neuron->C;
			if (delay_n) {
				delay_histogram = new int[delay_n];
				input_delay_histogram = new int[delay_n];
				input_max_sum_delay_histogram = new int[delay_n];
				H_histogram = new int[11];
				word_delay_histogram = new int[delay_n];
				synapse_before_delay_histogram = new int[delay_n];
				synapse_after_delay_histogram = new int[delay_n];
			}
		}
		if (delay_histogram) {
			memset(delay_histogram, 0, delay_n * sizeof(int));
			memset(input_delay_histogram, 0, delay_n * sizeof(int));
			memset(input_max_sum_delay_histogram, 0, delay_n * sizeof(int));
			memset(H_histogram, 0, 11 * sizeof(int));
			memset(word_delay_histogram, 0, delay_n * sizeof(int));
			memset(synapse_before_delay_histogram, 0, delay_n * sizeof(int));
			memset(synapse_after_delay_histogram, 0, delay_n * sizeof(int));
		}
	}

	// Given a set of (hopefully) learned and a set of test
	// words, collect the confusion matrix statistics
	//
	void test(Neuron* neuron, Wordset* train, Wordset* test)
	{
		int i, j;

		if (!delay_histogram) {
			initialize(neuron);
		}

		if (train) {
			for (i = 0; i < train->n; i++) {
				if (neuron->train(train->words[i]) == train->get_delay(i)
				    && 0 <= train->get_delay(i)
					&& train->get_delay(i) < neuron->slots())
				{
					true_true++;
					delay_histogram[train->get_delay(i)]++;
				} else {
					true_false++;
				}
				for (j = 0; j < train->l; ++j) {
					if (train->words[i][j] < neuron->slots())
						word_delay_histogram[train->words[i][j]]++;
				}
			}
			count++;
			Qafter_sum += neuron->Qafter;
			Qafter = Qafter_sum / count;
		}

		if (test) {
			for (i = 0; i < test->n; i++) {
				if (0 <= neuron->train(test->words[i])) {
					false_true++;
				} else {
					false_false++;
				}
				for (j = 0; j < test->l; ++j) {
					if (test->words[i][j] < neuron->slots())
						word_delay_histogram[test->words[i][j]]++;
				}
			}
		}
	}

	Bob& operator+=(Bob& other)
	{
		false_false += other.false_false;
		false_true += other.false_true; // false positive
		true_false += other.true_false; // false negative
		true_true += other.true_true;
		count += other.count;
		Qafter_sum += other.Qafter_sum;
		Qafter = Qafter_sum / count;

		if (other.delay_histogram) {
			if (!delay_histogram) {
				delay_n = other.delay_n;
				delay_histogram = new int[delay_n];
				memcpy(delay_histogram, other.delay_histogram, delay_n * sizeof(int));
				input_delay_histogram = new int[delay_n];
				memcpy(input_delay_histogram, other.input_delay_histogram, delay_n * sizeof(int));
				input_max_sum_delay_histogram = new int[delay_n];
				memcpy(input_max_sum_delay_histogram, other.input_delay_histogram, delay_n * sizeof(int));
				H_histogram = new int[11];
				memcpy(H_histogram, other.H_histogram, 11 * sizeof(int));
				word_delay_histogram = new int[delay_n];
				memcpy(word_delay_histogram, other.word_delay_histogram, delay_n * sizeof(int));
				synapse_before_delay_histogram = new int[delay_n];
				memcpy(synapse_before_delay_histogram, other.synapse_before_delay_histogram, delay_n * sizeof(int));
				synapse_after_delay_histogram = new int[delay_n];
				memcpy(synapse_after_delay_histogram, other.synapse_after_delay_histogram, delay_n * sizeof(int));
			} else {
				int i;
				for (i = 0; i < other.delay_n; ++i) {
					delay_histogram[i] += other.delay_histogram[i];
					input_delay_histogram[i] += other.input_delay_histogram[i];
					input_max_sum_delay_histogram[i] += other.input_max_sum_delay_histogram[i];
					word_delay_histogram[i] += other.word_delay_histogram[i];
					synapse_before_delay_histogram[i] += other.synapse_before_delay_histogram[i];
					synapse_after_delay_histogram[i] += other.synapse_after_delay_histogram[i];
				}
				for (i = 0; i < 11; ++i) {
					H_histogram[i] += other.H_histogram[i];
				}
			}
		}
		return *this;
	}

	double bpn(int W, double D)
	{
		int i;
		double a = 2.0; // for now
		double pF = (1.0 / 360.0) + false_true / (double)(false_false + false_true);
		double pL = true_true / (double)(true_true + true_false);
		int n = (int)(1.0 / (a * pF));
		int k = W;

		double log_choices = 0.0;

		if (n > k) {
			/* log2(n! / (k! * (n - k)!)) */
			for (i = 1; i <= n; ++i)
				log_choices += log2(i);

			for (i = 1; i <= k; ++i)
				log_choices -= log2(i);

			for (i = 1; i <= n - k; ++i)
				log_choices -= log2(i);
		}

		return (W * pL * log2(D) + log_choices);
	}

	// return the (natural) log of the choose function
	double lchoose(double n, double k)
	{
		double l = lgamma(n + 1) - lgamma(n - k + 1) - lgamma(k + 1);
		return l;
	}


	// This function calculates the information stored by a single neuron
	double bpn_keith(double H, double Q, int R, int C, int D1, int W)
	{

		// Calculate the false alarm and learning probabilities
		double Pf = (1.0 / 360.0) + false_true / (double)(false_false + false_true);
		double Pl = true_true / (double)(true_true + true_false);

		// Calculate bits
		double infoValue = W*(log(1.0 - Pl) - log(1.0 - Pf) - Pl*log(1.0 - Pl) + Pl*log(1.0 - Pf) + Pl*log(Pl) - Pl*log(Pf))/log(2.0);

		// return the calculated bits
		return infoValue;
	}


	// This function calculates the information stored by a single neuron
	double bpn_keith_old(int H, double Q, int R, int C, int D1, int W)
	{
		double logZ; // Size of the input space.
		double infoValue;
		double newLogZ;
		double newW;

		// Calculate the false alarm and learning probabilities
		double Pf = (1.0 / 360.0) + false_true / (double)(false_false + false_true);
		double Pl = true_true / (double)(true_true + true_false);


		// Calculate log(Z)
		// Z = C* (HQR)!/((HQR-HQ)!*(HQ)!) * D^(HQ)
		// -> log(Z) = log(C) + gammalog(HQR + 1) - gammalog(HQR-HQ + 1) - gammalog(HQ+1) + H*Q*log(D)
		logZ = log((double)C) + lgamma(H*Q*R + 1) - lgamma(H*Q*R - H*Q + 1) - lgamma(H*Q+1) + H*Q*log((double)D1);

		//printf("Z: %f \n \n",logZ);
		// Now calculate the information of the NN
		// I = log2(term1) - log2(term2) - log2(term3)
		// term1 = C(Z,W)
		// term2 = C(Z*pF - W*pF + W*pL, W*pL)
		// term3 = C(Z*(1-pF) + W*pF - W*pL, W*(1-pL))

		// First calculate term1
		// term1 = Z!/( (Z-W)! * W! )
		// term1 = (Z*(Z-1)*...*(Z-W+1))/W!
		// log(term1) = log(Z) + log(Z-1) + .. + log(Z-W+1) - log(W!)
		// log(term1) ~ W*log(Z) - gammalog(W+1)
 		infoValue = (W*logZ - lgamma(W+1))/log(2.0);

		// Second calculate term2
		// Here we have the same form as term1 except replace the Z and W as above
		// newZ = Z*pF - W*pF + W*pL
		// Since Z >> W, we have: newLogZ ~ max(logZ + log(pF), log(W) + log(pL)
		newLogZ = fmax(logZ + log(Pf), log((double)W) + log(Pl));
		// newW = W*pL
		newW = W*Pl;
		// Finish calculation of term2
		infoValue = infoValue - (newW*newLogZ - lgamma(newW+1))/log(2.0);

		// Lastly calculate term3
		// Here we have the same form as term1 except replace the Z and W as above
		// newZ = Z*(1-pF) + W*pF - W*pL
		// newLogZ ~ max(logZ + log(1-pF), log(W) + log(pF - pL)
		newLogZ = fmax(logZ + log(1.0-Pf), log((double)W) + (Pl < Pf ? log(Pf-Pl) : -1.0e20));
		// newW = W*(1-pL)
		newW = W*(1-Pl);
		// Finish calculation of term3
		infoValue = infoValue - (newW*newLogZ - lgamma(newW+1))/log(2.0);

		// return the calculated information
		return infoValue;
	}

	double I_M(double H, double Q, int R, int C, int D)
	// Z = total number of possible words
	// W = number of words Alice tried to teach
	// Wl = number of words learned by net
	// Pt = probability of successfully training a word (Wl / W)
	// PF = probability of a false positive
	// D = number of delay slots
	{
		double Z = pow(2.0, (double)(C * H * Q * R));
		double W = true_true + true_false;
		double Wl = true_true;
		double Pt = true_true / (true_true + true_false);
		double PF = false_true / (false_true + false_false);
		double i_m = (lchoose(Z * Pt, Wl) - lchoose((Z - W) * Pt * PF + Wl, (Z - W) * Pt * PF)) / log(2.0);

		// printf("Z = %f, W = %f, Wl = %f, Pt = %f, PF = %f, D = %d, i_m = %f\n",
		//	Z, W, Wl, Pt, PF, D, i_m);
		fflush(stdout);
		if (1 < D) {
			double Ild = 0.0;
			i_m += Wl * log2(D) - Ild;
		}
		return i_m;
	}

	double spn(int C, double H, double Q, int R)
	{
		return (C * H * Q * R);
	}
};
}  // namespace
#endif  // COGNON_ORIG_H
