CXX=c++
CFLAGS=-fopenmp -O3 # -mtune=i686 -march=i686 -msse3 #-g

VERSION=1.0-rc4

HDRS=\
	alice.h \
	bob.h \
	cognon.h \
	cognon-orig.h \
	compat.h \
	monograph.h \
	mtrand.h \
	neuron.h \
	wordset.h

SRCS=\
	alice.cc \
	bob.cc \
	cognon.cc \
	compat.cc \
	monograph.cc \
	neuron.cc \
	wordset.cc

MAINS=\
	alice_test.cc \
	bob_test.cc \
	cognon_main.cc \
	cognon_stats.cc \
	cognon_test.cc \
	graph-2.1.cc \
	graph-2.2.cc \
	graph-2.3.cc \
	neuron_test.cc \
	table-2.1.cc \
	table-2.3.cc \
	table-2.4.cc \
	table-3.3.cc \
	wordset_test.cc

SAMPLES=\
	sample-input.spec \
	sample-optimize.spec

SPECS=\
	table-2.1.spec \
	table-2.3.spec \
	table-2.4.spec \
	table-3.2.spec \
	table-3.3.spec \
	table-3.4.spec

GRAPHS=\
	graph-2.1.data \
	graph-2.2.data \
	graph-2.3.data \
	predict-2.1.data \
	predict-2.3.data

SEARCHES=\
	search-table-2.1.csv \
	search-table-2.3.csv \
	search-table-2.4.csv \
	search-table-3.3.csv

TABLES=\
	table-2.1.csv \
	table-2.3.csv \
	table-2.4.csv \
	table-3.2.csv \
	table-3.3.csv \
	table-3.4.csv

TESTS=\
	alice_test \
	bob_test \
	cognon_test \
	neuron_test \
	wordset_test

all: tests cognon # tables graphs

graphs: ${GRAPHS}
searches: ${SEARCHES}
tables: ${TABLES}
tests : ${TESTS}
	for t in ${TESTS}; do nice ./$$t; done

cognon: $(HDRS) $(SRCS) cognon_main.cc
	$(CXX) $(CFLAGS) -o cognon $(SRCS) cognon_main.cc -lm

cognon_stats: cognon_stats.cc
	$(CXX) $(CFLAGS) -o cognon_stats cognon_stats.cc -lm

.SUFFIXES: .spec .csv

.spec.csv: cognon
	./cognon $< > $@

predict-2.1.data : cognon_stats
	./$< -m 1 > $@
predict-2.3.data : cognon_stats
	./$< -m 3 > $@

graph-2.1.data : graph-2.1
	./$< > $@
graph-2.1 : $(HDRS) $(SRCS) graph-2.1.cc
	$(CXX) $(CFLAGS) -o graph-2.1 $(SRCS) graph-2.1.cc -lm

graph-2.2.data : graph-2.2
	./$< > $@
graph-2.2 : $(HDRS) $(SRCS) graph-2.2.cc
	$(CXX) $(CFLAGS) -o graph-2.2 $(SRCS) graph-2.2.cc -lm

graph-2.3.data : graph-2.3
	./$< > $@
graph-2.3 : $(HDRS) $(SRCS) graph-2.3.cc
	$(CXX) $(CFLAGS) -o graph-2.3 $(SRCS) graph-2.3.cc -lm

search-table-2.1.csv : table-2.1
	g="$<-`date +%Y%m%d_%H:%M`.csv"; \
	nice ./$< > "$${g}"; \
	fgrep -v '#' "$${g}" > $@
table-2.1: $(HDRS) $(SRCS) table-2.1.cc
	$(CXX) $(CFLAGS) -o table-2.1 $(SRCS) table-2.1.cc -lm

search-table-2.3.csv : table-2.3
	g="$<-`date +%Y%m%d_%H:%M`.csv"; \
	nice ./$< > "$${g}"; \
	fgrep -v '#' "$${g}" > $@
table-2.3: $(HDRS) $(SRCS) table-2.3.cc
	$(CXX) $(CFLAGS) -o table-2.3 $(SRCS) table-2.3.cc -lm

search-table-2.4.csv : table-2.4
	g="$<-`date +%Y%m%d_%H:%M`.csv"; \
	nice ./$< > "$${g}"; \
	fgrep -v '#' "$${g}" > $@
table-2.4: $(HDRS) $(SRCS) table-2.4.cc
	$(CXX) $(CFLAGS) -o table-2.4 $(SRCS) table-2.4.cc -lm

search-table-3.3.csv : table-3.3
	g="$<-`date +%Y%m%d_%H:%M`.csv"; \
	nice ./$< > "$${g}"; \
	fgrep -v '#' "$${g}" > $@
table-3.3: $(HDRS) $(SRCS) table-3.3.cc
	$(CXX) $(CFLAGS) -o table-3.3 $(SRCS) table-3.3.cc -lm

alice_test: $(HDRS) $(SRCS) alice_test.cc
	$(CXX) $(CFLAGS) -o alice_test $(SRCS) alice_test.cc -lm

bob_test: $(HDRS) $(SRCS) bob_test.cc
	$(CXX) $(CFLAGS) -o bob_test $(SRCS) bob_test.cc -lm

cognon_test: $(HDRS) $(SRCS) cognon_test.cc
	$(CXX) $(CFLAGS) -o cognon_test $(SRCS) cognon_test.cc -lm

neuron_test: $(HDRS) $(SRCS) cognon-orig.h neuron_test.cc
	$(CXX) $(CFLAGS) -o neuron_test $(SRCS) neuron_test.cc -lm

wordset_test: $(HDRS) $(SRCS) wordset_test.cc
	$(CXX) $(CFLAGS) -o wordset_test $(SRCS) wordset_test.cc -lm

.PHONY: dist snapshot
dist : Makefile README $(HDRS) $(SRCS) $(MAINS) $(SAMPLES) $(SPECS)
	rm -rf /tmp/cognon-$(VERSION)
	mkdir /tmp/cognon-$(VERSION)
	cp README COPYING Makefile \
		$(HDRS) $(SRCS) $(MAINS) $(SAMPLES) $(SPECS) \
		/tmp/cognon-$(VERSION)
	(cd /tmp; zip -r cognon-$(VERSION).zip . -i cognon-$(VERSION)/\*)
	(cd /tmp; tar czf cognon-$(VERSION).tgz cognon-$(VERSION))
	mv /tmp/cognon-$(VERSION).zip /tmp/cognon-$(VERSION).tgz .
	rm -rf /tmp/cognon-$(VERSION)

snapshot: Makefile README $(HDRS) $(SRCS) $(MAINS) $(SAMPLES) $(TABLES)
	rm -f cognon.zip
	zip cognon-`date +%Y%m%d_%H:%M`.zip \
		README COPYING Makefile \
		$(HDRS) $(SRCS) $(MAINS) $(SAMPLES) $(TABLES) $(GRAPHS)
	tar czf xxx.tgz \
		README COPYING Makefile \
		$(HDRS) $(SRCS) $(MAINS) $(SAMPLES) $(TABLES) $(GRAPHS)
	mv xxx.tgz ./"cognon-`date +%Y%m%d_%H:%M`".tgz

clean:
	rm -f *~ cognon cognon_stat
	rm -f alice_test cognon_test neuron_test wordset_test
	rm -f table-2.1 table-2.3 table-2.4 table-3.3

realclean: clean
	rm -f ${TABLES}
	rm -f ${GRAPHS}
	rm -f ${SEARCHES}


