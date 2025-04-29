include $(FLUTEMK)
include $(TESTMK)

MODE ?= sim
HW = $(if $(filter $(MODE),sim),0,1)

#WIDTH = 3
#HEIGHT = 2


SIMFLAGS = cycles=10000000 $(SIMOPTS) threads=8
PAINTVARS = N P NX NY IORATX IORATY FLIP ORIGIN
PAINTFLAGS = lib=microcode lib=.. lib=waxpy lib=app/md $(PAINTOPTS) verbose toolflag w "liveness" arch=$(ARCH) threads=8

PIPELINEFLAGS=arch=$(ARCH)

NXTOT = $(shell expr $(NX) \* $(NZX) )
NYTOT = $(shell expr $(NY) \* $(NZY) )

porder = $(shell grep 'core00.params ' bench.map | sed -e 's/  //g' -e 's/^.*\[\|\].*//g' -e 's/<[0-9]*//g')

ifeq ($(TEST),t0)
  # If running the t0 test, remove 'y'
  worder = "x n m _word"
  order = "x n m"
  oorder = "x n n2"
else
  worder = "y x n m _word"
  order = "y x n m"
  oorder = "y x n n2"
endif
all : check
test : test/$(MODE)
clean: cleanup

test/$(MODE): bench.obj checkbins #out-$(MODE).bin #out-ref.bin

init:
	echo hello, nxtot=$(NXTOT), origin=$(ORIGIN)

generate_bins: init
	python ../make-inputbin.py $(NX) $(NY) $(N) $(P) params.bin x0.bin $(ORIGIN)

checkbins: out-$(MODE).bin
	python ../check-inoutbin.py $(NX) $(NY) $(N) $(P) {params,x0,x,y}.bin

bench.img: bench.obj generate_bins
	cp $< $@
	#miset image=$@ out=$@ map=bench.map symbol=core00.x order=$(worder) data=x0.bin.00
	#miset image=$@ out=$@ map=bench.map symbol=core01.x order=$(worder) data=x0.bin.01
	#miset image=$@ out=$@ map=bench.map symbol=core10.x order=$(worder) data=x0.bin.10
	#miset image=$@ out=$@ map=bench.map symbol=core11.x order=$(worder) data=x0.bin.11
	#
	miset image=$@ out=$@ map=bench.map symbol=core00.params order="$(porder)" data=params.bin
	miset image=$@ out=$@ map=bench.map symbol=core01.params order="$(porder)" data=params.bin
	miset image=$@ out=$@ map=bench.map symbol=core10.params order="$(porder)" data=params.bin
	miset image=$@ out=$@ map=bench.map symbol=core11.params order="$(porder)" data=params.bin

out-sim.bin: bench-sim.core bench.map
	miget image=bench-sim.core map=bench.map symbol=core00.tstamp order=$(order) data=tstamp.bin
	miget image=bench-sim.core map=bench.map symbol=core00.x order=$(worder) data=x.bin.00
	miget image=bench-sim.core map=bench.map symbol=core01.x order=$(worder) data=x.bin.01
	miget image=bench-sim.core map=bench.map symbol=core10.x order=$(worder) data=x.bin.10
	miget image=bench-sim.core map=bench.map symbol=core11.x order=$(worder) data=x.bin.11
	miget image=bench-sim.core map=bench.map symbol=core00.y order=$(worder) data=y.bin.00
	miget image=bench-sim.core map=bench.map symbol=core01.y order=$(worder) data=y.bin.01
	miget image=bench-sim.core map=bench.map symbol=core10.y order=$(worder) data=y.bin.10
	miget image=bench-sim.core map=bench.map symbol=core11.y order=$(worder) data=y.bin.11
	date > $@

.PRECIOUS: bench-sim.log bench-sim.core
