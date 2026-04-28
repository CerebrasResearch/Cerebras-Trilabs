include $(FLUTEMK)
include $(TESTMK)

ORIGORDER = 1
VERBOSE = 0
INTINIT = 0
NTSC = 2

NDOT ?= 3

PAINTVARS=H V Z NDOT

SIMFLAGS = cycles=800000 $(SIMOPTS)
PAINTFLAGS=lib=microcode lib=.. $(PAINTOPTS) verbose arch=sdr toolflag w "recharge verbose ra_max_attempts=20" threads=8
PIPELINEFLAGS=arch=sdr

MODE ?= sim

all: test
clean: cleanup
test: test/$(MODE)

test/$(MODE): out-$(MODE).bin
	fcmp a=dot-out-$(MODE).bin b=dot_out.bin aformat=sp bformat=sp report=1 limit=0.01

bench.img: bench.obj bench.map
	cp $< $@
	cp ../hpcg_ref1.c .
	gcc -o hpcg_ref1 -g3 hpcg_ref1.c -lm
	cp ../hpcg_router1.c .
	gcc -std=c99 -o hpcg_router1 -g3 hpcg_router1.c
	./hpcg_router1 NX=$H NY=$V mode=dot
	./hpcg_ref1 1 1 1 $H $V $Z 1 --test dot --init fixed --perm perm1 --levels 1 --bin_split
	miset image=$@ out=$@ map=bench.map symbol=a data=a0.bin order="y, x, z, _word";
	miset image=$@ out=$@ map=bench.map symbol=b data=b0.bin order="y, x, z, _word";
	miset image=$@ out=$@ map=bench.map symbol=router_dot data=router_dot.bin order="y, x, n0"
	miset image=$@ out=$@ map=bench.map symbol=switch_dot data=switch_dot.bin order="y, x, n"
	miset image=$@ out=$@ map=bench.map symbol=router_forward data=router_dot_edge.bin order="y, n0"
	miset image=$@ out=$@ map=bench.map symbol=router_recolor data=router_dot_recolor.bin order="n0"
	miset image=$@ out=$@ map=bench.map symbol=router_src_left_content data=router_dot_src_left.bin order="y, n0"
	miset image=$@ out=$@ map=bench.map symbol=router_src_top_content data=router_dot_src_top.bin order="n0"

# simulation target
out-sim.bin: bench-sim.core bench.map
	miget image=$< map=bench.map symbol=dot_res data=dot-out-$(MODE).bin;
	miget image=$< map=bench.map symbol=tstamp data=tstamp.bin
	python3 ../read_tstamp.py tstamp.bin $H $V $(NDOT)

.PRECIOUS: bench.elf-core
