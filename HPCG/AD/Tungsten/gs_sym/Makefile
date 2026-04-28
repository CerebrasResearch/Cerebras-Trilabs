include $(FLUTEMK)
include $(TESTMK)

NTSC = 2

L ?= 0
C ?= 2 
P ?= 1

LEVELS ?= $$(( $(L)+1 ))

NGS ?= $$(($(H)>$(V) ? $(V) : $(H)))

ZL = $$(($Z>>$L))
ZP = $$(($(ZL)/$P))

NRX = $$(( $(H)/$(NGS) ))
NRY = $$(( $(V)/$(NGS) ))
NRZ = $P
nx = $(NGS)
ny = $(NGS)
nz = $$(($(Z)/$P))

ZL1 = $$(($(ZL)<<1))
Z2 = $$(($(Z)<<1))

NNZ = $$(( $(Z2) - $(ZL) ))
BASE = $$(( $(Z2) - $(ZL1) ))
NNZP = $$(( $(NNZ) + (2*$(LEVELS)) ))
BASEP = $$(( $(BASE) + (2*$(L)) ))

NDOT = 1

PAINTVARS=H V Z ZL C P ZP NNZ BASE NNZP BASEP NDOT


# system parameters
x0 ?= 3
y0 ?= 3
xlim ?= $$(($H+2)))
ylim ?= $$(($V+2)))
TARGET = sc-r8rb15-s1
REMOTETMP = tmpgs
PFABRIC = pfabric.bin
VFABRIC = vfabric.bin
CHUNKSZ = 32
SLEEP = 10

SIMFLAGS = cycles=250000 $(SIMOPTS)
#SIMFABRIC=/net/miladh-dev/srv/nfs/miladh-data/ws/monolith/technology/sysroot-x86/bin/simfabric
PAINTFLAGS=lib=microcode lib=.. $(PAINTOPTS) verbose arch=sdr toolflag w "recharge verbose" threads=8
PIPELINEFLAGS=arch=sdr

MODE ?= sim

all: test
clean: cleanup
test: test/$(MODE)

test/$(MODE): out-$(MODE).bin 
	fcmp a=x-out-$(MODE).bin b=x_out.bin aformat=sp bformat=sp report=1

bench.img: bench.obj bench.map
	cp $< $@
	cp ../../../C/hpcg_ref1.c .
	gcc -o hpcg_ref1 -g3 hpcg_ref1.c -lm
	cp ../../../C/hpcg_router1.c .
	gcc -std=c99 -o hpcg_router1 -g3 hpcg_router1.c 
	./hpcg_router1 NX=$H NY=$V NGS=$(NGS) mode=sym level=$L out_split=2
	./hpcg_ref1 $(NRX) $(NRY) $(NRZ) $(nx) $(ny) $(nz) --test gs_sym --init random --perm perm2 --levels $(LEVELS) --test-level $L
	miset image=$@ out=$@ map=bench.map symbol=router_stencil data=router_stencil.bin order="y, x, n0, n1"
	miset image=$@ out=$@ map=bench.map symbol=active_stencil data=active_stencil.bin order="y, x, n0"
	miset image=$@ out=$@ map=bench.map symbol=boundary_stencil data=boundary_stencil.bin order="y, x, n0, n1"
	miset image=$@ out=$@ map=bench.map symbol=colors_stencil data=colors_stencil.bin order="y, x, n0, n1"
	miset image=$@ out=$@ map=bench.map symbol=x data=x.bin order="y, x, z, _word";\
	miset image=$@ out=$@ map=bench.map symbol=r data=r.bin order="y, x, z, _word";\
	miset image=$@ out=$@ map=bench.map symbol=A data=A.bin order="y, x, z, _word";\
	miset image=$@ out=$@ map=bench.map symbol=invd data=invdiag.bin order="y, x, z, _word";\
	miset image=$@ out=$@ map=bench.map symbol=router_dot data=router_dot.bin order="y, x, n0"
	miset image=$@ out=$@ map=bench.map symbol=switch_dot data=switch_dot.bin order="y, x, n"
	miset image=$@ out=$@ map=bench.map symbol=router_forward data=router_dot_edge.bin order="y, n0"
	miset image=$@ out=$@ map=bench.map symbol=router_recolor data=router_dot_recolor.bin order="n0"
	miset image=$@ out=$@ map=bench.map symbol=router_src_left_content data=router_dot_src_left.bin order="y, n0"
	miset image=$@ out=$@ map=bench.map symbol=router_src_top_content data=router_dot_src_top.bin order="n0"

# simulation target
out-sim.bin: bench-sim.core bench.map
	# read result data
	miget image=$< map=bench.map symbol=x data=x-out-$(MODE).bin;
	miget image=bench-sim.core map=bench.map symbol=tstamp data=tstamp.bin
	python ../read_tstamp_file.py tstamp.bin $H $V $Z $L


# hardware target
out-hw.bin: bench.img bench.map
	# create temp remote directory and copy local files
	ssh $(TARGET) rm -rf $(REMOTETMP)
	ssh $(TARGET) mkdir $(REMOTETMP)
	scp * $(TARGET):$(REMOTETMP) &>> remote-log

	# wipe, program virtual fabric size/IO streams, program image
	ssh $(TARGET) cswipe fabric=$(PFABRIC) &>> remote-log
	ssh $(TARGET) csvfabric pfabric=$(PFABRIC) vfabric=$(VFABRIC) \
			x0=$(x0) y0=$(y0) w=$(xlim) h=$(ylim) \
			&>> remote-log
	ssh $(TARGET) csprog image=$(REMOTETMP)/bench.img &>> remote-log

	# NOTE: debug
	ssh $(TARGET) csdump image=$(REMOTETMP)/preckpt.img &>> remote-log

	# allow PEs to start processing data
	ssh $(TARGET) cscontrol run &>> remote-log

	# wait for computation to finish
	ssh $(TARGET) sleep $(SLEEP) &>> remote-log
	
	# pause PEs and copy artifacts back to local
	ssh $(TARGET) cscontrol pause &>> remote-log

	# read symbols from hw
	ssh $(TARGET) csget map=$(REMOTETMP)/bench.map symbol=x  data=$(REMOTETMP)/x-out-$(MODE).bin &>> remote-log

	#ssh $(TARGET) csget map=$(REMOTETMP)/bench.map symbol=tscmem  data=$(REMOTETMP)/tsc-out-$(MODE).bin &>> remote-log

	# NOTE: debug
	ssh $(TARGET) csdump image=$(REMOTETMP)/ckpt.img &>> remote-log

	# copy all files back to dev server
	scp $(TARGET):$(REMOTETMP)/* . &>> remote-log

.PRECIOUS: bench.elf-core
