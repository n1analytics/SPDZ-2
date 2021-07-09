
include CONFIG

MATH = $(patsubst %.cpp,%.o,$(wildcard Math/*.cpp))

TOOLS = $(patsubst %.cpp,%.o,$(wildcard Tools/*.cpp))

NETWORK = $(patsubst %.cpp,%.o,$(wildcard Networking/*.cpp))

PROCESSOR = $(patsubst %.cpp,%.o,$(wildcard Processor/*.cpp))

FHEOFFLINE = $(patsubst %.cpp,%.o,$(wildcard FHEOffline/*.cpp FHE/*.cpp)) Protocols/CowGearOptions.o

GC = $(patsubst %.cpp,%.o,$(wildcard GC/*.cpp)) $(PROCESSOR)
GC_SEMI = GC/SemiSecret.o GC/SemiPrep.o GC/square64.o

OT = $(patsubst %.cpp,%.o,$(wildcard OT/*.cpp))
OT_EXE = ot.x ot-offline.x

COMMON = $(MATH) $(TOOLS) $(NETWORK) GC/square64.o Processor/OnlineOptions.o Processor/BaseMachine.o Processor/DataPositions.o Processor/ThreadQueues.o Processor/ThreadQueue.o
COMPLETE = $(COMMON) $(PROCESSOR) $(FHEOFFLINE) $(TINYOTOFFLINE) $(GC) $(OT)
YAO = $(patsubst %.cpp,%.o,$(wildcard Yao/*.cpp)) $(OT) BMR/Key.o
BMR = $(patsubst %.cpp,%.o,$(wildcard BMR/*.cpp BMR/network/*.cpp))
VM = $(PROCESSOR) $(COMMON) GC/square64.o GC/Instruction.o OT/OTTripleSetup.o OT/BaseOT.o $(LIBSIMPLEOT)


LIB = libSPDZ.a
LIBRELEASE = librelease.a

ifeq ($(AVX_OT), 0)
LIBSIMPLEOT = ECDSA/P256Element.o
else
LIBSIMPLEOT = SimpleOT/libsimpleot.a
endif

# used for dependency generation
OBJS = $(BMR) $(FHEOFFLINE) $(TINYOTOFFLINE) $(YAO) $(COMPLETE) $(patsubst %.cpp,%.o,$(wildcard Machines/*.cpp Utils/*.cpp))
DEPS := $(wildcard */*.d */*/*.d)

# never delete
.SECONDARY: $(OBJS)


all: arithmetic binary gen_input online offline externalIO bmr ecdsa doc

.PHONY: doc
doc:
	cd doc; $(MAKE) html

arithmetic: rep-ring rep-field shamir semi2k-party.x semi-party.x mascot sy
binary: rep-bin yao semi-bin-party.x tinier-party.x tiny-party.x ccd-party.x malicious-ccd-party.x real-bmr

ifeq ($(USE_NTL),1)
all: overdrive she-offline
arithmetic: hemi-party.x soho-party.x gear
endif

-include $(DEPS)
include $(wildcard *.d static/*.d)

%.o: %.cpp
	$(CXX) -o $@ $< $(CFLAGS) -MMD -MP -c

online: Fake-Offline.x Server.x Player-Online.x Check-Offline.x emulate.x

offline: $(OT_EXE) Check-Offline.x mascot-offline.x cowgear-offline.x mal-shamir-offline.x

gen_input: gen_input_f2n.x gen_input_fp.x

externalIO: bankers-bonus-client.x

bmr: bmr-program-party.x bmr-program-tparty.x

real-bmr: $(patsubst Machines/%.cpp,%.x,$(wildcard Machines/*-bmr-party.cpp))

yao: yao-party.x

she-offline: Check-Offline.x spdz2-offline.x

overdrive: simple-offline.x pairwise-offline.x cnc-offline.x gear
gear: cowgear-party.x chaigear-party.x lowgear-party.x highgear-party.x

rep-field: malicious-rep-field-party.x replicated-field-party.x ps-rep-field-party.x

rep-ring: replicated-ring-party.x brain-party.x malicious-rep-ring-party.x ps-rep-ring-party.x rep4-ring-party.x

rep-bin: replicated-bin-party.x malicious-rep-bin-party.x ps-rep-bin-party.x Fake-Offline.x

replicated: rep-field rep-ring rep-bin

spdz2k: spdz2k-party.x ot-offline.x Check-Offline-Z2k.x galois-degree.x Fake-Offline.x
mascot: mascot-party.x spdz2k mama-party.x

tldr:
	-echo ARCH = -march=native >> CONFIG.mine
	$(MAKE) mascot-party.x

ifeq ($(OS), Darwin)
tldr: mac-setup
else
tldr: mpir
endif

ifeq ($(MACHINE), aarch64)
tldr: simde/simde
endif

shamir: shamir-party.x malicious-shamir-party.x galois-degree.x

sy: sy-rep-field-party.x sy-rep-ring-party.x sy-shamir-party.x

ecdsa: $(patsubst ECDSA/%.cpp,%.x,$(wildcard ECDSA/*-ecdsa-party.cpp)) Fake-ECDSA.x
ecdsa-static: static-dir $(patsubst ECDSA/%.cpp,static/%.x,$(wildcard ECDSA/*-ecdsa-party.cpp))

$(LIBRELEASE): Protocols/MalRepRingOptions.o $(PROCESSOR) $(COMMON) $(OT) $(GC)
	$(AR) -csr $@ $^

static/%.x: Machines/%.o $(LIBRELEASE) $(LIBSIMPLEOT)
	$(CXX) $(CFLAGS) -o $@ $^ -Wl,-Map=$<.map -Wl,-Bstatic -static-libgcc -static-libstdc++  $(LIBRELEASE) $(LIBSIMPLEOT) $(BOOST) $(LDLIBS) -Wl,-Bdynamic -ldl

static/%.x: ECDSA/%.o ECDSA/P256Element.o $(VM) $(OT) $(LIBSIMPLEOT)
	$(CXX) $(CFLAGS) -o $@ $^ -Wl,-Map=$<.map -Wl,-Bstatic -static-libgcc -static-libstdc++ $(BOOST) $(LDLIBS) -Wl,-Bdynamic -ldl

static-dir:
	@ mkdir static 2> /dev/null; true

static-release: static-dir $(patsubst Machines/%.cpp, static/%.x, $(wildcard Machines/*-party.cpp)) static/emulate.x

Fake-ECDSA.x: ECDSA/Fake-ECDSA.cpp ECDSA/P256Element.o $(COMMON) Processor/PrepBase.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LDLIBS)

Check-Offline.x: $(PROCESSOR)

ot.x: $(OT) $(COMMON) Machines/OText_main.o Machines/OTMachine.o $(LIBSIMPLEOT)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDLIBS)

ot-offline.x: $(OT) $(LIBSIMPLEOT) Machines/TripleMachine.o

gc-emulate.x: $(VM) GC/FakeSecret.o GC/square64.o

bmr-%.x: $(BMR) $(VM) Machines/bmr-%.cpp $(LIBSIMPLEOT)
	$(CXX) -o $@ $(CFLAGS) $^ $(BOOST) $(LDLIBS)

%-bmr-party.x: Machines/%-bmr-party.o $(BMR) $(VM) $(LIBSIMPLEOT)
	$(CXX) -o $@ $(CFLAGS) $^ $(BOOST) $(LDLIBS)

bmr-clean:
	-rm BMR/*.o BMR/*/*.o GC/*.o

bankers-bonus-client.x: ExternalIO/bankers-bonus-client.cpp $(COMMON)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDLIBS)

simple-offline.x: $(FHEOFFLINE)
pairwise-offline.x: $(FHEOFFLINE)
cnc-offline.x: $(FHEOFFLINE)
spdz2-offline.x: $(FHEOFFLINE)

yao-party.x: $(YAO)
static/yao-party.x: $(YAO)

yao-clean:
	-rm Yao/*.o

galois-degree.x: Utils/galois-degree.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDLIBS)

default-prime-length.x: Utils/default-prime-length.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDLIBS)

secure.x: Utils/secure.o
	$(CXX) -o $@ $(CFLAGS) $^

%.x: Utils/%.o $(COMMON)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDLIBS)

%.x: Machines/%.o $(VM) OT/OTTripleSetup.o OT/BaseOT.o $(LIBSIMPLEOT)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDLIBS)

%gear-party.x: Machines/%gear-party.o $(VM) OT/OTTripleSetup.o OT/BaseOT.o $(LIBSIMPLEOT)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDLIBS) -lntl

hemi-party.x: Machines/hemi-party.o $(VM)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDLIBS) -lntl

soho-party.x: Machines/soho-party.o $(VM)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDLIBS) -lntl

%-ecdsa-party.x: ECDSA/%-ecdsa-party.o ECDSA/P256Element.o $(VM)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDLIBS)

replicated-bin-party.x: GC/square64.o
replicated-ring-party.x: GC/square64.o
replicated-field-party.x: GC/square64.o
brain-party.x: GC/square64.o
malicious-rep-bin-party.x: GC/square64.o
ps-rep-bin-party.x: GC/PostSacriBin.o
semi-bin-party.x: $(VM) $(OT) GC/SemiSecret.o GC/SemiPrep.o GC/square64.o
tiny-party.x: $(OT)
tinier-party.x: $(OT)
spdz2k-party.x: $(OT) $(patsubst %.cpp,%.o,$(wildcard Machines/SPDZ2*.cpp))
static/spdz2k-party.x: $(patsubst %.cpp,%.o,$(wildcard Machines/SPDZ2*.cpp))
semi-party.x: $(OT) GC/SemiSecret.o GC/SemiPrep.o GC/square64.o
semi2k-party.x: $(OT) GC/SemiSecret.o GC/SemiPrep.o GC/square64.o
hemi-party.x: $(FHEOFFLINE) $(GC_SEMI) $(OT)
soho-party.x: $(FHEOFFLINE) $(GC_SEMI) $(OT)
cowgear-party.x: $(FHEOFFLINE) Protocols/CowGearOptions.o $(OT)
chaigear-party.x: $(FHEOFFLINE) Protocols/CowGearOptions.o $(OT)
lowgear-party.x: $(FHEOFFLINE) $(OT) Protocols/CowGearOptions.o Protocols/LowGearKeyGen.o
highgear-party.x: $(FHEOFFLINE) $(OT) Protocols/CowGearOptions.o Protocols/HighGearKeyGen.o
static/hemi-party.x: $(FHEOFFLINE)
static/soho-party.x: $(FHEOFFLINE)
static/cowgear-party.x: $(FHEOFFLINE)
static/chaigear-party.x: $(FHEOFFLINE)
static/lowgear-party.x: $(FHEOFFLINE) Protocols/CowGearOptions.o Protocols/LowGearKeyGen.o
static/highgear-party.x: $(FHEOFFLINE) Protocols/CowGearOptions.o Protocols/HighGearKeyGen.o
mascot-party.x: Machines/SPDZ.o $(OT)
static/mascot-party.x: Machines/SPDZ.o
Player-Online.x: Machines/SPDZ.o $(OT)
mama-party.x: $(OT)
ps-rep-ring-party.x: Protocols/MalRepRingOptions.o
malicious-rep-ring-party.x: Protocols/MalRepRingOptions.o
sy-rep-ring-party.x: Protocols/MalRepRingOptions.o
rep4-ring-party.x: GC/Rep4Secret.o
no-party.x: Protocols/ShareInterface.o
semi-ecdsa-party.x: $(OT) $(LIBSIMPLEOT) GC/SemiPrep.o GC/SemiSecret.o
mascot-ecdsa-party.x: $(OT) $(LIBSIMPLEOT)
fake-spdz-ecdsa-party.x: $(OT) $(LIBSIMPLEOT)
emulate.x: GC/FakeSecret.o
semi-bmr-party.x: GC/SemiPrep.o GC/SemiSecret.o $(OT)
real-bmr-party.x: $(OT)
paper-example.x: $(VM) $(OT) $(FHEOFFLINE)
mascot-offline.x: $(VM) $(OT)
cowgear-offline.x: $(OT) $(FHEOFFLINE)
Fake-Offline.x: $(VM)
static/rep-bmr-party.x: $(BMR)
static/mal-rep-bmr-party.x: $(BMR)
static/shamir-bmr-party.x: $(BMR)
static/mal-shamir-bmr-party.x: $(BMR)
static/semi-bmr-party.x: $(BMR)
static/real-bmr-party.x: $(BMR)
static/bmr-program-party.x: $(BMR)
static/no-party.x: Protocols/ShareInterface.o

ifeq ($(AVX_OT), 1)
$(LIBSIMPLEOT): SimpleOT/Makefile
	$(MAKE) -C SimpleOT

OT/BaseOT.o: SimpleOT/Makefile

SimpleOT/Makefile:
	git submodule update --init SimpleOT
endif

.PHONY: Programs/Circuits
Programs/Circuits:
	git submodule update --init Programs/Circuits

.PHONY: mpir-setup mpir-global mpir
mpir-setup:
	git submodule update --init mpir
	cd mpir; \
	autoreconf -i
	- $(MAKE) -C mpir clean

mpir-global: mpir-setup
	cd mpir; \
	./configure --enable-cxx;
	$(MAKE) -C mpir
	sudo $(MAKE) -C mpir install

mpir: mpir-setup
	cd mpir; \
	./configure --enable-cxx --prefix=$(CURDIR)/local
	$(MAKE) -C mpir install
	-echo MY_CFLAGS += -I./local/include >> CONFIG.mine
	-echo MY_LDLIBS += -Wl,-rpath -Wl,./local/lib -L./local/lib >> CONFIG.mine

mac-setup:
	brew install openssl boost libsodium mpir yasm ntl
	-echo MY_CFLAGS += -I/usr/local/opt/openssl/include >> CONFIG.mine
	-echo MY_LDLIBS += -L/usr/local/opt/openssl/lib >> CONFIG.mine
	-echo USE_NTL = 1 >> CONFIG.mine

simde/simde:
	git submodule update --init simde

clean:
	-rm -f */*.o *.o */*.d *.d *.x core.* *.a gmon.out */*/*.o static/*.x
