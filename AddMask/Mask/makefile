CPP = clang++
CFLAGs = -c -fPIC -O2 -std=c++11 $(shell pkg-config --cflags opencv)
LFLAGs = $(shell pkg-config --libs opencv) -shared # -ltcmalloc
# CFLAGs = -c -g -std=c++11 $(shell pkg-config --cflags opencv) -DMAIN
# LFLAGs = $(shell pkg-config --libs opencv) -ltcmalloc -pthread
ODIR = ./build
LIBDIR = ./lib

SRCs = $(wildcard *.cpp)
OBJs_CURR = $(patsubst %.cpp, %.o, $(SRCs))
OBJs = $(patsubst %.cpp, $(ODIR)/%.o, $(SRCs))


all: libmossaic.so libmossaic.a


libmossaic.so: $(OBJs_CURR)
	@mkdir -p $(LIBDIR)
	$(CPP) $(OBJs) -o $(LIBDIR)/$@ $(LFLAGs)


libmossaic.a: $(OBJs_CURR)
	@mkdir -p $(LIBDIR)
	ar rsc $(LIBDIR)/$@ $(OBJs)

main: $(OBJs_CURR)
	$(CPP) $(OBJs) -o $@ $(LFLAGs)


$(OBJs_CURR): %.o: %.cpp
	@mkdir -p $(ODIR)
	$(CPP) $(CFLAGs) $^ -o $(ODIR)/$@


clean:
	rm -rf try main $(ODIR) $(LIBDIR)

