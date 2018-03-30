CPP = clang++
CFLAGs = -std=c++11 -c -g $(shell pkg-config --cflags opencv)
LFLAGs = -ltcmalloc $(shell pkg-config --libs opencv)
ODIR = ./build

SRCs = $(wildcard *.cpp)
OBJs_CURR = $(patsubst %.cpp, %.o, $(SRCs))
OBJs = $(patsubst %.cpp, $(ODIR)/%.o, $(SRCs))


main: $(OBJs_CURR)
	$(CPP) $(OBJs) -o $@ $(LFLAGs)


$(OBJs_CURR): %.o: %.cpp
	@mkdir -p $(ODIR)
	$(CPP) $(CFLAGs) $^ -o $(ODIR)/$@


clean:
	rm -rf try main $(ODIR)

