CC = clang-11
CFLAGS = -Wall -O3 -mavx2 -mavx -msse4.2 -msse4.1 -msse4 -msse3 -mssse3 -msse2 -msse
LDFLAGS = -lm

ifeq ("$(ENABLE_OPENMP)","1")
CFLAGS += -DOPT_PLANE -fopenmp
LDFLAGS += -fopenmp -lpthread
endif


all: main

THIRD_PARTIES = stb_image.h stb_image_write.h
stb_image.h:
	wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
stb_image_write.h:
	wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

OBJS = main.o
deps := $(OBJS:%.o=%.o.d)

%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c -MMD -MF $@.d $<
main: $(THIRD_PARTIES) $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

check: main
	$(RM) out.jpg
	./main jserv.jpg

clean:
	$(RM) main $(OBJS) $(deps)
	$(RM) out.jpg

distclean: clean
	$(RM) $(THIRD_PARTIES)

-include $(deps)
