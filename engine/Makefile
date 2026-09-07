# engine/Makefile - Sisyphus engine pieces. Step 2: the streamer.
#   make -C engine            # builds engine/streamer
#   sudo apt install liburing-dev   (Ubuntu) if liburing.h is missing
CC      ?= gcc
CFLAGS  := -O2 -Wall -Wextra -std=gnu11 -pthread $(EXTRA_CFLAGS)
LDLIBS  := -luring -pthread

all: streamer

streamer: streamer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f streamer

.PHONY: all clean
