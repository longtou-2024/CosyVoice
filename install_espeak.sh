#!/usr/bin/env bash

git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng && ./autogen.sh && ./configure --prefix=$PWD && make && make install

