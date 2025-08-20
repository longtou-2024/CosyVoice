git clone --recursive https://github.com/longtou-2024/CosyVoice.git
or
git submodule update --init --recursive

# install phonemizer
# git clone https://github.com/espeak-ng/espeak-ng.git
# cd espeak-ng && ./autogen.sh && ./configure --prefix=$PWD && make && make install
# pip install phonemizer==3.0
#
# export PATH="${TOOL_DIR}"/espeak-ng/bin:"${PATH:-}"
# export LD_LIBRARY_PATH="${TOOL_DIR}"/espeak-ng/lib:"${LD_LIBRARY_PATH:-}"
