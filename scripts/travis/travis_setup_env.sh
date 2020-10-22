# script to be sourced in travis yml

# install miniconda
export MINICONDA=$HOME/miniconda
export PATH="$MINICONDA/bin:$PATH"
hash -r
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -f -p $MINICONDA
conda config --set always_yes yes
conda update conda
conda info -a
conda create -n testenv python=$TRAVIS_PYTHON_VERSION
source activate testenv
conda install $PYEIT_DEPS

# copied from dmlc-core
# setup all enviroment variables

# export PATH=${HOME}/.local/bin:${PATH}
export PATH=${PATH}:${CACHE_PREFIX}/bin
export CACHE_PREFIX=${HOME}/.cache/usr
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${CACHE_PREFIX}/include
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${CACHE_PREFIX}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${CACHE_PREFIX}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CACHE_PREFIX}/lib
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${CACHE_PREFIX}/lib

alias make="make -j4"

# setup the cache prefix folder
if [ ! -d ${HOME}/.cache ]; then
    mkdir ${HOME}/.cache
fi

if [ ! -d ${CACHE_PREFIX} ]; then
    mkdir ${CACHE_PREFIX}
fi
if [ ! -d ${CACHE_PREFIX}/include ]; then
    mkdir ${CACHE_PREFIX}/include
fi
if [ ! -d ${CACHE_PREFIX}/lib ]; then
    mkdir ${CACHE_PREFIX}/lib
fi
if [ ! -d ${CACHE_PREFIX}/bin ]; then
    mkdir ${CACHE_PREFIX}/bin
fi

