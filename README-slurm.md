To run on slurm, first we need to set up the slurm machines (not hdp006) with a custom, user-installed python:


```
# Have to manuall install dependency

wget ftp://sourceware.org/pub/libffi/libffi-3.2.1.tar.gz
tar xvf libffi-3.2.1.tar.gz
cd libffi-3.2.1
./configure --prefix=$HOME/python/
make
make install

# Now download and install Python

wget https://www.python.org/ftp/python/3.8.4/Python-3.8.4.tgz

tar xvf Python-3.8.4.tgz
cd Python-3.8.4
export LD_LIBRARY_PATH=$HOME/pythonlib64
export LD_RUN_PATH=/$HOME/python/lib64
export PKG_CONFIG_PATH=$HOME/python/lib/pkgconfig
CPPFLAGS=-I$HOME/python/libffi-3.2.1/include LDFLAGS=-L$HOME/python/lib64 ./configure --enable-optimizations --with-ensurepip=install  --prefix=$HOME/python

make -j 8
make install

ln -s $HOME/python/bin/python3 $HOME/python/bin/python
export PATH=$HOME/python/bin:$PATH
export PYTHONPATH=$HOME/python/lib/python3.8
```

Now, use `venv` to create a virtual environment to install python packages:

```
cd autotext
python -m venv $HOME/autotext/flaml_env_slurm

source $HOME/autotext/flaml_env_slurm/bin/activate

pip install --upgrade pip
python -m pip install -r requirements.txt
```

Now, you can `source flaml_env_slurm/bin/activate` and you're off to the races!
