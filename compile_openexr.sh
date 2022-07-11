mkdir -p openexr
cd openexr
wget https://github.com/AcademySoftwareFoundation/openexr/archive/v2.5.3.tar.gz
tar -xvzf v2.5.3.tar.gz
mkdir -p  openexr-2.5.3-build
cd openexr-2.5.3-build
cmake ../openexr-2.5.3
make clean
make -j `getconf _NPROCESSORS_ONLN`
sudo make install
#/usr/local/include/OpenEXR/
# /usr/local/lib/