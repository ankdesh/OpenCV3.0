Install following package to build with opengl and gtk support
sudo apt-get install libgtkglext1 libgtkglext1-dev

Install python-dev for python support
libpython-dev

Configure 
---------------------
>> cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=ON -DWITH_OPENGL=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules -DCMAKE_INSTALL_PREFIX=../install ../opencv-3.0.0-beta/
>> make 
>> make install

# To use SIFT, SURF from Python 
>> export PYTHONPATH=/home/ankdesh/workspace/vision/opencv/opencv-3.0/install/lib/python2.7/dist-packages

# To check if proper opencv module is loaded in python use
>>> import cv2
>>> print cv2.__version__

# Try sift in ipython
>>> sift = = cv2.xfeatures2d.SIFT_create() 

