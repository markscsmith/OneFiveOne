
# https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/install-pytorch.html

pip3 install -r requirements.txt

if [ ! -f "torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl" ]; then
    curl -O https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl
fi

if [ ! -f "torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl" ]; then
    curl -O https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl
fi

pip3 install --force-reinstall torch-2.1.2+rocm6.0-cp310-cp310-linux_x86_64.whl torchvision-0.16.1+rocm6.0-cp310-cp310-linux_x86_64.whl


# workaround for bug in parsing of driver names in tensorflow-rocm 2.14.0.600 https://github.com/ROCm/tensorflow-upstream/issues/2410
if [ ! -f "tensorflow_rocm-2.14.0.600-cp310-cp310-manylinux2014_x86_64.whl" ]; then
    curl -O http://ml-ci.amd.com:21096/job/tensorflow/job/release-rocmfork-r214-rocm-enhanced/job/release-build-whl/lastSuccessfulBuild/artifact/packages-3.10/tensorflow_rocm-2.14.0.600-cp310-cp310-manylinux2014_x86_64.whl
fi

pip3 install tensorflow_rocm-2.14.0.600-cp310-cp310-manylinux2014_x86_64.whl


