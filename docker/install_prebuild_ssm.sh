#!/bin/bash

# Check Torch version
torch_version=$(python -c "import torch; print(torch.__version__)")

# Check if Torch version has any letters
if [[ $torch_version =~ [a-zA-Z] ]]; then
    echo "Torch version contains letters indicating it is the nvidia image. Skipping installation."
else
    # Download file from <link>
    # Check if the file already exists
    if [ -f ssm_file.tar.gz ]; then
        echo "File already exists. Skipping download."
    else
        # Download file from <link>
        wget https://github.com/state-spaces/mamba/releases/download/v2.2.3.post2/mamba_ssm-2.2.3.post2+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    fi

    # Install the downloaded file using pip
    pip install mamba_ssm-2.2.3.post2+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi