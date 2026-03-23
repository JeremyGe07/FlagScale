#!/bin/bash

set -e

# Helper function to check if a package is installed
check_package_installed() {
    local package_name=$1
    pip show "$package_name" >/dev/null 2>&1
}

# Helper function to check if a package is installed with specific version
check_package_version() {
    local package_name=$1
    local required_version=$2
    if check_package_installed "$package_name"; then
        local installed_version=$(pip show "$package_name" | grep "^Version:" | awk '{print $2}')
        if [ "$installed_version" = "$required_version" ]; then
            return 0
        fi
    fi
    return 1
}

configure_cudnn_env() {
    local site_packages_dir=""
    local cudnn_root=""

    site_packages_dir=$(python -c "import site; print(site.getsitepackages()[0])")
    cudnn_root="${site_packages_dir}/nvidia/cudnn"

    if [ ! -d "$cudnn_root/include" ] || [ ! -d "$cudnn_root/lib" ]; then
        return 1
    fi

    export CUDNN_PATH="$cudnn_root"
    export CUDNN_INCLUDE_PATH="$cudnn_root/include"
    export CUDNN_LIBRARY_PATH="$cudnn_root/lib"
    export LD_LIBRARY_PATH="$cudnn_root/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

# Extract CUDA version and format as cu128, cu124, etc.
CUDA_VERSION=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | tr -d ',')
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d '.' -f 1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d '.' -f 2)
CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}" 

# Install torch based on CUDA version
if [ "${CUDA_MAJOR}.${CUDA_MINOR}" = "12.4" ]; then
    # CUDA 12.4: install torch 2.5.1+cu124
    if check_package_version "torch" "2.5.1+cu124" && check_package_installed "torchvision" && check_package_installed "torchaudio"; then
        echo "✓ torch 2.5.1, torchvision, and torchaudio are already installed, skipping..."
    else
        echo "Installing torch 2.5.1 for CUDA 12.4"
        pip install --no-cache-dir --verbose torch==2.5.1+cu124 torchvision==0.22.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
    fi
elif [ "${CUDA_MAJOR}.${CUDA_MINOR}" = "12.8" ]; then
    # CUDA 12.8: install torch 2.7.1+cu128
    if check_package_version "torch" "2.7.1+cu128" && check_package_version "torchvision" "0.22.1+cu128" && check_package_version "torchaudio" "2.7.1+cu128"; then
        echo "✓ torch 2.7.1, torchvision 0.22.1, and torchaudio 2.7.1 are already installed, skipping..."
    else
        echo "Installing torch 2.7.1 for CUDA 12.8"
        pip install --no-cache-dir --verbose torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    fi
else
    # Other CUDA versions: install torch without version specification
    if check_package_installed "torch" && check_package_installed "torchvision" && check_package_installed "torchaudio"; then
        echo "✓ torch, torchvision, and torchaudio are already installed, skipping..."
    else
        echo "Installing torch (latest) for CUDA ${CUDA_VERSION}"
        pip install --no-cache-dir --verbose torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}
    fi
fi

# Install causal-conv1d
if check_package_installed "causal_conv1d"; then
    echo "✓ causal-conv1d is already installed, skipping..."
else
    echo "Installing causal-conv1d v1.2.2.post1"
    pip install --no-cache-dir --no-build-isolation --verbose git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.2.post1
fi

# Install grouped_gemm
if check_package_installed "grouped_gemm"; then
    echo "✓ grouped_gemm is already installed, skipping..."
else
    echo "Installing grouped_gemm v1.1.2"
    pip install --no-cache-dir --no-build-isolation --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.2
fi

# Install flash-attention  for megatron-lm
if check_package_installed "flash_attention"; then
    echo "✓ flash_attention is already installed, skipping..."
else
    echo "Installing flash_attention v2.8.0.post2"
    cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
    torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
    cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
    cxx11abi=$(python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
    flash_attn_version="2.8.0.post2"
    flash_attn_wheel="flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx11abi${cxx11abi}-cp${cp}-cp${cp}-linux_x86_64.whl"
    pip install --no-cache-dir --verbose "https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/${flash_attn_wheel}"
fi

# Install TransformerEngine for megatron-lm
if check_package_installed "transformer_engine"; then
    echo "✓ transformer_engine is already installed, skipping..."
else
    configure_cudnn_env || true
    python -m pip uninstall -y transformer-engine transformer-engine-torch >/dev/null 2>&1 || true
    echo "Installing TransformerEngine (commit 0289e76380088358a584d809faf69effab1a7cda)"
    git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    git checkout 0289e76380088358a584d809faf69effab1a7cda
    NVTE_FRAMEWORK=pytorch python -m pip install --no-build-isolation --verbose .
    cd ..
    rm -rf ./TransformerEngine
fi

# Install Apex for megatron-lm
if check_package_installed "apex"; then
    echo "✓ apex is already installed, skipping..."
else
    echo "Installing apex"
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./
    cd ..
    rm -r ./apex
fi
