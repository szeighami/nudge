# Use the NVIDIA PyTorch image
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-03.html
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set the working directory in the container
WORKDIR /ft

# Copy the entire current directory into the container
COPY . .

# Install the Python dependencies
RUN pip install -r ./requirements.txt
