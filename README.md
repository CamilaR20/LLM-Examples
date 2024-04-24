# LLMs
LLM text generation examples with Hugging Face Transformer's library and commercial APIs.

## **Installation**

## Using your local machine
You can use Miniconda to setup the environment. To install it, you can follow the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install).

```
git clone --recursive https://github.com/CamilaR20/LLMs.git
conda env create -f environment.yml 
``` 

## Using Google Colab
Run the following commands to install the packages needed to run the notebooks:
``` 
    %pip install -q -q transformers accelerate bitsandbytes
``` 
You can also add the following lines to mount google drive and use it as the cache directory for the models:
```  
    from google.colab import drive  
    drive.mount('/content/drive')
```
