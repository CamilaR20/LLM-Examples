# Transformer Examples
Text generation examples with Hugging Face Transformers library and commercial APIs.

## **Installation**
### Using a local machine
You can use Miniconda to setup the environment.

```
conda env create -f environment.yml 
``` 

### Using Google Colab
Run the following commands to install the packages needed to run the notebooks:
``` 
    %pip install -q -q transformers accelerate bitsandbytes
``` 
You can also add the following lines to mount google drive and use it as the cache directory for the models:
```  
    from google.colab import drive  
    drive.mount('/content/drive')
```
