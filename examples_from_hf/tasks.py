from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel


if __name__ == '__main__':
    # pipeline(): loads a default model and a pre-processing class capable of inference for any task
    # Sentiment analysis
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    classifier("Having three long haired, heavy shedding dogs at home, I was pretty skeptical that this could hold up "
               "to all the hair and dirt they trek in, but this wonderful piece of tech has been nothing short of a "
               "godsend for me! ")

    # Topic classification
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    classifier("Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners to "
               "the broad range of Machine Learning concepts, applications, challenges, and solutions, "
               "while utilizing interesting real-life datasets", candidate_labels=["art", "natural science",
                                                                                   "data analysis"],)
    # Text generator
    generator = pipeline("text-generation", model="gpt2")
    generator("This course will teach you")

    # Masked Language Modelling
    unmasker = pipeline("fill-mask", "distilroberta-base")
    unmasker("This course will teach you all about <mask> models.", top_k=4)

    # Name Entity Recognition (NER): identifying and categorizing key information (entities) in text.
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
    ner("My name is Roberta and I work with IBM Skills Network in Toronto")

    # Question Answering
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    question = "Which name is also used to describe the Amazon rainforest in English?"
    context = "The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle."
    qa_model(question=question, context=context)

    # Text Summarization
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summarizer("""Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners 
    to the broad range of Machine Learning concepts, applications, challenges, and solutions, while utilizing 
    interesting real-life datasets. So, what is EDA and why is it important to perform it before we dive into any 
    analysis? EDA is a visual and statistical process that allows us to take a glimpse into the data before the 
    analysis. It lets us test the assumptions that we might have about the data, proving or disproving our prior 
    believes and biases. It lays foundation for the analysis, so our results go along with our expectations. In a 
    way, it’s a quality check for our predictions. As any data scientist would agree, the most challenging part in 
    any data analysis is to obtain a good quality data to work with. Nothing is served to us on a silver plate, 
    data comes in different shapes and formats. It can be structured and unstructured, it may contain errors or be 
    biased, it may have missing fields, it can have different formats than what an untrained eye would perceive. For 
    example, when we import some data, very often it would contain a time stamp. To a human it is understandable 
    format that can interpreted. But to a machine, it is not interpretable, so it needs to be told what that means, 
    the data needs to be transformed into simple numbers first. There are also different date-time conventions 
    depending on a country (i.e., Canadian versus USA), metric versus imperial systems, and many other data features 
    that need to be recognized before we start doing the analysis. Therefore, the first step before performing any 
    analysis – is get really aquatinted with your data! This course will teach you to ‘see’ and to ‘feel’ the data as 
    well as to transform it into analysis-ready format. It is introductory level course, so no prior knowledge is 
    required, and it is a good starting point if you are interested in getting into the world of Machine Learning. 
    The only thing that is needed is some computer with internet, your curiosity and eagerness to learn and to apply 
    acquired knowledge.  If you live in Canada, you might be interested about gasoline prices in different cities or 
    if you are an insurance actuary you need to analyze the financial risks that you will take based on your clients 
    information. Whatever is the case, you will be able to do your own analysis, and confirm or disprove some of the 
    existing information. The course contains videos and reading materials, as well as well as a lot of interactive 
    practice labs that learners can explore and apply the skills learned. It will allow you to use Python language in 
    Jupyter Notebook, a cloud-based skills network environment that is pre-set for you with all available to be 
    downloaded packages and libraries. It will introduce you to the most common visualization libraries such as 
    Pandas, Seaborn, and Matplotlib to demonstrate various EDA techniques with some real-life datasets.""")

    # Translation
    en_fr_translator = pipeline("translation_en_to_fr", model="t5-small")
    en_fr_translator("How old are you?")

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    translator("La science des données est la meilleure.")

    # GPT-2
    configuration = GPT2Config()  # Initializing a GPT2 configuration
    model = GPT2Model(configuration)  # Initializing a model (with random weights) from the configuration
    print(model.config)  # Different pre-trained models are available for different tasks

    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Preprocessing
    print(tokenizer("Hello world")["input_ids"])

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)  # Inference
    next_token_logits = outputs.logits[:, -1]

    print(next_token_logits)
