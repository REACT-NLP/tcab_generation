"""
Toxic comment classificaion (binary) models.
"""
import torch
import OpenAttack
import numpy as np
from textattack.datasets import TextAttackDataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from transformers import XLNetTokenizer
from transformers import XLNetForSequenceClassification
from transformers import GPT2Model
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from transformers import AutoTokenizer
from transformers import AutoModel


stop_words = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
    "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back",
    "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
    "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg",
    "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
    "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first",
    "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get",
    "give", "go", "had", "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon",
    "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc",
    "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
    "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most",
    "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine",
    "nobody", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others",
    "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put",
    "rather", "re", "same", "see", "serious", "several", "she", "should", "show", "side", "since", "sincere",
    "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
    "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence",
    "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too",
    "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very",
    "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who",
    "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
    "yours", "yourself", "yourselves", ">>>>"
]


####################
# Model Definitions
####################


class BERTClassifier(torch.nn.Module):
    """
    Simple text classification model using a pretrained
    BERTSequenceClassifier model to tokenize, embed, and classify the input.
    """
    def __init__(self, pretrained_weights='bert-base-cased', max_seq_len=100, num_labels=2):
        super(BERTClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load BERT-base pretrained model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.classifier = BertForSequenceClassification.from_pretrained(pretrained_weights,
                                                                        return_dict=True,
                                                                        num_labels=num_labels)

    def forward(self, text_list):
        """
        Define the forward pass.
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_seq_len, return_tensors='pt').to(self.device)
        return self.classifier(**inputs).logits

    def gradient(self, text, label, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Return gradients for this sample.
        """
        self.classifier.zero_grad()  # reset gradients
        pred = self.forward(text)  # forward pass
        label = label.to(self.device)
        loss = loss_fn(pred, label)  # compute loss
        loss.backward()  # backward pass
        gradients = [p.grad for p in self.classifier.parameters()]
        return gradients


class RoBERTaClassifier(torch.nn.Module):
    """
    Simple text classification model using
    a pretrained RoBERTaSequenceClassifier model to tokenize, embed, and classify
    the input.
    """
    def __init__(self, pretrained_weights='roberta-base', max_seq_len=100, num_labels=2):
        super(RoBERTaClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load RoBERTa-base pretrained model
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        self.classifier = RobertaForSequenceClassification.from_pretrained(pretrained_weights,
                                                                           return_dict=True,
                                                                           num_labels=num_labels)

    def forward(self, text_list):
        """
        Define the forward pass.
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_seq_len, return_tensors='pt').to(self.device)
        return self.classifier(**inputs).logits

    def gradient(self, text, label, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Return gradients for this sample.
        """
        self.classifier.zero_grad()  # reset gradients
        pred = self.forward(text)  # forward pass
        label = label.to(self.device)
        loss = loss_fn(pred, label)  # compute loss
        loss.backward()  # backward pass
        gradients = [p.grad for p in self.classifier.parameters()]
        return gradients


class XLNetClassifier(torch.nn.Module):
    """
    Simple text classification model using a pretrained
    BERTSequenceClassifier model to tokenize, embed, and classify the input.
    """
    def __init__(self, pretrained_weights='xlnet-base-cased', max_seq_len=250, num_labels=2):
        super(XLNetClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load pretrained model
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
        self.classifier = XLNetForSequenceClassification.from_pretrained(pretrained_weights,
                                                                         return_dict=True,
                                                                         num_labels=num_labels)

    def forward(self, text_list):
        """
        Define the forward pass.
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_seq_len, return_tensors='pt').to(self.device)
        return self.classifier(**inputs).logits

    def gradient(self, text, label, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Return gradients for this sample.
        """
        self.classifier.zero_grad()  # reset gradients
        pred = self.forward(text)  # forward pass
        label = label.to(self.device)
        loss = loss_fn(pred, label)  # compute loss
        loss.backward()  # backward pass
        gradients = [p.grad for p in self.classifier.parameters()]
        return gradients


class UCLMRClassifier(torch.nn.Module):
    """
    UCL Machine Reading group Fake News Classifier.

    Input: 5K TF-IDF features for the headline, concatenated with
           5K TF-IDF features for the body, concatenated with
           cosine similarity between the two feature vectors.

    Model: 1 hidden layer with dimension 100 and ReLU activation.

    Output: 4 classes: agree, disagree, discuss, unrelated.

    Random batches
    Dropout: 0.4
    Learning rate: 0.001
    Optimizer: adam
    Loss_fn: crossentropy
    Weight decay: 1e-5
    Batch_size: 500
    Epochs: 1,000
    Max_norm: 5.0
    Trains on the entire training set, no validation set.
    """
    def __init__(self, tf_vectorizer, tfidf_vectorizer, n_hidden=100, n_classes=4, dropout=0.4):
        super(UCLMRClassifier, self).__init__()

        # save no. classes
        self.n_classes = n_classes

        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # transforms the input, make sure max tokens is 5000
        self.tf_vectorizer = tf_vectorizer
        self.tfidf_vectorizer = tfidf_vectorizer
        assert self.tfidf_vectorizer.max_features == 5000
        assert self.tf_vectorizer.max_features == 5000

        # hidden layer
        self.fc1 = torch.nn.Linear(10001, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_classes)

        # dropout layer
        self.d1 = torch.nn.Dropout(dropout)
        self.d2 = torch.nn.Dropout(dropout)

        # relu activation function
        self.relu = torch.nn.ReLU()

    def forward(self, text_input_list, logits=None):
        """
        `logits` only there for compatibility.
        """

        # separate input into headline and body text
        token = '|||||'
        text_list = [x.split(token) for x in text_input_list]
        head_list, body_list = list(zip(*text_list))

        # transform
        head_tf = np.array(self.tf_vectorizer.transform(head_list).todense())
        body_tf = np.array(self.tf_vectorizer.transform(body_list).todense())

        head_tfidf = np.array(self.tfidf_vectorizer.transform(head_list).todense())
        body_tfidf = np.array(self.tfidf_vectorizer.transform(body_list).todense())

        # compute cosine similarity between head and body
        similarity = np.array([np.dot(head_tfidf[i], body_tfidf[i].T) for i in range(head_tfidf.shape[0])])

        # concatenate head and body features, shape=(batch_size, 10,001)
        feature_vec = np.hstack([head_tf, body_tf, similarity.reshape(-1, 1)])
        feature_vec = torch.tensor(feature_vec, dtype=torch.float32).to(self.device)

        # fully connected network
        x = self.fc1(feature_vec)
        x = self.relu(x)
        x = self.d1(x)

        x = self.fc2(x)
        x = self.d2(x)

        return x

    def gradient(self, text, label, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Return gradients for this sample.
        """
        self.zero_grad()  # clear gradients
        pred = self.forward(text)  # forward pass
        label = label.to(self.device)
        loss = loss_fn(pred, label)
        loss.backward()  # compute gradients
        gradients = [p.grad for p in self.parameters()]
        return gradients


####################
# Feature Extractors
####################


class BertSampleWiseFeatureEmbedder:
    """
    Embedder using pretrained bert from huggingface
    the code was basically copied from the link below.
    see: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens

    Modified from: https://github.com/zhouhanxie/react-detection/\
    blob/71e9e4ea5204cd35514f06bb61c49bdf4e5c06c6/lineardetect-bert.py#L40
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask  # normalize

    def embed_texts(self, sentences: list):
        """
        Embed each sentence string into a weighted sum of its token embeddings.
        """

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True,
                                       max_length=128,
                                       return_tensors='pt')
        encoded_input.to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings


####################
# Wrappers
####################

class TextAttackModelWrapper:
    """
    Wrapper to interface more easily with TextAttack.
    """
    def __init__(self, model, batch_size=32):
        if not isinstance(model, torch.nn.Module):
            raise TypeError('Model must be torch.nn.Module, got type {}'.format(str(model)))

        self.model = model
        self.batch_size = batch_size

    def __call__(self, text_list):
        outputs = []
        with torch.no_grad():
            for i in range(0, len(text_list), self.batch_size):
                input_ = text_list[i: i + self.batch_size]
                output = self.model(input_)
                outputs.append(np.array(output.cpu()))

        return np.vstack(outputs)


class OpenAttackModelWrapper(OpenAttack.Classifier):
    """
    Wrapper around a custom model to make interfacing with
    OpenAttack easier.
    """
    def __init__(self, model, attack_name):
        self.model = model
        self.attack_name = attack_name

    # call your custom model here with the given "input_"
    def get_prob(self, input_):

        # get output probabilities for this batch of samples
        output = self.model(input_)

        # output probabilities should be of shape=(len(input_), no. of classes)
        output = torch.softmax(output, axis=1)

        return output.detach().cpu().numpy()


class TextAttackDatasetWrapper(TextAttackDataset):
    """
    Creates a dataset object that interacts well with TextAttack.
    """
    def __init__(self, examples, label_names=['Non-Toxic', 'Toxic']):
        self.examples = examples
        self.label_names = label_names
        self.i_ = 0

    def __next__(self):
        if self.i_ >= len(self.examples):
            raise StopIteration
        example = self.examples[self.i_]
        self.i_ += 1
        return example

    def __getitem__(self, i):
        if isinstance(i, int):
            result = self.examples[i]
        else:
            result = [ex for ex in self.examples[i]]
        return result


class NodeActivationWrapper(torch.nn.Module):
    """
    Model that adds a forward hook to each layer
    to obtain activations of all internal nodes
    from a forward pass of the model.
    """
    def __init__(self, model):
        super().__init__()
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.activation_hook_ = Hook()

    def activation(self, text):
        """
        Do forward pass and collect node activations.
        """
        self._add_hooks()  # add forward hooks

        self.model(text)  # do forward pass to trigger forward hooks
        result = self.activation_hook_.values_  # collect activations

        self.activation_hook_.clear()  # clear hook values
        self._clear_hooks()  # remove handles

        return result

    # private
    def _add_hooks(self):
        """
        Register a hook for each layer.
        """
        self.handles_ = []
        for name, module in self._get_individual_modules():
            self.handles_.append(module.register_forward_hook(self.activation_hook_))

    def _clear_hooks(self):
        """
        Remove hooks.
        """
        for handle in self.handles_:
            handle.remove()
        self.handles_ = []

    def _get_individual_modules(self):
        """
        Returns all individual (not grouped) modules in the network.
        """
        assert isinstance(self.model, torch.nn.Module)

        result = []
        for name, module in self.model.named_modules():
            has_child = False

            for child in module.children():
                has_child = True

            if not has_child:
                result.append((name, module))

        return result


class SaliencyWrapper(torch.nn.Module):
    """
    Model wrapper that adds the following:
        forward hook to the embedding layer to obtain token embeddings;
        backward hook to the embedding layer to obtain gradients w.r.t token embeddings;
    The values returned from these hooks are then used to compute
    saliency features such as "simple gradients", "integrated gradients", etc.
    """
    def __init__(self, model, embedding_layer=None, saliency_type='simple_gradient'):
        super().__init__()
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.saliency_type = saliency_type
        self.handles = []

        # make sure given embedding layer is a PyTorch module
        if embedding_layer is not None:
            assert isinstance(embedding_layer, torch.nn.Module)

        elif 'XLNetForSequenceClassification' in self.model:
            embedding_layer = self.model.transformer.word_embedding

        # attempt to find the embedding layer if not already given one
        else:

            for module in self.model.modules():

                if isinstance(module, BertEmbeddings):
                    embedding_layer = module.word_embeddings

                elif isinstance(module, RobertaEmbeddings):
                    embedding_layer = module.word_embeddings

                elif isinstance(module, GPT2Model):
                    embedding_layer = module.wte

        # throw error if embedding layer is not available
        if embedding_layer is None:
            raise ValueError('embedding layer cannot be found!')

        else:
            self.embedding_layer = embedding_layer

    def saliency(self, text, label, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Input: str, text to compute the gradients with respect to;
               Torch.tensor, tensor containing the label for this sample.
        Returns: 1D Torch.tensor containing a normalized attribution values
                 for each token in the text input.
        """
        label = label.to(self.model.device)

        # compute simple gradients
        if self.saliency_type == 'simple_gradient':
            saliency = self._simple_gradient(text, label, loss_fn)

        # compute integrated gradients
        elif self.saliency_type == 'integrated_gradient':
            saliency = self._integrated_gradient(text, label, loss_fn)

        # compute integrated gradients
        elif self.saliency_type == 'smooth_gradient':
            saliency = self._smooth_gradient(text, label, loss_fn)

        return saliency

    # private
    def _clear_hooks(self):
        for handle in self.handles:
            handle.remove()

    def _simple_gradient(self, text, label, loss_fn):
        """
        Computes "simple gradient" attributions for each token.
        Refernce: https://github.com/allenai/allennlp/blob/master/\
                  allennlp/interpret/saliency_interpreters/simple_gradient.py
        """

        # data containers
        embeddings_hook = Hook()
        gradients_hook = Hook()

        # add hooks to obtain token embeddings and gradients
        self.handles.append(self.embedding_layer.register_forward_hook(embeddings_hook))
        self.handles.append(self.embedding_layer.register_backward_hook(gradients_hook))

        self.model.zero_grad()  # reset gradients
        pred = self.model.forward(text)  # forward pass
        loss = loss_fn(pred, label)  # compute loss
        loss.backward()  # backward pass

        # extract embeddings and gradients (gradients come in reverse order)
        embeddings = embeddings_hook.values_[0][0]
        gradients = gradients_hook.values_[0][0][0].flip(dims=[0])

        # normalize
        emb_grad = embeddings * gradients
        norm = torch.linalg.norm(emb_grad, ord=1)
        saliency = torch.tensor([(torch.abs(e) / norm).item() for e in emb_grad])

        # clean up
        self._clear_hooks()

        return saliency

    def _integrated_gradient(self, text, label, loss_fn, steps=10):
        """
        Computes "integrated gradient" attributions for each token.
        Refernce: https://github.com/allenai/allennlp/blob/master/\
                  allennlp/interpret/saliency_interpreters/integrated_gradient.py
        """
        embeddings_list = []  # container to store the original embeddings
        gradient_sum = None  # stores the running total of gradients

        # approximate integration by summing over a finte number of steps
        for i, alpha in enumerate(np.linspace(0.1, 1.0, num=steps, endpoint=True)):

            # custom hook to modify embeddings on the forward pass
            def custom_forward_hook(module, module_in, module_out):

                # save original embeddings
                if i == 0:
                    embeddings_list.append(module_out)

                # modify embeddings to generate different gradients
                module_out.mul_(alpha)

            # data container
            gradients_hook = Hook()

            # add hooks to obtain token embeddings and gradients
            self.handles.append(self.embedding_layer.register_forward_hook(custom_forward_hook))
            self.handles.append(self.embedding_layer.register_backward_hook(gradients_hook))

            self.model.zero_grad()  # reset gradients
            pred = self.model.forward(text)  # forward pass
            loss = loss_fn(pred, label)  # compute loss
            loss.backward()  # backward pass

            # extract modified gradients (these come in reverse order)
            gradients = gradients_hook.values_[0][0][0].flip(dims=[0])

            # keep a running sum of the modified gradients
            if gradient_sum is None:
                gradient_sum = gradients

            else:
                gradient_sum += gradients

            # remove hooks
            self._clear_hooks()

        embeddings = embeddings_list[0][0]  # extract original embeddings
        gradients = gradient_sum / steps  # compute average gradients

        # normalize
        emb_grad = embeddings * gradients
        norm = torch.linalg.norm(emb_grad, ord=1)
        saliency = torch.tensor([(torch.abs(e) / norm).item() for e in emb_grad])

        return saliency

    def _smooth_gradient(self, text, label, loss_fn, std_dev=0.01, num_samples=10):
        """
        Computes "smooth gradient" attributions for each token.
        Refernce: https://github.com/allenai/allennlp/blob/master/allennlp/\
                  interpret/saliency_interpreters/smooth_gradient.py
        """
        embeddings_list = []  # container to store the original embeddings
        gradient_sum = None  # stores the running total of gradients

        # approximate integration by summing over a finte number of steps
        for i in range(num_samples):

            # add random noise to the embeddings
            def custom_forward_hook(module, module_in, module_out):

                # save original embeddings
                if i == 0:
                    embeddings_list.append(module_out)

                # random noise = N(0, stdev * (max - min))
                scale = module_out.detach().max() - module_out.detach().min()
                noise = torch.randn(module_out.shape, device=module_out.device) * std_dev * scale

                # add the random noise
                module_out.add_(noise)

            # data container
            gradients_hook = Hook()

            # add hooks to obtain token embeddings and gradients
            self.handles.append(self.embedding_layer.register_forward_hook(custom_forward_hook))
            self.handles.append(self.embedding_layer.register_backward_hook(gradients_hook))

            self.model.zero_grad()  # reset gradients
            pred = self.model.forward(text)  # forward pass
            loss = loss_fn(pred, label)  # compute loss
            loss.backward()  # backward pass

            # extract modified gradients (these come in reverse order)
            gradients = gradients_hook.values_[0][0][0].flip(dims=[0])

            # keep a running sum of the modified gradients
            if gradient_sum is None:
                gradient_sum = gradients

            else:
                gradient_sum += gradients

            # remove hooks
            self._clear_hooks()

        embeddings = embeddings_list[0][0]  # extract original embeddings
        gradients = gradient_sum / num_samples  # compute average gradients

        # normalize
        emb_grad = embeddings * gradients
        norm = torch.linalg.norm(emb_grad, ord=1)
        saliency = torch.tensor([(torch.abs(e) / norm).item() for e in emb_grad])

        return saliency


class Hook:
    """
    Object to store output from registered hooks.
    """
    def __init__(self, output=True, custom_fn=None):
        self.output = output
        self.custom_fn = custom_fn

        self.values_ = []

    def __call__(self, module, module_in, module_out):
        if self.output:
            result = module_out
        else:
            result = module_in

        self.values_.append(result)

    def clear(self):
        self.values_ = []
