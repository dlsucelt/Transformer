# Transformer
Here's two demo notebooks on the now-mainstream **Transformer** architecture. We show how to use it for neural language modeling, as well as how to use a trained language model for transfer learning into a downstream task (in this case, text classification).

Aside from two demo notebooks, we provide an implementation of the Transformer architecture, standalone and ready for use.

## Requirements
* PyTorch v.1.1.0 -- We need 1.1.0 as this incorporates ``nn.MultiheadAttention```. Otherwise, we'll have to write our own implementation of the self attention mechanism. I'll add in an implementation of it soon for backwards compatibility.
* A sizeable GPU -- We need a big enough GPU to train and finetune models. Transformers are *huge* and we need some space to accommodate a large enough batch size that won't hurt performance. We use an NVIDIA Tesla V100 with 16GB VRAM for testing the notebooks. Anything around the same amount of GPU memory will do.

## The Transformer Architecture
The Transformer (Vaswani, et al., 2015) was introduced as a drop-in replacement for RNN-based encoders in neural machine translation systems. It uses only attention mechanisms and standard feed-forward layers, dispensing with both recurrent and convolutional layers used by other wors at the time. This offers two advantages:

1. The model is significantly faster. The rpoblem with RNNs is its sequential nature, which makes it hard to encode batches of data at once. It also makes it tediously hard to parallelize efficiently. Since Transformers only use feed-forward layers, parallelization is easy and the model is faster than RNN-based models at training and inference time.
2. The model can see multiple "contexts" at once. The sequential nature of an RNN makes it hard to backtrack once it encodes a certain token suboptimally. Multihead attention in Transformers allow it to "attend to multiple positions" of tokens in a sequence all at once, giving it more flexibility in representing and learning from data.

We won't discuss the attention mechnism in detail. The models in this repository can be used as direct replacements for RNN layers in your own work, given enough PyTorch know-how. For good readings on attention mechanisms and how are used in the context of NLP, we refer you to Vaswani, et al. (2015); Luong, et al., (2014); Shaw, et al. (2018); Devlin, et al. (2019), and; Radford, et al. (2019).

## Acknowledgements
This work is partly based and inspired from the NAACL Transfer Learning tutorial [code](https://github.com/huggingface/naacl_transfer_learning_tutorial) by Sebastian Ruder, Matthew Peters, Swabha Swayamdipta and Thomas Wolf. Our demo notebooks were written to give a more fine-grained example of the use of Transformers as their implementations are a little more self-contained. That said, we made sure our implementations are compaitble with their checkpoints (we even use their pretrained models in the code!) 
