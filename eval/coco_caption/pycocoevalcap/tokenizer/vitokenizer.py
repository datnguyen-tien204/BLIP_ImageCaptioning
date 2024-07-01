from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from underthesea import word_tokenize
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"] 

class VITokenizer:
    def __init__(self):
        self.model = word_tokenize

    def tokenize(self, captions_for_image):
        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        final_tokenized_captions_for_image = {}
        image_id = [k for k, v in list(captions_for_image.items()) for _ in range(len(v))]
        sentences = [c['caption'] for k, v in list(captions_for_image.items()) for c in v]

        list_of_tokens = []
        for sent in sentences:
            doc = self.model(sent, format="text")
            list_of_tokens.append(doc.lower())

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for k, tokens in zip(image_id, list_of_tokens):
            if not k in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            tokenized_caption = ' '.join([w for w in tokens.rstrip().split(' ') \
                    if w not in PUNCTUATIONS])
            final_tokenized_captions_for_image[k].append(tokenized_caption)

        return final_tokenized_captions_for_image