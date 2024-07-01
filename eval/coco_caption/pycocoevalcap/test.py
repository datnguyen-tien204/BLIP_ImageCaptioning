from tokenizer.ptbtokenizer import PTBTokenizer

tokenizer = PTBTokenizer()

gts = {"annotations": [{"image_id": 184613, "caption": "A child holding a flowered umbrella and petting a yak.", "id": 474921}]}
gts = tokenizer.tokenize(gts)
print(gts)
