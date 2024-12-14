from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import AddedToken
import sys

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])

    added_tokens = ['<|speech-{}|>'.format(i) for i in range(4096)] + \
                   ['<|speaker-{}|>'.format(i) for i in range(256)] + \
                   ['<|speed-{}|>'.format(i) for i in range(7)]

    added_tokens = [AddedToken(x) for x in added_tokens]
    added_special_tokens = ['<|START_TEXT|>', '<|END_TEXT|>', '<|START_AUDIO|>', '<|END_AUDIO|>', "<|START_SPEAKER|>",
                            "<|END_SPEAKER|>", '<|START_SPEED|>', '<|END_SPEED|>', '<|SEP_AUDIO|>']
    added_special_tokens = [AddedToken(x, special=True) for x in added_special_tokens]
    tokenizer.add_tokens(added_tokens)
    tokenizer.add_tokens(added_special_tokens)
    tokenizer.save_pretrained(sys.argv[2])
    model = AutoModelForCausalLM.from_pretrained(sys.argv[1])
    model.resize_position_embeddings(len(tokenizer))
    model.save_pretrained(sys.argv[2])
