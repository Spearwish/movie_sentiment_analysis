import config
from dataset import normalize_text
from minbpe.regex import RegexTokenizer
from stats import load_reviews

# initialize tokenizer
tokenizer = RegexTokenizer()

# load train reviews
all_reviews = load_reviews("datasets/train/", ["pos", "neg"])

# join everything into one long string
very_long_training_string = "\n".join(all_reviews)

# train tokenizer on normalized text
tokenizer.train(normalize_text(very_long_training_string, mode=config.TEXT_NORMALIZATION_MODE),
                vocab_size=config.VOCAB_SIZE, verbose=True)

# save tokXYZ.model and tokXYZ.vocab
tokenizer.save(f"assets/tok{config.VOCAB_SIZE}_{config.TEXT_NORMALIZATION_MODE}")
