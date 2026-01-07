import torch

import config
from dataset import normalize_text, normalize_sequence_length
from minbpe.regex import RegexTokenizer
from model import TransformerArchitecture

# initialize tokenizer
tokenizer = RegexTokenizer()
tokenizer.load(f"assets/tok{config.VOCAB_SIZE}_{config.TEXT_NORMALIZATION_MODE}.model")

# samples from a test set for inference
pos_test_review = """I attended an advance screening of this film not sure of what to expect from Kevin Costner and Ashton Kutcher; both have delivered less than memorable performances & films. While the underlying "general" storyline is somewhat familiar, this film was excellent. Both Costner and Kutcher delivered powerful performances playing extremely well off each other. The human frailties and strengths of their respective characters were incredibly played by both; the scene when Costner confronts Kutcher with the personal reasons why Kutcher joined the Coast Guard rescue elite was the film's most unforgettable emotional moment. The "specific" storyline was an education in itself depicting the personal sacrifice and demanding physical training the elite Coast Guard rescuers must go through in preparation of their only job & responsibility...to save lives at sea. The special effects of the rescue scenes were extremely realistic and "wowing"...I haven't seen such angry seas since "The Perfect Storm". Co-star Clancy Brown (HBO's "Carnivale" - great to see him again) played the captain of the Coast Guard's Kodiak, Alaska base in a strong, convincing role as a leader with the prerequisite and necessary ice water in his veins. The film wonderfully, and finally, gives long overdue exposure and respect to the Coast Guard; it had the audience applauding at the end."""
neg_test_review = """Some may go for a film like this but I most assuredly did not. A college professor, David Norwell, suddenly gets a yen for adoption. He pretty much takes the first child offered, a bad choice named Adam. As it turns out Adam doesn't have both oars in the water which, almost immediately, causes untold stress and turmoil for Dr. Norwell. This sob story drolly played out with one problem after another, all centered around Adam's inabilities and seizures. Why Norwell wanted to complicate his life with an unknown factor like an adoptive child was never explained. Along the way the good doctor managed to attract a wifey to share in all the hell the little one was dishing out. Personally, I think both of them were one beer short of a sixpack. Bypass this yawner."""

# initialize model and load weights
model = TransformerArchitecture(
    config.EMBED_DIM, config.VOCAB_SIZE, config.SEQ_LEN,
    config.N_LAYER, config.N_HEADS, config.FF_DIM, config.DROPOUT
).to(config.DEVICE)

model.load_state_dict(torch.load(f"./weights/{config.run_name}.pt", weights_only=True))
model.eval()

for x, y in [(pos_test_review, "positive"), (neg_test_review, "negative")]:
    print("\n", x)

    # normalize the text and encode it into token IDs
    x = torch.tensor(tokenizer.encode(normalize_text(x, mode=config.TEXT_NORMALIZATION_MODE)), dtype=torch.long)

    # adjust the sequence length to match config.SEQ_LEN (pad or truncate)
    x = normalize_sequence_length(x)

    # add a batch dimension and move the input to the target device
    x = x.unsqueeze(0).to(config.DEVICE)

    # run the model to obtain the prediction
    logit = model(x)
    prediction = (torch.sigmoid(logit) > 0.5).float()

    # convert the prediction to a readable format
    label = "positive" if prediction.item() else "negative"
    print(f"- prediction: {label}, logit: {logit.item():.2f}, ground truth: {y}")
