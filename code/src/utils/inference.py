import wandb

from src.models import CompressorModel, ExpanderModel

data_dict = {
    "Overweight Kid": "Dan's parents were overweight. Dan was overweight as well. The doctors told his parents it was unhealthy. His parents understood and decided to make a change. They got themselves and Dan on a diet.",
    "Lottery Winner": "I knew of a young man who won the lottery. He used to ride lawn mowers. After he won, he went on to using drugs. He blew a lot of money. Eventually his winnings were revoked after a dui.",
    "Broken Ankle": "I was walking to school. Since I wasn't looking at my feet, I stepped on a rock. I landed on the ground in pain. Thankfully, a stranger rushed to pick me up. He took me to the hospital to seek treatment.",
    "Jeff Moving": "Jeff wanted to move out of his house. He had no money to pay for a new one. One day he bought a scratching ticket. He won enough money for a down payment. Jeff ended up moving to a new house.",
    "Hearing Voices": "Bob walked into the ship elevator and heard a voice on the speaker. He told his wife the voice sounded the same as his audio book. They were surprised to hear the voice in the hallway to their cabin. The voice could still be in their cabin, so they called the steward. The steward found that the voice was coming from Bob's pocket phone.",
}

# run = wandb.init()
# artifact = run.use_artifact('amru/template-tests/experiment-ckpts:v10', type='checkpoints')
# artifact_dir = artifact.download()

exp = ExpanderModel.load_from_checkpoint("./.artifacts/experiment-ckpts:v10/expander-06-14_14-33-epoch=04.ckpt")
# exp = ExpanderModel(model_name_or_path='facebook/bart-base')

exp_examples = list(data_dict.keys())
generated_stories = exp.generate(exp_examples)
print(generated_stories)

print("=" * 100)

comp = CompressorModel.load_from_checkpoint("./.artifacts/experiment-ckpts:v9/compressor-06-14_12-54-epoch=01.ckpt")

comp_examples = list(data_dict.values())
generated_summaries = comp.generate(comp_examples)
print(generated_summaries)
