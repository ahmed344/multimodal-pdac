"""Compare directly against the original model at the specified baseline commit."""
from pathlib import Path
import importlib.util
import json
import subprocess
import tempfile
import torch

from dann.config import load_config
from dann.model import AdversarialLatentFusion
from dann.losses import ZILNLoss
from dann.targets import target_contract
from dann.train import load_training_checkpoint


def main():
    torch.set_num_threads(2)
    config = load_config(Path('dann/config.yaml'))
    m = config['model']
    for key in ('spectral_encoder', 'aggregation', 'heads'):
        m.pop(key)
    m.update(num_peaks=5, embedding_dim=3, peak_hidden_dims=[7, 6], peak_output_dim=6,
             aggregation_hidden_dims=[6, 6], latent_dim=6, biology_hidden_dims=[6, 5, 4],
             discriminator_hidden_dims=[6, 5, 4], dropout=0.)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'baseline.py'
        path.write_bytes(subprocess.check_output(['git', 'show', '256dd33:dann/model.py']))
        spec = importlib.util.spec_from_file_location('baseline_model', path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        torch.manual_seed(51)
        old = baseline.AdversarialLatentFusion.from_config(config, 2, 4)
        torch.manual_seed(51)
        new = AdversarialLatentFusion.from_config(config, 2, 4)
        for key, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
        assert list(old.state_dict()) == list(new.state_dict())
        assert [n for n, _ in old.named_parameters()] == [n for n, _ in new.named_parameters()]
        batch = dict(peak_indices=torch.tensor([0,2,1,4,3,1]), intensities=torch.rand(6),
                     sample_indices=torch.tensor([0,0,2,2,2,3]), peak_counts=torch.tensor([2,0,3,1]))
        targets = torch.tensor([[0., .1, .3, .2], [.2, .1, 0., .5], [0., 1., .3, 0.], [.4, .1, .6, .3]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        mask[2,0] = False
        values_old, values_new = old(batch), new(batch)
        max_prediction = max(float((values_old[k]-values_new[k]).abs().max().detach()) for k in values_old)
        loss_fn = ZILNLoss.from_config(config)
        losses = []
        for model, values in ((old, values_old), (new, values_new)):
            loss = loss_fn(values['pi_logits'], values['mu'], values['sigma'], targets, mask).total
            loss += torch.nn.functional.cross_entropy(values['batch_logits'], torch.tensor([0,1,0,1]))
            losses.append(float(loss.detach()))
            loss.backward()
        assert losses[0] == losses[1]
        max_gradient = max(float((a.grad-b.grad).abs().max()) for a,b in zip(old.parameters(),new.parameters()))
        assert max_prediction == 0 and max_gradient == 0
        optimizer = torch.optim.AdamW(old.parameters())
        optimizer.step()
        checkpoint = Path(temporary)/'legacy.pt'
        torch.save(dict(config=config, model_state=old.state_dict(), optimizer_state=optimizer.state_dict(),
                        epoch=0, target_columns=config['data']['target_columns'], target_transform=target_contract(config)), checkpoint)
        new.load_state_dict(torch.load(checkpoint, weights_only=False)['model_state'])
        for k, v in old(batch).items():
            torch.testing.assert_close(v, new(batch)[k], rtol=0, atol=0)
        print(json.dumps(dict(baseline='256dd33', seeded_initialization_identical=True, max_prediction_difference=max_prediction,
                              max_gradient_difference=max_gradient, loss_difference=abs(losses[0]-losses[1]),
                              legacy_weight_loading=True)))


if __name__ == '__main__':
    main()
