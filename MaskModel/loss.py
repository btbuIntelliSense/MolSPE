import torch
import torch.nn.functional as F
from torch import nn


class MolLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.padding_idx = args.pad_idx
        self.seed = args.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.args = args  # Assuming task.args contains the necessary hyperparameters

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()

        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            encoder_rep,
            new_encoder_rep,
        ) = model(**sample[input_key], encoder_masked_tokens=masked_tokens)

        target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = target[masked_tokens]

        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )

        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size

        loss = masked_token_loss * self.args.masked_token_loss

        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1) * sample[target_key]["tokens_target"].size(0),
            "masked_token_loss": masked_token_loss.item(),
            "masked_token_hit": masked_hit.item(),
            "masked_token_cnt": masked_cnt.item(),
        }

        if encoder_coord is not None:
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss += masked_coord_loss * self.args.masked_coord_loss
            logging_output["masked_coord_loss"] = masked_coord_loss.item()

        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss += masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.item()

        # if self.args.x_norm_loss > 0 and x_norm is not None:
        #     loss += self.args.x_norm_loss * x_norm
        #     logging_output["x_norm_loss"] = x_norm.item()

        # if self.args.delta_pair_repr_norm_loss > 0 and delta_encoder_pair_rep_norm is not None:
        #     loss += self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
        #     logging_output["delta_pair_repr_norm_loss"] = delta_encoder_pair_rep_norm.item()

        logging_output["loss"] = loss.item()
        return loss,new_encoder_rep

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid"):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)

        print(f"Loss: {loss_sum / sample_size:.3f}")
        print(f"Sequence Length: {seq_len / bsz:.3f}")
        print(f"Masked Token Loss: {sum(log.get('masked_token_loss', 0) for log in logging_outputs) / sample_size:.3f}")
        print(
            f"Masked Accuracy: {sum(log.get('masked_token_hit', 0) for log in logging_outputs) / sum(log.get('masked_token_cnt', 0) for log in logging_outputs):.3f}")
        print(f"Masked Coord Loss: {sum(log.get('masked_coord_loss', 0) for log in logging_outputs) / sample_size:.3f}")
        print(f"Masked Dist Loss: {sum(log.get('masked_dist_loss', 0) for log in logging_outputs) / sample_size:.3f}")
        print(f"X Norm Loss: {sum(log.get('x_norm_loss', 0) for log in logging_outputs) / sample_size:.3f}")
        print(
            f"Delta Pair Repr Norm Loss: {sum(log.get('delta_pair_repr_norm_loss', 0) for log in logging_outputs) / sample_size:.3f}")

    @staticmethod
    def logging_outputs_can_be_summed(is_train):
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][dist_masked_tokens]

        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = sample["net_input"]["src_tokens"].ne(self.padding_idx)
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)

        if normalize:
            masked_distance_target = (masked_distance_target.float() - self.dist_mean) / self.dist_std

        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss