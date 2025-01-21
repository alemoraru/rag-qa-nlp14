import argparse
import json
import logging
import os
import pickle
import sys
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from DRhard_utils.adore.inference import evaluate
from DRhard_utils.model import RobertaDot
from DRhard_utils.retrieve_utils import (
    construct_flatindex_from_embeddings,
    convert_index_to_gpu,
    index_retrieve,
)
from tqdm import tqdm
from transformers import RobertaConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO,
)


class AdoreInferencePipeline:
    def __init__(self, config):
        """
        Initialize the Adore Inference pipeline.
        """
        self.config = config
        self.use_gpu = torch.cuda.is_available() and not config.no_cuda
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

    def _load_model(self):
        """Load the Adore model."""
        config = RobertaConfig.from_pretrained(self.config.model_dir)
        model = RobertaDot.from_pretrained(self.config.model_dir, config=config)
        model.to(self.device)
        logger.info("Model loaded and moved to device: %s", self.device)
        return model

    def load_embeddings(self, path, embedding_size):
        return np.memmap(path, dtype=np.float32, mode="r").reshape(-1, embedding_size)

    def run_inference(self):
        """Run inference"""
        model = self._load_model()

        args = SimpleNamespace(
            **vars(self.config),
            model_device=self.device,
            qmemmap_path=os.path.join(
                self.config.output_dir, f"{self.config.mode}.qembed.memmap"
            ),
            n_gpu=1,
        )

        logger.info("Starting evaluation with args: %s", args)
        evaluate(args, model)

        doc_embeddings = self.load_embeddings(
            self.config.dmemmap_path, model.output_embedding_size
        )
        query_embeddings = self.load_embeddings(
            args.qmemmap_path, model.output_embedding_size
        )

        # Construct FAISS index
        index = construct_flatindex_from_embeddings(doc_embeddings, None)
        faiss.omp_set_num_threads(32)
        nearest_neighbors = index_retrieve(index, query_embeddings, args.topk, batch=32)

        self.save_rankings(nearest_neighbors)
        logger.info("Inference done.")

    def save_rankings(self, nearest_neighbors):
        """Save the ranking."""
        output_file = os.path.join(
            self.config.output_dir, f"{self.config.mode}.rank.tsv"
        )
        with open(output_file, "w") as f:
            for qid, neighbors in enumerate(nearest_neighbors):
                for rank, pid in enumerate(neighbors, start=1):
                    f.write(f"{qid}\t{pid}\t{rank}\n")

    def reverse_adore_id_mapping(self):
        """Reverse ADORE ID mapping (adapted from DRhard)."""
        input_path = os.path.join(
            self.config.output_dir, f"{self.config.mode}.rank.tsv"
        )
        output_path = os.path.join(
            self.config.output_dir, f"{self.config.mode}.mapped.rank.tsv"
        )

        pid2offset = self.load_pickle("pid2offset.pickle")
        qid2offset = self.load_pickle(f"{self.config.mode}-qid2offset.pickle")

        offset2pid = {v: k for k, v in pid2offset.items()}
        offset2qid = {v: k for k, v in qid2offset.items()}

        self.map_and_save_ids(input_path, output_path, offset2qid, offset2pid)

    def load_pickle(self, filename):
        """Load a pickle file."""
        with open(os.path.join(self.config.preprocess_dir, filename), "rb") as f:
            return pickle.load(f)

    def map_and_save_ids(self, input_path, output_path, offset2qid, offset2pid):
        """Map and save IDs to a file."""
        with open(output_path, "w") as output:
            for line in tqdm(open(input_path)):
                qid, pid, rank = line.split()
                qid, pid, rank = int(qid), int(pid), int(rank)
                qid, pid = offset2qid[qid], offset2pid[pid]
                output.write(f"{qid}\t{pid}\t{rank}\n")

    def convert_ids_to_orig_and_save(self):
        """Convert query ids to the original ones."""
        rank_file = os.path.join(
            self.config.output_dir, f"{self.config.mode}.mapped.rank.tsv"
        )
        mappings_path = (
            f"adore_data/dataset_mapped/query_ids_mapping.{self.config.mode}"
        )

        with open(mappings_path, "rb") as f:
            mappings = json.load(f)

        result_dict = self.map_to_original_ids(rank_file, mappings)

        output_file = os.path.join(
            self.config.output_dir, f"{self.config.mode}.mapped.orig.rank.json"
        )
        with open(output_file, "w") as f:
            json.dump(result_dict, f)

    def map_to_original_ids(self, rank_file, mappings):
        """Map to original IDs."""
        result_dict = {}
        with open(rank_file, "r") as f:
            for line in f:
                qid, pid, rank = line.strip().split()
                key = mappings[qid]
                if key not in result_dict:
                    result_dict[key] = {}
                result_dict[key][pid] = rank

        return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="The data split to load",
    )
    args = parser.parse_args()

    config = SimpleNamespace(
        model_dir="models/adore_model_finetuned_v2/epoch-6",
        output_dir="adore_data/passage/evaluate_rel",
        preprocess_dir="adore_data/passage/preprocess_rel",
        mode=args.mode,  # "test"
        topk=15,
        dmemmap_path="star_embeddings/passages.memmap",
        pergpu_eval_batch_size=32,
        max_seq_length=64,
        eval_batch_size=32,
        no_cuda=True,
    )

    pipeline = AdoreInferencePipeline(config)
    # Run inference
    pipeline.run_inference()

    # Convert QIDs and PIDS mapped by ADORE back
    pipeline.reverse_adore_id_mapping()

    # Map query ids back from integers
    pipeline.convert_ids_to_orig_and_save()
