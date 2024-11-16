from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from config import config

class Filter:
    def __init__(self, topic):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic = self.model.encode([topic])[0]

    @staticmethod
    def reformat_batch(batch):
        batch = batch.replace("-", " ").replace("...", " [...] ").replace("  ", " ")

        for i, char in enumerate(batch):
            if i != len(batch) - 1:
                if char == "." and batch[i + 2].islower():
                    batch = batch[:i] + " [...] " + batch[i + 2:]

        return batch if batch[0] != " " and batch[1] != " " else batch[1:-1]

    def assess_relevance_score(self, batch):
        """Checks the relevance of a batch to the topic."""
        return 1-distance.cosine(self.topic, self.model.encode([batch])[0])

    def process_and_filter_batch(self, batch):
        """Reformats and filters a batch for relevance before sending to summarizer."""
        reformatted_batch = self.reformat_batch(batch)
        if self.assess_relevance_score(reformatted_batch) >= config.filter_passing_percentage:
            return reformatted_batch
        return None
