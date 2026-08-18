"""Candidate: coarse-to-fine retrieval with temporal de-duplication.

Hypothesis: Uniform 32-frame sampling wastes budget on irrelevant segments of
60+ minute videos. Meanwhile, pure top-k retrieval (keyframe_image_rag with k=6)
shows too few frames and returns near-duplicates from the same moment. By
ingesting 64 frames with CLIP embeddings, retrieving the top-16 by text→image
similarity, then clustering by temporal proximity and selecting diverse frames
across clusters, we get BOTH relevance AND temporal coverage. The VLM sees 12
highly relevant frames from distinct moments rather than 32 mostly-irrelevant
uniform frames or 6 potentially-redundant retrieved frames.

Axes: E (coarse-to-fine retrieval: first find relevant regions, then pick diverse
frames within them) + F (budget packing: 12 well-chosen frames instead of 32
uniform ones).
"""

from typing import Any

import numpy as np

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream


PROMPT = (
    "You are answering a multiple-choice question about a long video "
    "({dur:.0f}s duration).\n"
    "You are shown {k} frames retrieved as most relevant to the question, "
    "drawn from {n_clusters} distinct temporal segments.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Think step by step about what you see, then end with: ANSWER: <letter>\n"
)

TAIL = "\n\nAfter your reasoning, write your final answer as: ANSWER: <letter>"


class CoarseToFineRetrieval(VideoMemoryHarness):
    """CLIP retrieval + temporal clustering for diverse relevant frame selection."""

    NUM_INGEST = 340
    RETRIEVE_K = 16
    NUM_CLUSTERS = 6
    FRAMES_PER_CLUSTER = 2
    CLUSTER_GAP_SECONDS = 30.0

    def build_memory(self, video: VideoStream) -> Any:
        frames = video.sample_uniform(self.NUM_INGEST)
        images = [f.image if f.image is not None else f.caption for f in frames]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return {
            "frames": frames,
            "img_emb": img_emb,
            "duration": video.duration,
        }

    def _temporal_cluster(self, frames: list[Frame]) -> list[list[Frame]]:
        """Group frames into temporal clusters based on timestamp proximity."""
        if not frames:
            return []
        sorted_frames = sorted(frames, key=lambda f: f.timestamp)
        clusters: list[list[Frame]] = [[sorted_frames[0]]]
        for f in sorted_frames[1:]:
            if f.timestamp - clusters[-1][-1].timestamp <= self.CLUSTER_GAP_SECONDS:
                clusters[-1].append(f)
            else:
                clusters.append([f])
        return clusters

    def _select_from_clusters(
        self, clusters: list[list[Frame]], query_vec, all_frames: list[Frame], img_emb
    ) -> list[Frame]:
        """Select diverse frames across top clusters."""
        frame_to_idx = {id(f): i for i, f in enumerate(all_frames)}
        sims_all = np.asarray(img_emb) @ np.asarray(query_vec).reshape(-1)

        cluster_scores: list[tuple[float, int]] = []
        for ci, cluster in enumerate(clusters):
            indices = [frame_to_idx[id(f)] for f in cluster if id(f) in frame_to_idx]
            if indices:
                cluster_scores.append((float(np.max(sims_all[indices])), ci))

        cluster_scores.sort(reverse=True)
        top_clusters = cluster_scores[:self.NUM_CLUSTERS]

        selected: list[Frame] = []
        for _, ci in top_clusters:
            cluster = clusters[ci]
            indices = [frame_to_idx[id(f)] for f in cluster if id(f) in frame_to_idx]
            if not indices:
                continue
            csims = sims_all[indices]
            top_in_cluster = np.argsort(-csims)[:self.FRAMES_PER_CLUSTER]
            for idx in top_in_cluster:
                selected.append(cluster[idx])

        return sorted(selected, key=lambda f: f.timestamp)

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        frames: list[Frame] = memory["frames"]
        img_emb = memory["img_emb"]
        duration: float = memory["duration"]

        qv = self.embed_texts([question])[0]

        top_idxs = self.topk_indices(qv, img_emb, self.RETRIEVE_K)
        retrieved = [frames[i] for i in top_idxs]

        clusters = self._temporal_cluster(retrieved)
        selected = self._select_from_clusters(clusters, qv, frames, img_emb)

        if not selected:
            selected = retrieved[:self.NUM_CLUSTERS * self.FRAMES_PER_CLUSTER]

        parts = [
            {
                "type": "text",
                "text": PROMPT.format(
                    dur=duration,
                    k=len(selected),
                    n_clusters=min(len(clusters), self.NUM_CLUSTERS),
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        parts += self.render_frames(selected)
        parts.append({"type": "text", "text": TAIL})

        resp = self.ask_vlm(parts)

        import re
        m = re.search(r"ANSWER\s*:\s*\(?([A-Za-z])\)?", resp, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            allowed = {chr(ord("A") + i) for i in range(len(options))}
            if letter in allowed:
                return letter, {
                    "num_retrieved": len(retrieved),
                    "num_clusters": len(clusters),
                    "num_shown": len(selected),
                    "raw": resp[:200],
                }
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        return letter, {
            "num_retrieved": len(retrieved),
            "num_clusters": len(clusters),
            "num_shown": len(selected),
            "raw": resp[:200],
        }
