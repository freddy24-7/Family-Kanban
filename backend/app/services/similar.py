"""Planning suggestions: for each backlog topic, the most similar earlier tasks of the
same household: who did it, did it get done, how big was it really.
See docs/learning/07-embeddings.md.

Retrieval is TF-IDF (character n-grams), fitted per topic on that topic plus the
household's reviewed tasks: a household has hundreds of tickets, so this takes
milliseconds and always reflects edits and deletions. It beat a multilingual sentence
embedding in the Phase 8 comparison (ml/retrieval_eval.py, same function as here).

The suggestions are history for the planner to judge, shown next to the model's guess:
overall the vote of similar tasks ties the effort model, but it beat it on the drifted
category (groceries after the move) in the simulated drift world.
"""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.models import Household, Topic
from app.schemas import SimilarTaskRead, TopicSuggestionRead
from ml import retrieval


async def suggestions(
    session: AsyncSession, household: Household, topics: Sequence[Topic], k: int
) -> list[TopicSuggestionRead]:
    reviews = await repository.latest_reviews(session, household.id)
    if not topics or not reviews:
        return [TopicSuggestionRead(topic_id=t.id, similar=[]) for t in topics]
    names = {
        m.user_id: m.user.display_name for m in await repository.list_members(session, household.id)
    }
    result = []
    for topic in topics:
        candidates = [r for r in reviews if r.topic_id != topic.id]
        similarities = retrieval.tfidf_similarities(topic.text, [r.topic.text for r in candidates])
        hits = retrieval.top_k(similarities, k, retrieval.MIN_SIMILARITY)
        result.append(
            TopicSuggestionRead(
                topic_id=topic.id,
                similar=[
                    SimilarTaskRead(
                        topic_id=candidates[i].topic_id,
                        text=candidates[i].topic.text,
                        similarity=round(sim, 3),
                        assignee_name=names.get(candidates[i].assignee_id),
                        completed=candidates[i].completed,
                        effort_actual=candidates[i].effort_actual,
                        reviewed_at=candidates[i].reviewed_at,
                    )
                    for i, sim in hits
                ],
            )
        )
    return result
