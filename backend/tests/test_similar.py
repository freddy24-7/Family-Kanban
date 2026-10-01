"""Phase 8: similar-task retrieval for planning, and the embedding experiment code."""

import re

import numpy as np
import pytest

from ml import embeddings, retrieval
from tests.helpers import create_household, register_and_login

# --- pure functions ------------------------------------------------------------------


def test_embed_is_normalised_and_cached():
    calls = []

    def backend(texts):
        calls.append(list(texts))
        return np.array([[3.0, 4.0] + [0.0] * (embeddings.DIM - 2) for _ in texts])

    embeddings.set_backend(backend)
    vectors = embeddings.embed(["a", "b", "a"])
    assert vectors.shape == (3, embeddings.DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert calls == [["a", "b"]]  # duplicates embedded once
    embeddings.embed(["b", "c"])
    assert calls[-1] == ["c"]  # "b" came from the cache
    assert embeddings.embed([]).shape == (0, embeddings.DIM)


def test_top_k_and_vote():
    sims = np.array([0.2, 0.9, 0.5, 0.9])
    assert retrieval.top_k(sims, 3) == [(1, pytest.approx(0.9)), (3, pytest.approx(0.9)), (2, 0.5)]
    assert [i for i, _ in retrieval.top_k(sims, 5, min_similarity=0.5)] == [1, 3, 2]
    assert retrieval.top_k(np.zeros(0), 3) == []

    order = ["S", "M", "L"]
    votes = retrieval.weighted_vote(["M", "S", "L", None], [0.9, 0.6, 0.2, 0.9], order)
    assert votes == pytest.approx({"S": 0.4, "M": 0.6, "L": 0.0})  # L below threshold
    assert retrieval.weighted_vote(["S"], [0.1], order) is None
    assert retrieval.precision_at_k([True, False, True, True], 3) == pytest.approx(2 / 3)
    assert retrieval.precision_at_k([True], 5) == 1.0
    assert retrieval.precision_at_k([], 5) is None


def test_tfidf_finds_the_same_kind_of_task():
    sims = retrieval.tfidf_similarities(
        "gras maaien achtertuin", ["melk halen supermarkt", "gras maaien voortuin"]
    )
    assert sims.argmax() == 1 and sims[1] >= retrieval.MIN_SIMILARITY > sims[0]
    assert retrieval.tfidf_similarities("x", []).shape == (0,)
    assert np.allclose(np.linalg.norm(retrieval.tfidf_vectors(["a b", "c"]), axis=1), 1.0)


# --- planning suggestions through the API ------------------------------------------------


async def _reviewed_history(client, outbox):
    """A family that finished one sprint: Sem mowed the grass (M), Mama got groceries (S),
    the gutter wasn't finished. Then new backlog topics arrive."""
    parent = await register_and_login(client, "mama@example.com", "Mama")
    h = await create_household(client, parent, "Familie Test")
    await client.post(
        f"/households/{h}/invites",
        json={"email": "sem@example.com", "is_child": True},
        headers=parent,
    )
    token = re.search(r"invite\?token=(\S+)", outbox[-1]["body"]).group(1)
    kid = await register_and_login(client, "sem@example.com", "Sem")
    await client.post("/invites/accept", json={"token": token}, headers=kid)
    members = (await client.get(f"/households/{h}", headers=parent)).json()["members"]
    ids = {m["display_name"]: m["user_id"] for m in members}

    async def topic(text):
        r = await client.post(f"/households/{h}/topics", json={"text": text}, headers=parent)
        return r.json()["id"]

    done = {
        "gras maaien voortuin": ("Sem", True, "M"),
        "melk halen supermarkt": ("Mama", True, "S"),
        "dakgoot schoonmaken": ("Mama", False, None),
    }
    sprint = (
        await client.post(
            f"/households/{h}/sprints",
            json={"name": "Week 40", "start_date": "2026-09-28", "end_date": "2026-10-04"},
            headers=parent,
        )
    ).json()
    s = f"/households/{h}/sprints/{sprint['id']}"
    items = {}
    for text, (who, _, _) in done.items():
        r = await client.post(
            f"{s}/items",
            json={"topic_id": await topic(text), "assignee_id": ids[who]},
            headers=parent,
        )
        items[text] = r.json()["id"]
    await client.post(f"{s}/start", headers=parent)
    for text, (_, completed, effort) in done.items():
        body = {"completed": completed, **({"effort_actual": effort} if effort else {})}
        r = await client.put(f"{s}/items/{items[text]}/review", json=body, headers=parent)
        assert r.status_code == 200, r.text
    assert (await client.post(f"{s}/complete", json={}, headers=parent)).status_code == 200

    new = {text: await topic(text) for text in ["gras maaien achtertuin", "paspoort verlengen"]}
    return {"parent": parent, "kid": kid, "h": h, "new": new}


async def test_backlog_suggestions(client, outbox):
    f = await _reviewed_history(client, outbox)
    r = await client.get(f"/households/{f['h']}/backlog/suggestions", headers=f["kid"])
    assert r.status_code == 200, r.text
    by_topic = {t["topic_id"]: t["similar"] for t in r.json()}
    assert len(by_topic) == 3  # every backlog topic, also without suggestions

    mow = by_topic[f["new"]["gras maaien achtertuin"]]
    assert mow[0]["text"] == "gras maaien voortuin"
    assert mow[0]["assignee_name"] == "Sem" and mow[0]["effort_actual"] == "M"
    assert mow[0]["completed"] is True
    assert all(s["similarity"] >= retrieval.MIN_SIMILARITY for s in mow)

    # Nothing similar was ever done: no suggestion rather than a bad one.
    assert by_topic[f["new"]["paspoort verlengen"]] == []

    # The unfinished gutter is back in the backlog; it doesn't suggest itself.
    gutter = next(v for t, v in by_topic.items() if t not in f["new"].values())
    assert all(s["text"] != "dakgoot schoonmaken" for s in gutter)


async def test_suggestions_follow_edits_and_deletes(client, outbox):
    f = await _reviewed_history(client, outbox)
    url = f"/households/{f['h']}/backlog/suggestions"
    topic_id = f["new"]["paspoort verlengen"]
    await client.patch(
        f"/households/{f['h']}/topics/{topic_id}",
        json={"text": "melk halen bakker"},
        headers=f["parent"],
    )
    by_topic = {
        t["topic_id"]: t["similar"] for t in (await client.get(url, headers=f["parent"])).json()
    }
    assert by_topic[topic_id][0]["text"] == "melk halen supermarkt"

    await client.delete(f"/households/{f['h']}/topics/{topic_id}", headers=f["parent"])
    topics = {t["topic_id"] for t in (await client.get(url, headers=f["parent"])).json()}
    assert topic_id not in topics


async def test_suggestions_stay_inside_the_household(client, outbox):
    f = await _reviewed_history(client, outbox)
    other = await register_and_login(client, "other@example.com", "Ander")
    h2 = await create_household(client, other, "Andere familie")
    await client.post(
        f"/households/{h2}/topics", json={"text": "gras maaien achtertuin"}, headers=other
    )
    body = (await client.get(f"/households/{h2}/backlog/suggestions", headers=other)).json()
    # Familie Test's mowing history is not visible to another household.
    assert [t["similar"] for t in body] == [[]]
    denied = await client.get(f"/households/{f['h']}/backlog/suggestions", headers=other)
    assert denied.status_code == 404
