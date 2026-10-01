"""Client for the djev /api/evaluate endpoint (NanoJev contract), plus a mock for offline runs."""
import base64, json, random, time, urllib.error, urllib.request


class JevError(RuntimeError):
    pass


def _by_state(resp):
    """{state_id: answers} from an evaluate response."""
    for key in ("states", "results"):
        if isinstance(resp.get(key), list):
            return {s["id"]: s["answers"] for s in resp[key]}
    raise JevError(f"unexpected response shape: {json.dumps(resp)[:300]}")


class Jev:
    def __init__(self, url, auth=None, timeout=20):
        self.url = url.rstrip("/") + "/api/evaluate"
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
        if auth:
            self.headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()

    def evaluate(self, states):
        """states: [{"id", "state", "questions"}] -> ({state_id: answers}, raw response)."""
        body = json.dumps({"states": states}).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            raise JevError(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise JevError(str(e))
        return _by_state(resp), resp


class OfficialJev:
    """The official Jev (TypeSafe System One, POST /v1/systemone, Bearer key) behind the same evaluate() as djev:
    one request per state (in parallel when there are several), our question types mapped to theirs (boolean ->
    noul), and their answers back in djev's shape. The key comes from TYPESAFE_API_KEY (.env)."""

    URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, key, model="jev-latest", timeout=20):
        self.model, self.timeout = model, timeout
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    @staticmethod
    def _question(q):
        out = {k: v for k, v in q.items() if k in ("instructions", "criteria")}
        out["type"] = "noul" if q["type"] == "boolean" else q["type"]
        return out

    @staticmethod
    def _answer(q, a):
        if q["type"] == "boolean":
            p = a["noul"]
            return {"type": "boolean", "probabilities": {"true": p, "false": 1 - p}, "p_true": p, "value": p >= 0.5}
        return {**a, "value": a.get("choice", a.get("score"))}

    def _one(self, s):
        body = {"model": self.model, "state": s["state"],
                "questions": {qid: self._question(q) for qid, q in s["questions"].items()}}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), headers=self.headers, method="POST")
        for attempt in range(3):
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and attempt < 2:
                    time.sleep(0.2 * 2 ** attempt)
                    continue
                raise JevError(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                raise JevError(str(e))
        answers = {qid: self._answer(q, resp["answers"][qid]) for qid, q in s["questions"].items()}
        return answers, {"model": resp.get("model"), "usage": resp.get("usage"), "seconds": round(time.time() - t0, 3)}

    def evaluate(self, states):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, len(states))) as ex:
            done = list(ex.map(self._one, states))
        by_state = {s["id"]: a for s, (a, _) in zip(states, done)}
        return by_state, {"states": [{"id": s["id"], "answers": a} for s, (a, _) in zip(states, done)],
                          "execution": {"official": [x for _, x in done]}}


class MockJev:
    """Random distributions in the real response shape; for testing the loop without the GPU server.

    Booleans answer p_true[qid] (default 0.1), so e.g. every chat message counts as an instruction."""

    def __init__(self, p_true=None, delay=0.2):
        self.p_true = {"is_instruction": 0.9, **(p_true or {})}
        self.delay = delay

    def _answer(self, qid, q):
        if q["type"] == "boolean":
            p = self.p_true.get(qid, 0.1)
            return {"type": "boolean", "probabilities": {"true": p, "false": 1 - p}, "p_true": p,
                    "value": p >= 0.5, "candidate_mass": 1.0}
        keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else list(range(len(q["criteria"])))
        w = [random.random() for _ in keys]
        probs = {k: x / sum(w) for k, x in zip(keys, w)}
        best = max(probs, key=probs.get)
        return {"type": q["type"], "probabilities": probs, "choice": best, "value": best, "candidate_mass": 1.0}

    def evaluate(self, states):
        time.sleep(self.delay)
        resp = {"states": [{"id": s["id"], "answers": {qid: self._answer(qid, q) for qid, q in s["questions"].items()}}
                           for s in states],
                "execution": {"mock": True, "forward_passes": 1}}
        return _by_state(resp), resp
