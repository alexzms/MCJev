"""Long-term memory for harness v2, after Claude Code's memory directory.

One markdown file per memory with a frontmatter header, plus a MEMORY.md index. Jev cannot write, so the
harness writes every entry, from what humans say (an explicit "remember ...", or praise / correction right
after an instruction ends). Recall asks Jev one yes/no question per memory, all in one forward pass, and keeps
the best few. Only things the bot cannot observe for itself are kept: named places, people, feedback, facts.
"""
import os, re, time

TYPES = {
    "place": "a named place and where it is",
    "person": "who someone is or what they like",
    "feedback": "how a human wants things done: praise or a correction",
    "fact": "anything else worth keeping for later",
}
MAX_RECALLED = 5
MAX_SCANNED = 60  # newest first; a state holds at most 63 questions


def age_words(t, now=None):
    """Models are bad at date arithmetic; say how long ago in words, like Claude Code's memoryAge."""
    s = max(0, (now or time.time()) - t)
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{s / 60:.0f} min ago"
    if s < 86400:
        return f"{s / 3600:.0f} h ago"
    days = s / 86400
    return "yesterday" if days < 2 else f"{days:.0f} days ago"


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "memory"


class MemoryStore:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def entries(self):
        out = []
        os.makedirs(self.root, exist_ok=True)  # may have been wiped (the benchmarks clear it between episodes)
        for name in os.listdir(self.root):
            if not name.endswith(".md") or name == "MEMORY.md":
                continue
            path = os.path.join(self.root, name)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
            if not m:
                continue
            meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
            meta.update(file=name, body=m.group(2).strip(), created=float(meta.get("created", 0)))
            if meta.get("pos"):
                x, y, z = (float(v) for v in meta["pos"].split())
                meta["pos"] = {"x": x, "y": y, "z": z}
            out.append(meta)
        return sorted(out, key=lambda e: -e["created"])[:MAX_SCANNED]

    def add(self, type, description, body, by, pos=None):
        assert type in TYPES
        base = f"{type}_{_slug(description)}"
        path = os.path.join(self.root, base + ".md")
        n = 2
        while os.path.exists(path):
            path = os.path.join(self.root, f"{base}_{n}.md")
            n += 1
        head = [f"type: {type}", f"description: {description}", f"by: {by}", f"created: {time.time():.0f}"]
        if pos:
            head.append(f"pos: {pos['x']:.1f} {pos['y']:.1f} {pos['z']:.1f}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("---\n" + "\n".join(head) + "\n---\n" + body.strip() + "\n")
        self.write_index()
        return os.path.basename(path)

    def write_index(self):
        lines = [f"- [{e['type']}] [{e['description']}]({e['file']}) — by {e.get('by', '?')}, "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(e['created']))}" for e in self.entries()]
        with open(os.path.join(self.root, "MEMORY.md"), "w", encoding="utf-8") as f:
            f.write("# Memory index (written by the harness; one file per memory)\n\n" + "\n".join(lines) + "\n")

    def places(self):
        return [e for e in self.entries() if e["type"] == "place" and e.get("pos")]


RECALL_CHUNK = 10  # memories per state: all their answers must fit in half of Jev's canvas


def recall_requests(entries, instruction, sender):
    """States (ids recall0, recall1, ...) whose questions ask, per memory, whether it helps with this instruction;
    split so that no state has more answers than Jev's canvas holds. Send them all in one request."""
    state = (f'A Minecraft bot was just given this instruction by {sender}: "{instruction}". '
             f"Each question below is about one thing the bot remembers from earlier.")
    return [{"id": f"recall{k // RECALL_CHUNK}", "state": state,
             "questions": {f"m{i}": {"type": "boolean",
                                     "instructions": f"Would this memory help the bot carry out the instruction? "
                                                     f"Memory ({entries[i]['type']}, {age_words(entries[i]['created'])}): "
                                                     f"{entries[i]['description']}"}
                           for i in range(k, min(k + RECALL_CHUNK, len(entries)))}}
            for k in range(0, len(entries), RECALL_CHUNK)]


def pick_recalled(entries, answers, threshold=0.5):
    """answers: the whole response (all recall states)."""
    p = lambda i: answers[f"recall{i // RECALL_CHUNK}"][f"m{i}"]["p_true"]
    scored = sorted(((p(i), i) for i in range(len(entries))), reverse=True)
    return [dict(entries[i], p=q) for q, i in scored if q >= threshold][:MAX_RECALLED]


def memory_lines(recalled):
    if not recalled:
        return ["(none)"]
    lines = [f"- ({e['type']}, saved {age_words(e['created'])} by {e.get('by', '?')}) {e['description']}"
             for e in recalled]
    lines.append("Memories are what someone said earlier, not what you see now; they may be out of date.")
    return lines
