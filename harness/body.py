"""Python side of the body: runs body/body.js as a child process and talks JSON lines to it."""
import itertools, json, os, queue, socket, subprocess, threading

HERE = os.path.dirname(os.path.abspath(__file__))
BODY_JS = os.path.join(HERE, "body", "body.js")


class BodyError(RuntimeError):
    pass


class Body:
    def __init__(self, name="Jev", host="127.0.0.1", port=25565, version="1.11.2", camera_port=None,
                 browser=None, ready_timeout=60, camera_size=None, camera_fps=0, camera_gpu=False, camera_kind="browser",
                 camera_view=None):
        cmd = ["node", BODY_JS, f"--name={name}", f"--host={host}", f"--port={port}", f"--version={version}"]
        if camera_kind == "vox":
            cmd.append("--camera=vox")   # the CPU ray-caster: no browser, rendered on demand in a few ms
        elif camera_port:
            cmd.append(f"--camera-port={camera_port}")
        if camera_size:
            cmd.append(f"--camera-size={camera_size}")
        if camera_fps:
            cmd.append(f"--camera-fps={camera_fps}")   # render continuously: a snapshot is the latest frame
        if camera_gpu:
            cmd.append("--camera-gpu=1")
        if camera_view:
            cmd.append(f"--camera-view={camera_view}")   # chunks the browser camera draws round the bot (default 4)
        if browser:
            cmd.append(f"--browser={browser}")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.replies = queue.Queue()
        self.events = queue.Queue()
        self.ids = itertools.count(1)
        threading.Thread(target=self._read, daemon=True).start()
        ev = self._event(ready_timeout)
        while ev.get("event") == "death":  # saved dead or mid-fall last time: it respawns and then is ready
            ev = self._event(ready_timeout)
        if ev.get("event") != "ready":
            raise BodyError(f"body did not spawn: {ev}")
        self.name = ev["name"]
        self.camera = ev.get("camera")  # viewer URL, or None
        if (camera_port or camera_kind == "vox") and not self.camera:
            raise BodyError(f"camera did not start: {ev.get('camera_error')}")

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            (self.events if "event" in msg else self.replies).put(msg)
        self.events.put({"event": "end", "reason": "body process exited"})
        self.replies.put({"id": None, "ok": False, "error": "body process exited"})

    def _event(self, timeout):
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return {"event": "timeout"}

    def call(self, cmd, timeout=30, **params):
        rid = next(self.ids)
        try:
            self.proc.stdin.write(json.dumps({"id": rid, "cmd": cmd, **params}) + "\n")
        except BrokenPipeError:
            raise BodyError("body process exited")
        while True:
            try:
                msg = self.replies.get(timeout=timeout)
            except queue.Empty:
                raise BodyError(f"{cmd}: no reply in {timeout}s")
            if msg["id"] in (rid, None):
                break
        if not msg["ok"]:
            raise BodyError(f"{cmd}: {msg['error']}")
        return msg["result"]

    def lost(self):
        """Why the bot left the server (kicked / disconnected), or None while it is still in."""
        while not self.events.empty():
            ev = self.events.get()
            if ev.get("event") in ("end", "kicked"):
                return ev.get("reason")
        return None

    def observe(self, since=0):
        return self.call("observe", since=since)

    def act(self, action, **params):
        return self.call("act", action=action, **params)

    def say(self, text):
        return self.call("say", text=text)

    def landmarks(self, **params):
        """Clusters of uncommon blocks around the bot: [{name, count, min, max, center}]."""
        return self.call("landmarks", timeout=60, **params)["clusters"]

    def snapshot(self, yaw_offset=0.0):
        """First-person view as a JPEG data URI; yaw_offset (radians, + = left) turns only the camera, so
        math.pi gives a rear view while the bot keeps its heading."""
        r = self.call("snapshot", yaw_offset=yaw_offset)
        self.last_frame_age_ms = r.get("age_ms")   # continuous camera: how old the picture is
        return r["image"]

    def snapshot_zoom(self, zoom):
        """The view and a zoomed view of its middle (field of view `zoom` degrees, like a scope), rendered from the
        same moment: (image, zoomed image). Needs the vox camera."""
        r = self.call("snapshot", zoom=zoom)
        return r["image"], r["zoom"]

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
