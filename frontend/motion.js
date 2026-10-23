// Motion sampling: mirrors sceneforge/library.py + director._transform so the
// blocking computed by the agent is exactly what is rendered.
export const LOOPING = new Set(["locomotion", "idle", "dance", "daily"]);
const BLEND = 0.28; // seconds of crossfade between beats

export class MotionStore {
  constructor(library) {
    this.lib = library;
    this.fps = library.fps;
    this.nj = library.joints.length;
    this.ji = Object.fromEntries(library.joints.map((n, i) => [n, i]));
    this.meta = Object.fromEntries(library.clips.map((c) => [c.id, c]));
    this.clips = new Map();
  }

  async load(ids) {
    const need = [...new Set(ids)].filter((id) => !this.clips.has(id));
    await Promise.all(need.map(async (id) => {
      const res = await fetch(`/motions/${id}.json`);
      if (!res.ok) throw new Error(`motion ${id}: HTTP ${res.status}`);
      const data = await res.json();
      const n = data.frames.length;
      const arr = new Float32Array(n * this.nj * 3);
      data.frames.forEach((fr, f) => { for (let k = 0; k < fr.length; k++) arr[f * this.nj * 3 + k] = fr[k] / 1000; });
      const L = this.ji.LeftUpLeg, R = this.ji.RightUpLeg;
      const yawAt = (f) => {
        const o = f * this.nj * 3;
        const ax = arr[o + L * 3] - arr[o + R * 3], az = arr[o + L * 3 + 2] - arr[o + R * 3 + 2];
        return Math.atan2(-az, ax);
      };
      // unwrapped yaw at the last frame relative to the first (same as library.root_track)
      let prev = yawAt(0), acc = prev;
      for (let f = 1; f < n; f++) {
        let y = yawAt(f);
        while (y - prev > Math.PI) y -= 2 * Math.PI;
        while (y - prev < -Math.PI) y += 2 * Math.PI;
        // keep continuity relative to the running value
        y = acc + wrap(y - acc);
        acc = y; prev = y;
      }
      const last = (n - 1) * this.nj * 3;
      this.clips.set(id, {
        id, frames: arr, n, duration: (n - 1) / this.fps,
        endX: arr[last], endZ: arr[last + 2], endYaw: acc - yawAt(0),
        loops: LOOPING.has(this.meta[id].category),
      });
    }));
  }

  // Writes joint positions (world space) for one beat at beat-local time t into out[nj*3].
  sampleBeat(beat, t, out) {
    const c = this.clips.get(beat.clip);
    let tau = Math.max(0, t) * beat.speed;
    if (!c.loops) tau = Math.min(tau, c.duration);
    const L = c.duration;
    const loops = L > 0 ? Math.floor(tau / L) : 0;
    let rem = tau - loops * L;
    if (!c.loops && tau >= L) rem = L;
    let ox = 0, oz = 0, oyaw = 0;
    for (let i = 0; i < (c.loops ? loops : 0); i++) {
      const cs = Math.cos(oyaw), sn = Math.sin(oyaw);
      ox += c.endX * cs + c.endZ * sn;
      oz += -c.endX * sn + c.endZ * cs;
      oyaw += c.endYaw;
    }
    const f = Math.min(rem * this.fps, c.n - 1);
    const i0 = Math.floor(f), i1 = Math.min(i0 + 1, c.n - 1), a = f - i0;
    const s0 = i0 * this.nj * 3, s1 = i1 * this.nj * 3;
    const cy = Math.cos(oyaw), sy = Math.sin(oyaw);
    const cb = Math.cos(beat.yaw), sb = Math.sin(beat.yaw);
    for (let j = 0; j < this.nj; j++) {
      const k = j * 3;
      let x = c.frames[s0 + k] * (1 - a) + c.frames[s1 + k] * a;
      const y = c.frames[s0 + k + 1] * (1 - a) + c.frames[s1 + k + 1] * a;
      let z = c.frames[s0 + k + 2] * (1 - a) + c.frames[s1 + k + 2] * a;
      // loop offset (clip-local), then beat placement (world)
      let lx = ox + x * cy + z * sy, lz = oz - x * sy + z * cy;
      out[k] = beat.x + lx * cb + lz * sb;
      out[k + 1] = y;
      out[k + 2] = beat.z - lx * sb + lz * cb;
    }
    return out;
  }
}

function wrap(a) { return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; }

// Per-character timeline with crossfades between consecutive beats.
export class Performer {
  constructor(store, character, beats) {
    this.store = store;
    this.character = character;
    this.beats = beats.filter((b) => b.character === character.id).sort((a, b) => a.start - b.start);
    this.pose = new Float32Array(store.nj * 3);
    this.tmp = new Float32Array(store.nj * 3);
  }

  beatAt(t) {
    let idx = 0;
    for (let i = 0; i < this.beats.length; i++) if (this.beats[i].start <= t + 1e-6) idx = i;
    return idx;
  }

  evaluate(t) {
    if (!this.beats.length) return this.pose;
    const i = this.beatAt(t);
    const b = this.beats[i];
    const local = Math.min(t - b.start, b.duration);
    this.store.sampleBeat(b, local, this.pose);
    if (i > 0 && local < BLEND) {
      const p = this.beats[i - 1];
      this.store.sampleBeat(p, p.duration, this.tmp);
      const w = smooth(local / BLEND);
      for (let k = 0; k < this.pose.length; k++) this.pose[k] = this.tmp[k] * (1 - w) + this.pose[k] * w;
    }
    return this.pose;
  }
}

const smooth = (x) => x * x * (3 - 2 * x);
