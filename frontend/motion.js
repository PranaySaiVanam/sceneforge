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
    let tau = easeWarp(Math.max(0, t), beat.duration, beat.ease_in || 0, beat.ease_out || 0) * beat.speed;
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
    const sc = beat.scale || 1; // taller or smaller characters: the whole clip, root motion included, scales
    for (let j = 0; j < this.nj; j++) {
      const k = j * 3;
      let x = c.frames[s0 + k] * (1 - a) + c.frames[s1 + k] * a;
      const y = c.frames[s0 + k + 1] * (1 - a) + c.frames[s1 + k + 1] * a;
      let z = c.frames[s0 + k + 2] * (1 - a) + c.frames[s1 + k + 2] * a;
      // loop offset (clip-local), then beat placement (world)
      let lx = (ox + x * cy + z * sy) * sc, lz = (oz - x * sy + z * cy) * sc;
      out[k] = beat.x + lx * cb + lz * sb;
      out[k + 1] = y * sc;
      out[k + 2] = beat.z - lx * sb + lz * cb;
    }
    return out;
  }
}

// Mirrors director.ease_warp: accelerate from rest / settle into a stop without changing the end point.
export function easeWarp(t, D, ei, eo, r0 = 0.35) {
  if ((ei <= 0 && eo <= 0) || D <= 0) return t;
  if (t <= 0) return 0;
  if (t >= D) return t;
  ei = Math.min(ei, D / 3); eo = Math.min(eo, D / 3);
  const k = (D - r0 * (ei + eo) / 2) / (D - (ei + eo) / 2);
  if (ei > 0 && t < ei) return r0 * t + (k - r0) * t * t / (2 * ei);
  const u1 = ei > 0 ? r0 * ei + (k - r0) * ei / 2 : 0;
  if (t <= D - eo || eo <= 0) return u1 + k * (t - ei);
  const s = t - (D - eo);
  return u1 + k * (D - eo - ei) + k * s - (k - r0) * s * s / (2 * eo);
}

function wrap(a) { return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; }

const MOUNT = 1.0; // seconds to climb on or off a horse

// Seats a rider pose (clip-local, root at origin facing +Z) on a horse pose: hips over the saddle.
function seatOnHorse(store, riderLocal, horsePose, out) {
  const J = store.ji, n = store.nj;
  const hx = horsePose[J.Spine * 3], hy = horsePose[J.Spine * 3 + 1], hz = horsePose[J.Spine * 3 + 2];
  const ax = horsePose[J.LeftUpLeg * 3] - horsePose[J.RightUpLeg * 3], az = horsePose[J.LeftUpLeg * 3 + 2] - horsePose[J.RightUpLeg * 3 + 2];
  const yaw = Math.atan2(-az, ax); // facing of the horse (0 = +Z)
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const rx = riderLocal[J.Hips * 3], ry = riderLocal[J.Hips * 3 + 1], rz = riderLocal[J.Hips * 3 + 2];
  for (let j = 0; j < n; j++) {
    const k = j * 3;
    const x = riderLocal[k] - rx, y = riderLocal[k + 1] - ry, z = riderLocal[k + 2] - rz;
    out[k] = hx + x * c + z * s;
    out[k + 1] = hy + 0.16 + y;
    out[k + 2] = hz - x * s + z * c;
  }
  return out;
}

// Seat offsets in the vehicle frame (x = left, z = forward), mirrored in director.SEAT.
const SEATS = {
  car: [[0.38, 0.25], [-0.38, 0.25], [0.38, -0.75], [-0.38, -0.75]],
  bike: [[0, -0.1], [0, -0.55], [0, -0.55], [0, -0.55]],
};

// Seats a rider pose (clip-local, root at origin facing +Z) in a car or on a bike, leaning with the turn.
function seatInVehicle(store, riderLocal, mountPose, kind, seat, info, out) {
  const J = store.ji, n = store.nj;
  const hx = mountPose[J.Hips * 3], hy = mountPose[J.Hips * 3 + 1], hz = mountPose[J.Hips * 3 + 2];
  const ax = mountPose[J.LeftUpLeg * 3] - mountPose[J.RightUpLeg * 3], az = mountPose[J.LeftUpLeg * 3 + 2] - mountPose[J.RightUpLeg * 3 + 2];
  const yaw = Math.atan2(-az, ax);
  const [sx, sz] = (SEATS[kind] || SEATS.car)[Math.min(seat || 0, 3)];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const seatY = kind === "bike" ? 0.97 + (hy - 0.55) : 0.56 + (hy - 0.6);
  const bx = hx + sx * c + sz * s, bz = hz - sx * s + sz * c;
  const rx = riderLocal[J.Hips * 3], ry = riderLocal[J.Hips * 3 + 1], rz = riderLocal[J.Hips * 3 + 2];
  // bikes: the rider leans into the turn with the bike (roll about the contact line);
  // cars: the body sways a little against the turn
  const roll = kind === "bike" ? -(info.lean || 0) : (info.sway || 0);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const pivotY = kind === "bike" ? -seatY : -0.2;
  for (let j = 0; j < n; j++) {
    const k = j * 3;
    let x = riderLocal[k] - rx, y = riderLocal[k + 1] - ry, z = riderLocal[k + 2] - rz;
    const yy = y - pivotY; // roll about the vehicle's long axis
    const x2 = x * cr - yy * sr, y2 = x * sr + yy * cr + pivotY;
    x = x2; y = y2;
    out[k] = bx + x * c + z * s;
    out[k + 1] = seatY + y;
    out[k + 2] = bz - x * s + z * c;
  }
  return out;
}

// Per-character timeline with crossfades between consecutive beats.
export class Performer {
  constructor(store, character, beats) {
    this.store = store;
    this.character = character;
    this.beats = beats.filter((b) => b.character === character.id).sort((a, b) => a.start - b.start);
    this.pose = new Float32Array(store.nj * 3);
    this.tmp = new Float32Array(store.nj * 3);
    this.local = new Float32Array(store.nj * 3);
    this.rides = this.beats.some((b) => b.ride);
    this.others = null; // id -> Performer, set by the app so riders can follow their horse
  }

  // Pose of one beat at beat-local time t (riders are seated on their horse's current pose).
  sample(b, local, out) {
    if (b.ride && this.others?.[b.ride]) {
      const zero = { ...b, x: 0, z: 0, yaw: 0, ease_in: 0, ease_out: 0 };
      this.store.sampleBeat(zero, local, this.local);
      const mount = this.others[b.ride];
      const kind = mount.character.kind;
      if (kind === "car" || kind === "bike") {
        return seatInVehicle(this.store, this.local, mount.pose, kind, b.seat, mount.motionInfo(b.start + local), out);
      }
      return seatOnHorse(this.store, this.local, mount.pose, out);
    }
    return this.store.sampleBeat(b, local, out);
  }

  // Heading of this actor's root at time t (current beat only, no blending).
  headingAt(t) {
    if (!this.beats.length) return { yaw: 0, x: 0, z: 0 };
    const b = this.beats[this.beatAt(t)];
    this.scratch ||= new Float32Array(this.store.nj * 3);
    this.store.sampleBeat(b, Math.min(Math.max(t - b.start, 0), b.duration), this.scratch);
    const J = this.store.ji, p = this.scratch;
    const ax = p[J.LeftUpLeg * 3] - p[J.RightUpLeg * 3], az = p[J.LeftUpLeg * 3 + 2] - p[J.RightUpLeg * 3 + 2];
    return { yaw: Math.atan2(-az, ax), x: p[J.Hips * 3], z: p[J.Hips * 3 + 2] };
  }

  // Turn rate and speed over the last 0.3 s: drives lean, steering and body sway for vehicles.
  motionInfo(t) {
    const a = this.headingAt(Math.max(0, t - 0.3)), b = this.headingAt(t);
    let dy = b.yaw - a.yaw;
    while (dy > Math.PI) dy -= 2 * Math.PI;
    while (dy < -Math.PI) dy += 2 * Math.PI;
    const rate = dy / 0.3, speed = Math.hypot(b.x - a.x, b.z - a.z) / 0.3;
    const clamp = (v, m) => Math.max(-m, Math.min(m, v));
    return { rate, speed, lean: clamp(rate * speed * 0.09, 0.55), steer: clamp(rate * 1.4, 1.4), sway: clamp(-rate * speed * 0.025, 0.14) };
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
    this.sample(b, local, this.pose);
    const p = i > 0 ? this.beats[i - 1] : null;
    const mount = p && (!!b.ride !== !!p.ride); // getting on or off a horse: slower blend with a hop
    const vehicle = mount && ["car", "bike"].includes(this.others?.[b.ride || p.ride]?.character.kind);
    const quick = b.clip === "eject_roll";
    const span = quick ? 0.2 : vehicle ? 0.8 : mount ? MOUNT : BLEND;
    if (p && local < span && !b.teleport) {
      if (p.ride) this.sample(p, Math.min(p.duration, t - p.start), this.tmp); // still on the moving horse
      else this.sample(p, p.duration, this.tmp);
      const w = smooth(local / span);
      const hop = mount && !vehicle && !quick ? 0.35 * Math.sin(Math.PI * w) : 0;
      for (let k = 0; k < this.pose.length; k++) {
        this.pose[k] = this.tmp[k] * (1 - w) + this.pose[k] * w + (k % 3 === 1 ? hop : 0);
      }
    }
    return this.pose;
  }
}

const smooth = (x) => x * x * (3 - 2 * x);
