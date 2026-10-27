// Layered overlays on top of the body animation (command scripts):
//   mood  - additive posture: sad (droop), angry (lean, arms out, heavy steps), anxious (head darts,
//           tense shoulders, hands near the chest), stealthy (low centre of gravity, scanning)
//   grip  - hand IK poses for props held while walking or idling: pistol, rifle, suitcase, cup
//   look  - the head (and a little of the chest) tracks an actor, the camera or a stage point
//   snap  - IK constraints: feet never sink into the floor; hands and feet lock onto ladder rungs
// They run after every actor's base pose is known, so a look can follow someone who is moving.
// apply() also returns each actor's gaze (yaw, pitch) so the rig can draw where they are looking.
import * as THREE from "three";

const UP = new THREE.Vector3(0, 1, 0);
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const smooth = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
const wrap = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
const GRIP_CATS = new Set(["locomotion", "idle"]);
const NO_GRIP = /^(draw_weapon|holster|shoot_|sword|mace_smash|push_off|brake|turn_|standin|scale_ref|tpose|apose)/;
const hash = (n) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

const MAT = {
  cup: new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 0.4 }),
  coffee: new THREE.MeshStandardMaterial({ color: 0x4a2c17, roughness: 0.3 }),
  case: new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 0.55, metalness: 0.2 }),
  trim: new THREE.MeshStandardMaterial({ color: 0xb8bec6, metalness: 0.8, roughness: 0.3 }),
};

function makeProp(type) {
  const g = new THREE.Group();
  if (type === "cup") {
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.034, 0.11, 16), MAT.cup);
    const top = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.038, 0.004, 16), MAT.coffee);
    top.position.y = 0.05;
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.026, 0.007, 6, 12), MAT.cup);
    handle.position.x = 0.05; handle.rotation.y = Math.PI / 2;
    g.add(body, top, handle);
  } else { // suitcase: handle at the origin, case hanging below
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.34, 0.13), MAT.case);
    box.position.y = -0.2;
    const handle = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.025, 0.03), MAT.trim);
    handle.position.y = -0.02;
    g.add(box, handle);
  }
  g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  return g;
}

// two-bone IK: shoulder/hip s, target, bone lengths, pole hint -> { mid, end }
function ik2(s, tgt, l1, l2, poleHint) {
  const d = tgt.clone().sub(s);
  const dist = Math.min(Math.max(d.length(), 1e-4), l1 + l2 - 1e-3);
  d.normalize();
  const a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(l1 * l1 - a * a, 0));
  const pole = poleHint.clone().addScaledVector(d, -poleHint.dot(d));
  if (pole.lengthSq() < 1e-8) pole.set(0, 0, 1);
  pole.normalize();
  return { mid: s.clone().addScaledVector(d, a).addScaledVector(pole, h), end: s.clone().addScaledVector(d, dist) };
}

export class Overlays {
  constructor(world, store, scene, camera) {
    this.world = world;
    this.store = store;
    this.camera = camera;
    this.layers = scene.layers || [];
    this.kind = Object.fromEntries(scene.characters.map((c) => [c.id, c.kind || "human"]));
    this.beats = {};
    for (const b of scene.beats) (this.beats[b.character] ||= []).push(b);
    Object.values(this.beats).forEach((l) => l.sort((a, b) => a.start - b.start));
    this.props = this.layers.filter((l) => l.type === "grip").map((l) => {
      const g = makeProp(l.prop);
      g.visible = false;
      world.add(g);
      return { l, g };
    });
    this.weaponOf = () => null;
    this.ladders = () => [];
    this.gaze = {};
  }

  clipAt(cid, t) {
    const list = this.beats[cid] || [];
    let b = null;
    for (const x of list) if (x.start <= t + 1e-6) b = x;
    return b;
  }

  weight(l, t, ramp) {
    const a = l.start, z = l.start + l.duration;
    if (t < a || t > z) return 0;
    return Math.min(smooth((t - a) / ramp), smooth((z - t) / 0.4));
  }

  apply(t, poses) {
    const J = this.store.ji;
    const base = {}; // targets read the pre-overlay head positions so looks don't chase each other
    for (const [id, p] of Object.entries(poses)) base[id] = Float32Array.from(p);
    const P = (p, j) => V(p[j * 3], p[j * 3 + 1], p[j * 3 + 2]);
    const set = (p, j, v) => { p[j * 3] = v.x; p[j * 3 + 1] = v.y; p[j * 3 + 2] = v.z; };
    const rotAbout = (p, joints, pivot, axis, ang) => {
      if (Math.abs(ang) < 1e-5) return;
      const q = new THREE.Quaternion().setFromAxisAngle(axis, ang);
      for (const j of joints) set(p, j, P(p, j).sub(pivot).applyQuaternion(q).add(pivot));
    };
    const move = (p, joints, v) => { for (const j of joints) set(p, j, P(p, j).add(v)); };
    const frame = (p) => {
      const left = P(p, J.LeftUpLeg).sub(P(p, J.RightUpLeg)); left.y = 0; left.normalize();
      const fwd = new THREE.Vector3().crossVectors(left, UP).normalize();
      return { left, fwd };
    };
    const ARM_L = [J.LeftArm, J.LeftForeArm, J.LeftHand], ARM_R = [J.RightArm, J.RightForeArm, J.RightHand];
    const HEAD = [J.Head, J.Head_End];
    const CHEST = [J.Neck1, ...HEAD, ...ARM_L, ...ARM_R];
    const SPINE = [J.Spine1, ...CHEST];
    const ABOVE_LEGS = [J.Hips, J.Spine, J.LeftUpLeg, J.RightUpLeg, ...SPINE];
    // lower the pelvis by d and re-solve the knees so the feet stay planted
    const lowerHips = (p, d) => {
      if (d <= 1e-4) return;
      const keep = ["Left", "Right"].map((s) => {
        const hip = P(p, J[`${s}UpLeg`]), knee = P(p, J[`${s}Leg`]), foot = P(p, J[`${s}Foot`]);
        return { s, l1: knee.distanceTo(hip), l2: foot.distanceTo(knee), pole: knee.clone().sub(hip.clone().lerp(foot, 0.5)), foot };
      });
      move(p, ABOVE_LEGS, V(0, -d, 0));
      for (const k of keep) {
        const r = ik2(P(p, J[`${k.s}UpLeg`]), k.foot, k.l1, k.l2, k.pole.lengthSq() > 1e-6 ? k.pole : V(0, 0, 1));
        set(p, J[`${k.s}Leg`], r.mid);
      }
    };
    const gaze = {};

    for (const id of Object.keys(poses)) {
      const p = poses[id];
      if (this.kind[id] !== "human") continue;
      const b = this.clipAt(id, t);
      const meta = b && this.store.meta[b.clip];
      const upright = P(p, J.Head).y > 1.0;
      // ---- base gaze: chest facing, or the tipped Head_End of gaze-marked clips (eye-line sweep etc.)
      const chestL = P(p, J.LeftArm).sub(P(p, J.RightArm)); chestL.y = 0; chestL.normalize();
      const chestF = new THREE.Vector3().crossVectors(chestL, UP);
      let gy = Math.atan2(chestF.x, chestF.z), gp = 0;
      if (meta?.gaze) {
        const d = P(p, J.Head_End).sub(P(p, J.Head)); const hz = Math.hypot(d.x, d.z);
        if (hz > 0.02) { gy = Math.atan2(d.x, d.z); gp = -Math.atan2(hz, d.y) * 0.3; }
      }
      const { left, fwd } = frame(p);

      // ---- mood (additive posture on walks, runs and idles)
      if (upright && meta && GRIP_CATS.has(meta.category)) {
        for (const l of this.layers) {
          if (l.type !== "mood" || l.character !== id) continue;
          const w = this.weight(l, t, 0.6);
          if (w <= 0) continue;
          const neck = () => P(p, J.Neck1), spine = () => P(p, J.Spine);
          if (l.mood === "sad") {
            rotAbout(p, SPINE, spine(), left, 0.15 * w);
            move(p, [...ARM_L, ...ARM_R], fwd.clone().multiplyScalar(0.03 * w).addScaledVector(UP, -0.035 * w));
            rotAbout(p, HEAD, neck(), left, 0.38 * w);
            gp -= 0.38 * w;
          } else if (l.mood === "angry") {
            const feet = Math.min(P(p, J.LeftFoot).y, P(p, J.RightFoot).y);
            lowerHips(p, 0.035 * w * (1 - smooth((feet - 0.07) / 0.1))); // heavy, stomping footfalls
            rotAbout(p, SPINE, spine(), left, 0.17 * w);
            rotAbout(p, ARM_L, P(p, J.LeftArm), fwd, 0.14 * w);
            rotAbout(p, ARM_R, P(p, J.RightArm), fwd, -0.14 * w);
            move(p, [...ARM_L, ...ARM_R], V(0, 0.02 * w, 0));
            rotAbout(p, HEAD, neck(), left, 0.12 * w);
          } else if (l.mood === "anxious") {
            const seg = Math.floor(t / 0.55 + hash(id.charCodeAt(0)));
            const from = (hash(seg) - 0.5) * 1.3, to = (hash(seg + 1) - 0.5) * 1.3;
            const u = smooth(((t / 0.55 + hash(id.charCodeAt(0))) % 1) / 0.22);
            const dart = (from + (to - from) * u) * w;
            move(p, [...ARM_L, ...ARM_R], V(0, 0.035 * w, 0)); // tense, raised shoulders
            this.gripPose(p, "chest", 0.65 * w, left, fwd, P, set);
            rotAbout(p, HEAD, neck(), UP, dart);
            gy += dart;
          } else if (l.mood === "stealthy") {
            lowerHips(p, 0.15 * w);
            rotAbout(p, SPINE, spine(), left, 0.2 * w);
            const scan = 0.65 * Math.sin((2 * Math.PI * t) / 3.2) * w;
            rotAbout(p, HEAD, neck(), UP, scan);
            gy += scan;
          }
        }
      }

      // ---- grips (hands first, so a look's chest twist carries the arms with it)
      if (upright && meta && GRIP_CATS.has(meta.category) && !NO_GRIP.test(b.clip)) {
        const grips = [];
        const w = this.weaponOf(id, t);
        if (w === "gun" || w === "rifle") grips.push({ prop: w, w: 1 });
        for (const l of this.layers) if (l.type === "grip" && l.character === id) {
          const k = this.weight(l, t, 0.35);
          if (k > 0) grips.push({ prop: l.prop, w: k });
        }
        const fade = smooth((t - b.start) / 0.35);
        const fr = frame(p);
        for (const g of grips) this.gripPose(p, g.prop, g.w * fade, fr.left, fr.fwd, P, set);
      }
      gaze[id] = { yaw: gy, pitch: gp };
    }

    // ---- looks
    for (const l of this.layers) {
      if (l.type !== "look") continue;
      const p = poses[l.character];
      if (!p || P(p, J.Head).y < 1.0) continue;
      const w = this.weight(l, t, l.snap ? 0.12 : 0.4);
      if (w <= 0) continue;
      let target;
      if (l.target === "camera") target = this.camera.position.clone();
      else if (l.target && base[l.target]) target = P(base[l.target], J.Head);
      else if (l.point) target = V(l.point[0], 1.55, l.point[1]);
      else continue;
      const { fwd } = frame(p);
      const head = P(p, J.Head);
      const body = Math.atan2(fwd.x, fwd.z);
      let delta = wrap(Math.atan2(target.x - head.x, target.z - head.z) - body);
      delta = Math.max(-1.9, Math.min(1.9, delta));
      const headPart = Math.max(-1.25, Math.min(1.25, delta));
      const chestPart = delta - headPart;
      const horiz = Math.hypot(target.x - head.x, target.z - head.z);
      const pitch = Math.max(-0.5, Math.min(0.5, Math.atan2(target.y - head.y, Math.max(horiz, 0.3))));
      rotAbout(p, CHEST, P(p, J.Spine1), UP, chestPart * w);
      const neck = P(p, J.Neck1);
      rotAbout(p, HEAD, neck, UP, headPart * w);
      const dir = V(Math.sin(body + delta), 0, Math.cos(body + delta));
      const axis = new THREE.Vector3().crossVectors(UP, dir).normalize();
      rotAbout(p, HEAD, neck, axis, -pitch * w * 0.8);
      const g = gaze[l.character] || { yaw: body, pitch: 0 };
      g.yaw = g.yaw + wrap(body + delta - g.yaw) * w;
      g.pitch = g.pitch * (1 - w) + pitch * w;
      gaze[l.character] = g;
    }

    // ---- IK snap: feet above the floor, hands and feet on ladder rungs
    const ladders = this.ladders(t);
    for (const id of Object.keys(poses)) {
      if (this.kind[id] !== "human") continue;
      const p = poses[id];
      for (const s of ["Left", "Right"]) {
        const fj = J[`${s}Foot`], tj = J[`${s}ToeBase`];
        const dy = Math.max(0.035 - p[fj * 3 + 1], 0.012 - p[tj * 3 + 1], 0);
        if (dy > 1e-4 && P(p, J.Hips).y > 0.45) {
          const hip = P(p, J[`${s}UpLeg`]), knee = P(p, J[`${s}Leg`]), foot = P(p, fj);
          const r = ik2(hip, foot.clone().add(V(0, dy, 0)), knee.distanceTo(hip), foot.distanceTo(knee), knee.clone().sub(hip));
          set(p, J[`${s}Leg`], r.mid); set(p, fj, r.end); set(p, tj, P(p, tj).add(V(0, dy, 0)));
        }
      }
      for (const lad of ladders) {
        if (lad.cid !== id) continue;
        const inv = new THREE.Matrix4().copy(lad.g.matrixWorld).invert();
        for (const [j, parent, base_] of [[J.LeftHand, J.LeftForeArm, J.LeftArm], [J.RightHand, J.RightForeArm, J.RightArm],
          [J.LeftFoot, J.LeftLeg, J.LeftUpLeg], [J.RightFoot, J.RightLeg, J.RightUpLeg]]) {
          const w = P(p, j), loc = w.clone().applyMatrix4(inv);
          if (Math.abs(loc.z) > 0.45 || Math.abs(loc.x) > 0.4 || loc.y < 0.15) continue;
          const rung = Math.max(0.3, Math.round(loc.y / 0.3) * 0.3);
          const snapped = V(Math.max(-0.2, Math.min(0.2, loc.x)), rung + (j === J.LeftFoot || j === J.RightFoot ? 0.04 : 0.0), -0.04)
            .applyMatrix4(lad.g.matrixWorld);
          const root = P(p, base_), mid = P(p, parent);
          const r = ik2(root, snapped, mid.distanceTo(root), w.distanceTo(mid), mid.clone().sub(root));
          set(p, parent, r.mid); set(p, j, r.end);
        }
      }
    }

    // ---- held props follow the right hand
    for (const { l, g } of this.props) {
      const p = poses[l.character];
      const on = p && t >= l.start && t <= l.start + l.duration;
      g.visible = !!on;
      if (!on) continue;
      const hand = P(p, J.RightHand), fore = P(p, J.RightForeArm);
      const { fwd } = frame(p);
      g.position.copy(hand);
      g.rotation.set(0, Math.atan2(fwd.x, fwd.z), 0);
      if (l.prop === "cup") g.position.add(hand.clone().sub(fore).normalize().multiplyScalar(0.05)).add(V(0, 0.02, 0));
    }
    this.gaze = gaze;
    return gaze;
  }

  gripPose(p, prop, w, left, fwd, P, set) {
    if (w <= 0) return;
    const J = this.store.ji;
    const sR = P(p, J.RightArm), sL = P(p, J.LeftArm);
    const at = (s, f, u, l) => s.clone().addScaledVector(fwd, f).addScaledVector(UP, u).addScaledVector(left, l);
    const targets = [];
    if (prop === "gun") { // low ready, two hands
      const r = at(sR, 0.4, -0.4, 0.12);
      targets.push(["Right", r], ["Left", r.clone().addScaledVector(left, -0.02).addScaledVector(UP, -0.04)]);
    } else if (prop === "rifle") { // port arms, muzzle low and forward
      targets.push(["Right", at(sR, 0.2, -0.32, 0.08)], ["Left", at(sR, 0.6, -0.26, 0.2)]);
    } else if (prop === "suitcase") {
      targets.push(["Right", at(sR, 0.03, -0.64, -0.1)]);
    } else if (prop === "cup") {
      targets.push(["Right", at(sR, 0.3, -0.3, 0.1)]);
    } else if (prop === "chest") { // anxious: hands drawn up near the chest
      targets.push(["Right", at(sR, 0.24, -0.26, 0.1)], ["Left", at(sL, 0.24, -0.28, -0.1)]);
    }
    for (const [side, tgt] of targets) {
      const s = side === "Right" ? sR : sL;
      const fj = J[`${side}ForeArm`], hj = J[`${side}Hand`];
      const e0 = P(p, fj), h0 = P(p, hj);
      const sx = side === "Right" ? -1 : 1;
      const pole = UP.clone().multiplyScalar(-0.5).addScaledVector(left, sx * 0.6).addScaledVector(fwd, -0.3);
      const r = ik2(s, tgt, e0.distanceTo(s), h0.distanceTo(e0), pole);
      set(p, fj, e0.lerp(r.mid, w));
      set(p, hj, h0.lerp(r.end, w));
    }
  }

  dispose() {
    for (const { g } of this.props) this.world.remove(g);
    this.props = [];
  }
}
