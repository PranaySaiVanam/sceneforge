// Interactive set pieces: scene objects (van, door, chair, ...), props anchored to a beat
// (a door that swings open, a button that lights up, a ladder, a chair under a seated actor),
// and carried items (luggage that is lifted, carried and stowed in the van; a barbell).
// Everything is computed from scene time and the actors' poses, so scrubbing stays consistent.
import * as THREE from "three";

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const UP = V(0, 1, 0);
const M = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.6, ...o });
const MAT = {
  van: M(0xe9edf1, { roughness: 0.35, metalness: 0.15 }),
  vanTrim: M(0x2d3640, { roughness: 0.5 }),
  glass: M(0x23384a, { roughness: 0.1, metalness: 0.3 }),
  tyre: M(0x15181b, { roughness: 0.9 }),
  cargo: M(0x9aa3ab, { roughness: 0.8 }),
  door: M(0x8a5a3b, { roughness: 0.6 }),
  frame: M(0xd8d2c4, { roughness: 0.7 }),
  steel: M(0xb7bec5, { metalness: 0.7, roughness: 0.3 }),
  panel: M(0x39424b, { roughness: 0.5 }),
  btnOff: M(0xb83a2e, { emissive: 0x401010, emissiveIntensity: 0.6 }),
  btnOn: M(0x3ccf72, { emissive: 0x1f9a4d, emissiveIntensity: 1.4 }),
  ladder: M(0xc9a227, { roughness: 0.6, metalness: 0.3 }),
  chair: M(0x5a6570, { roughness: 0.7 }),
  luggage: M(0x7a3b2e, { roughness: 0.75 }),
  crate: M(0xa77b45, { roughness: 0.9 }),
  plate: M(0x1d2125, { roughness: 0.5, metalness: 0.4 }),
  wall: M(0xb9b2a5, { roughness: 0.95 }),
  curb: M(0x8d9196, { roughness: 0.95 }),
  hazard: M(0xe0b43a, { roughness: 0.6 }),
  counter: M(0x6b4f3a, { roughness: 0.6 }),
  counterTop: M(0xd9d4cb, { roughness: 0.35 }),
  laptop: M(0x2a2f36, { roughness: 0.4, metalness: 0.5 }),
  screen: M(0x9fd3ff, { emissive: 0x3a6f9a, emissiveIntensity: 0.9 }),
  book: M(0x7a2430, { roughness: 0.7 }),
  pages: M(0xf3efe4, { roughness: 0.9 }),
  phone: M(0x111418, { roughness: 0.3, metalness: 0.4 }),
  broom: M(0x9a7b4f, { roughness: 0.8 }),
  bristle: M(0xc9a85c, { roughness: 1 }),
};
function buildCounter() {
  const g = new THREE.Group();
  g.add(mesh(boxG(1.8, 0.9, 0.6), MAT.counter, 0, 0.45, 0), mesh(boxG(1.9, 0.05, 0.66), MAT.counterTop, 0, 0.925, 0));
  return g;
}
function buildDesk() { // desk + laptop; the actor sits on the chair added separately
  const g = new THREE.Group();
  g.add(mesh(boxG(1.2, 0.04, 0.6), MAT.counterTop, 0, 0.74, 0));
  for (const [x, z] of [[-0.55, -0.25], [0.55, -0.25], [-0.55, 0.25], [0.55, 0.25]]) g.add(mesh(boxG(0.04, 0.72, 0.04), MAT.steel, x, 0.36, z));
  g.add(mesh(boxG(0.34, 0.018, 0.24), MAT.laptop, 0, 0.77, 0.05));
  const lid = new THREE.Group();
  lid.add(mesh(boxG(0.34, 0.23, 0.012), MAT.laptop, 0, 0.115, 0), mesh(boxG(0.31, 0.2, 0.002), MAT.screen, 0, 0.115, -0.007));
  lid.position.set(0, 0.78, 0.17);
  lid.rotation.x = 0.25;
  g.add(lid);
  return g;
}
function buildBook() {
  const g = new THREE.Group();
  g.add(mesh(boxG(0.3, 0.025, 0.21), MAT.book, 0, 0, 0), mesh(boxG(0.28, 0.03, 0.19), MAT.pages, 0, 0.004, 0));
  return g;
}
function buildPhone() {
  const g = new THREE.Group();
  g.add(mesh(boxG(0.075, 0.012, 0.15), MAT.phone, 0, 0, 0), mesh(boxG(0.065, 0.002, 0.13), MAT.screen, 0, 0.007, 0));
  return g;
}
function buildBroom() { // handle along +Y from the bristle head at the origin
  const g = new THREE.Group();
  g.add(mesh(new THREE.CylinderGeometry(0.014, 0.014, 1.35, 8), MAT.broom, 0, 0.72, 0), mesh(boxG(0.32, 0.1, 0.07), MAT.bristle, 0, 0.05, 0));
  return g;
}

function mesh(geo, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}
const boxG = (w, h, d) => new THREE.BoxGeometry(w, h, d);

// ---------------------------------------------------------------- builders (local +Z = facing)
function buildVan() {
  const g = new THREE.Group();
  g.add(mesh(boxG(1.95, 1.9, 3.3), MAT.van, 0, 1.35, -0.55)); // cargo box, rear at z = -2.2
  g.add(mesh(boxG(1.95, 1.45, 1.3), MAT.van, 0, 1.12, 1.75)); // cab
  g.add(mesh(boxG(1.9, 0.62, 0.05), MAT.glass, 0, 1.55, 2.41)); // windscreen
  g.add(mesh(boxG(2.0, 0.2, 4.7), MAT.vanTrim, 0, 0.42, 0.1)); // chassis
  g.add(mesh(boxG(1.86, 0.05, 3.2), MAT.cargo, 0, 0.56, -0.55)); // cargo floor
  for (const [x, z] of [[-0.9, 1.6], [0.9, 1.6], [-0.9, -1.4], [0.9, -1.4]]) {
    const w = mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.26, 18), MAT.tyre, x, 0.36, z);
    w.rotation.z = Math.PI / 2;
    g.add(w);
  }
  // open rear doors, swung outward
  for (const s of [-1, 1]) {
    const hinge = new THREE.Group();
    hinge.position.set(s * 0.97, 0, -2.2);
    hinge.add(mesh(boxG(0.96, 1.8, 0.05), MAT.van, -s * 0.48, 1.35, 0));
    hinge.rotation.y = s * 1.9;
    g.add(hinge);
  }
  g.userData.cargo = (k) => V(((k % 2) ? 0.45 : -0.45), 0.86, -1.95 + Math.floor(k / 2) * 0.62); // stowing slots
  return g;
}

function buildDoor() {
  const g = new THREE.Group();
  g.add(mesh(boxG(0.1, 2.25, 0.14), MAT.frame, -0.5, 1.12, 0), mesh(boxG(0.1, 2.25, 0.14), MAT.frame, 0.5, 1.12, 0),
    mesh(boxG(1.1, 0.1, 0.14), MAT.frame, 0, 2.25, 0));
  const hinge = new THREE.Group();
  hinge.position.set(-0.45, 0, 0); // hinge on the actor's right as they face the door
  const leaf = mesh(boxG(0.9, 2.15, 0.05), MAT.door, 0.45, 1.08, 0);
  const knob = mesh(new THREE.SphereGeometry(0.04, 10, 8), MAT.steel, 0.82, 1.0, -0.06);
  hinge.add(leaf, knob);
  g.add(hinge);
  g.userData.hinge = hinge;
  return g;
}

function buildButton() {
  const g = new THREE.Group();
  g.add(mesh(boxG(0.32, 1.32, 0.16), MAT.panel, 0, 0.66, 0));
  const btn = mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.04, 16), MAT.btnOff, 0, 1.22, -0.09);
  btn.rotation.x = Math.PI / 2;
  g.add(btn);
  g.userData.btn = btn;
  return g;
}

function buildLadder(h = 3.2) {
  const g = new THREE.Group();
  for (const s of [-1, 1]) g.add(mesh(boxG(0.05, h, 0.06), MAT.ladder, s * 0.24, h / 2, 0));
  for (let y = 0.3; y < h; y += 0.3) {
    const rung = mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.48, 8), MAT.ladder, 0, y, 0);
    rung.rotation.z = Math.PI / 2;
    g.add(rung);
  }
  return g;
}

function buildChair(seat = 0.46) {
  const g = new THREE.Group();
  g.add(mesh(boxG(0.46, 0.05, 0.44), MAT.chair, 0, seat, 0));
  g.add(mesh(boxG(0.46, 0.5, 0.05), MAT.chair, 0, seat + 0.27, -0.21));
  for (const [x, z] of [[-0.2, -0.19], [0.2, -0.19], [-0.2, 0.19], [0.2, 0.19]]) {
    g.add(mesh(boxG(0.035, seat, 0.035), MAT.chair, x, seat / 2, z));
  }
  return g;
}

function buildLuggage() {
  const g = new THREE.Group();
  g.add(mesh(boxG(0.5, 0.34, 0.26), MAT.luggage, 0, 0, 0));
  g.add(mesh(boxG(0.2, 0.04, 0.04), MAT.vanTrim, 0, 0.19, 0));
  return g;
}

function buildBarbell() {
  const g = new THREE.Group();
  const bar = mesh(new THREE.CylinderGeometry(0.014, 0.014, 1.7, 10), MAT.steel);
  bar.rotation.z = Math.PI / 2;
  g.add(bar);
  for (const s of [-1, 1]) {
    for (const k of [0, 1]) {
      const p = mesh(new THREE.CylinderGeometry(0.21 - k * 0.04, 0.21 - k * 0.04, 0.04, 22), MAT.plate, s * (0.62 + k * 0.05), 0, 0);
      p.rotation.z = Math.PI / 2;
      g.add(p);
    }
  }
  return g;
}

// Obstacles. Walls run along local X; curbs, gaps and beams are crossed along local Z.
function buildWall(length = 6) {
  const g = new THREE.Group();
  g.add(mesh(boxG(length, 2.4, 0.3), MAT.wall, 0, 1.2, 0));
  return g;
}
function buildCurb() {
  const g = new THREE.Group();
  g.add(mesh(boxG(2.6, 0.16, 0.32), MAT.curb, 0, 0.08, 0));
  for (let i = -1; i <= 1; i += 2) g.add(mesh(boxG(0.3, 0.165, 0.325), MAT.hazard, i * 0.9, 0.08, 0));
  return g;
}
function buildGap() { // two panels either side of a 0.9 m opening (blocking uses the same geometry)
  const g = new THREE.Group();
  for (const sg of [-1, 1]) g.add(mesh(boxG(2.0, 2.4, 0.3), MAT.wall, sg * 1.45, 1.2, 0));
  return g;
}
function buildBeam() { // low pipe on two posts; bottom at 1.3 m
  const g = new THREE.Group();
  for (const sg of [-1, 1]) g.add(mesh(boxG(0.12, 1.42, 0.12), MAT.steel, sg * 1.2, 0.71, 0));
  const beam = mesh(new THREE.CylinderGeometry(0.07, 0.07, 2.52, 14), MAT.hazard, 0, 1.37, 0);
  beam.rotation.z = Math.PI / 2;
  g.add(beam);
  return g;
}

const BUILDERS = { van: buildVan, door: buildDoor, button: buildButton, ladder: buildLadder, chair: buildChair,
  box: buildLuggage, barbell: buildBarbell, curb: buildCurb, gap: buildGap, beam: buildBeam, crate: () => { const g = new THREE.Group(); g.add(mesh(boxG(0.9, 0.9, 0.9), MAT.crate, 0, 0.45, 0)); return g; } };

const CARRY_START = new Set(["lift_heavy", "pick_carry_place"]);
const CARRY_CLIPS = new Set(["lift_heavy", "pick_carry_place", "carry_walk", "set_down_heavy"]);

export class StageProps {
  // Director's camera: set walls between the lens and the action turn see-through (like flying a wall out).
  ghost(cam, look) {
    const dir = look.clone().sub(cam);
    const len = dir.length();
    this.ray.set(cam, dir.normalize());
    this.ray.far = Math.max(len - 0.6, 0.1);
    for (const g of this.solid) {
      const hide = this.ray.intersectObject(g, true).length > 0;
      g.traverse((m) => { if (m.isMesh && m.userData.mat) m.material = hide ? m.userData.ghost : m.userData.mat; });
    }
  }

  constructor(world, scene, store, probe) {
    this.world = world;
    this.store = store;
    this.list = [];
    const J = store.ji;
    const add = (g) => { world.add(g); this.list.push(g); return g; };
    // scene objects
    this.objects = {};
    this.solid = [];
    this.ray = new THREE.Raycaster();
    for (const o of scene.objects || []) {
      const g = add(o.type === "wall" ? buildWall(o.length || 6) : BUILDERS[o.type] ? BUILDERS[o.type]() : new THREE.Group());
      g.position.set(o.x, o.type === "box" ? 0.17 : 0, o.z);
      g.rotation.y = o.yaw;
      g.userData.slots = 0;
      this.objects[o.id] = g;
      if (["wall", "gap", "van", "beam", "crate"].includes(o.type)) {
        g.traverse((m) => {
          if (!m.isMesh) return;
          m.userData.mat = m.material;
          m.userData.ghost = Object.assign(m.material.clone(), { transparent: true, opacity: 0.16, depthWrite: false });
        });
        this.solid.push(g);
      }
    }
    this.anchored = [];
    this.carried = [];
    this.barbells = [];
    this.handProps = [];
    const byChar = {};
    for (const b of scene.beats) (byChar[b.character] ||= []).push(b);
    for (const [cid, beats] of Object.entries(byChar)) {
      beats.sort((a, b) => a.start - b.start);
      // beat-anchored props, placed in front of the actor
      for (let i = 0; i < beats.length; i++) {
        const b = beats[i];
        const fwd = V(Math.sin(b.yaw), 0, Math.cos(b.yaw));
        const place = (g, dist) => {
          g.position.set(b.x, 0, b.z).addScaledVector(fwd, dist);
          g.rotation.y = b.yaw + Math.PI; // the prop faces the actor
          return g;
        };
        if (b.clip === "open_door" && !b.target_object) {
          const g = add(place(buildDoor(), 0.62));
          this.anchored.push({ g, a: b.start - 0.5, z: Infinity, kind: "door", b });
        } else if (b.clip === "push_button") {
          const g = add(place(buildButton(), 0.66));
          this.anchored.push({ g, a: b.start - 0.5, z: b.start + b.duration + 3, kind: "button", b });
        } else if (b.clip === "climb") {
          const g = add(place(buildLadder(), 0.38));
          this.anchored.push({ g, a: b.start - 0.5, z: b.start + b.duration + 0.5, kind: "ladder", b });
        } else if (b.clip === "sit_down" || b.clip === "sit_chair") {
          // chair under the seated hips; stays until the actor stands up
          const tSeat = b.clip === "sit_down" ? b.start + b.duration : b.start + 0.2;
          const p = probe(cid, tSeat);
          const hips = V(p[J.Hips * 3], p[J.Hips * 3 + 1], p[J.Hips * 3 + 2]);
          const g = add(buildChair(Math.max(0.3, hips.y - 0.12)));
          g.position.set(hips.x, 0, hips.z).addScaledVector(fwd, -0.05);
          g.rotation.y = b.yaw;
          const up = beats.slice(i + 1).find((x) => x.clip === "stand_up");
          this.anchored.push({ g, a: b.start - 0.3, z: up ? up.start + up.duration + 0.5 : Infinity, kind: "chair", b });
          if (b.clip === "sit_down" && beats[i + 1]?.clip === "sit_chair") i++;
        } else if ((b.clip === "wipe_counter" || b.clip === "type_laptop") && beats[i - 1]?.clip !== b.clip) {
          let end = i;
          while (beats[end + 1]?.clip === b.clip) end++;
          const z = beats[end].start + beats[end].duration + 0.5;
          if (b.clip === "wipe_counter") {
            this.anchored.push({ g: add(place(buildCounter(), 0.66)), a: b.start - 0.4, z, kind: "counter", b });
          } else {
            const desk = add(place(buildDesk(), 0.62));
            desk.rotation.y = b.yaw; // keyboard toward the actor
            this.anchored.push({ g: desk, a: b.start - 0.4, z, kind: "desk", b });
            const p = probe(cid, b.start + 0.5);
            const hips = V(p[J.Hips * 3], p[J.Hips * 3 + 1], p[J.Hips * 3 + 2]);
            const chair = add(buildChair(Math.max(0.3, hips.y - 0.1)));
            chair.position.set(hips.x, 0, hips.z).addScaledVector(fwd, -0.05);
            chair.rotation.y = b.yaw;
            this.anchored.push({ g: chair, a: b.start - 0.4, z, kind: "chair", b });
          }
        } else if (["read_book", "idle_phone", "sweep"].includes(b.clip)) {
          const kind = { read_book: "book", idle_phone: "phone", sweep: "broom" }[b.clip];
          const g = add({ book: buildBook, phone: buildPhone, broom: buildBroom }[kind]());
          this.handProps.push({ g, cid, kind, a: b.start, z: b.start + b.duration });
        } else if (b.clip === "weights") {
          const g = add(buildBarbell());
          this.barbells.push({ g, cid, a: b.start, z: b.start + b.duration });
        }
      }
      // carried luggage: lifted at the lowest hand point, set down at the next one
      let chain = null;
      const lowest = (a, z, first = true) => {
        let best = Infinity, bt = a, found = [];
        for (let t = a; t <= z; t += 1 / 30) {
          const p = probe(cid, t);
          const y = Math.min(p[J.LeftHand * 3 + 1], p[J.RightHand * 3 + 1]);
          found.push([t, y]);
          if (y < best - 1e-4 || (!first && y <= best + 0.02)) { best = Math.min(best, y); bt = t; }
        }
        return bt;
      };
      for (const b of beats) {
        if (!CARRY_CLIPS.has(b.clip)) { if (chain && chain.end == null) chain.end = b.start; continue; }
        if (!chain && !(CARRY_START.has(b.clip) || b.clip === "carry_walk")) continue;
        if (!chain || chain.end != null) {
          if (CARRY_START.has(b.clip) || b.clip === "carry_walk") {
            chain = { cid, attach: CARRY_START.has(b.clip) ? lowest(b.start, b.start + Math.min(b.duration, 3.5)) : b.start, end: null, target: null };
            this.carried.push(chain);
          }
        }
        if (!chain) continue;
        if (b.clip === "pick_carry_place") chain.end = lowest(b.start + b.duration * 0.6, b.start + b.duration, false);
        if (b.clip === "set_down_heavy") {
          chain.end = lowest(b.start, b.start + b.duration, false);
          chain.target = b.target_object || null;
        }
      }
    }
    // resolve resting spots for each carried item (before pickup and after setting down)
    for (const c of this.carried) {
      c.g = add(buildLuggage());
      const mid = (t) => {
        const p = probe(c.cid, t);
        return V((p[J.LeftHand * 3] + p[J.RightHand * 3]) / 2, (p[J.LeftHand * 3 + 1] + p[J.RightHand * 3 + 1]) / 2,
          (p[J.LeftHand * 3 + 2] + p[J.RightHand * 3 + 2]) / 2);
      };
      const s = mid(c.attach);
      c.before = V(s.x, 0.17, s.z);
      if (c.end != null) {
        const e = mid(c.end);
        const van = c.target ? this.objects[c.target] : null;
        if (van && van.userData.cargo && van.position.distanceTo(V(e.x, 0, e.z)) < 4.5) {
          const local = van.userData.cargo(van.userData.slots++);
          c.after = local.applyAxisAngle(UP, van.rotation.y).add(van.position);
        } else {
          c.after = V(e.x, 0.17, e.z);
        }
        c.drop = e;
      }
    }
  }

  // ladders in use at time t (the overlays snap hands and feet onto their rungs)
  laddersAt(t) {
    return this.anchored.filter((p) => p.kind === "ladder" && t >= p.b.start && t <= p.b.start + p.b.duration)
      .map((p) => { p.g.updateMatrixWorld(); return { cid: p.b.character, g: p.g }; });
  }

  update(t, poses) {
    const J = this.store.ji;
    for (const h of this.handProps) {
      const q = poses[h.cid];
      h.g.visible = !!q && t >= h.a && t <= h.z;
      if (!h.g.visible) continue;
      const P = (j) => V(q[j * 3], q[j * 3 + 1], q[j * 3 + 2]);
      const l = P(J.LeftHand), r = P(J.RightHand);
      const left = P(J.LeftUpLeg).sub(P(J.RightUpLeg)); left.y = 0; left.normalize();
      const yaw = Math.atan2(-left.z, left.x);
      if (h.kind === "book") {
        h.g.position.copy(l).lerp(r, 0.5).add(V(0, 0.03, 0));
        h.g.rotation.set(-0.5, yaw, 0, "YXZ");
      } else if (h.kind === "phone") {
        h.g.position.copy(r).add(V(0, 0.03, 0));
        h.g.rotation.set(-0.6, yaw, 0, "YXZ");
      } else { // broom: through both hands down to the floor
        const dir = r.clone().sub(l).normalize();
        const foot = l.clone().addScaledVector(dir, -l.y / Math.max(dir.y, 0.35));
        foot.y = 0.02;
        h.g.position.copy(foot);
        h.g.quaternion.setFromUnitVectors(UP, r.clone().sub(foot).normalize());
      }
    }
    for (const p of this.anchored) {
      p.g.visible = t >= p.a && t <= p.z;
      const local = t - p.b.start;
      if (p.kind === "door" && p.g.userData.hinge) {
        const u = Math.min(Math.max((local - 0.7) / 1.1, 0), 1);
        p.g.userData.hinge.rotation.y = -1.6 * (u * u * (3 - 2 * u));
      }
      if (p.kind === "button") p.g.userData.btn.material = local >= 0.9 ? MAT.btnOn : MAT.btnOff;
    }
    const hands = (cid) => {
      const q = poses[cid];
      if (!q) return null;
      const l = V(q[J.LeftHand * 3], q[J.LeftHand * 3 + 1], q[J.LeftHand * 3 + 2]);
      const r = V(q[J.RightHand * 3], q[J.RightHand * 3 + 1], q[J.RightHand * 3 + 2]);
      return { l, r, mid: l.clone().add(r).multiplyScalar(0.5) };
    };
    for (const b of this.barbells) {
      b.g.visible = t >= b.a && t <= b.z;
      const h = b.g.visible && hands(b.cid);
      if (!h) continue;
      b.g.position.copy(h.mid);
      b.g.quaternion.setFromUnitVectors(V(1, 0, 0), h.l.clone().sub(h.r).normalize());
    }
    for (const c of this.carried) {
      const h = hands(c.cid);
      c.g.visible = true;
      if (t < c.attach) {
        c.g.position.copy(c.before);
      } else if (c.end == null || t < c.end) {
        if (h) {
          c.g.position.copy(h.mid).add(V(0, -0.1, 0));
          c.g.quaternion.setFromUnitVectors(V(1, 0, 0), h.l.clone().sub(h.r).setY(0).normalize());
        }
      } else {
        const u = Math.min((t - c.end) / 0.4, 1);
        c.g.position.copy(c.drop).lerp(c.after, u * u * (3 - 2 * u));
      }
    }
  }

  dispose() {
    for (const g of this.list) this.world.remove(g);
    this.list = [];
  }
}
