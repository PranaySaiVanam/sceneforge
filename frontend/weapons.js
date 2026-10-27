// Hand-held weapons (gun, rifle, sword, gadha/mace, bow), muzzle flashes and arrows in flight.
// Everything is a pure function of scene time, so scrubbing backwards and recording both work.
import * as THREE from "three";

const UP = new THREE.Vector3(0, 1, 0);
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
// Original-library clips that imply a weapon (shown only while the clip plays)
const LEGACY = { sword: "sword" };

const MAT = {
  steel: new THREE.MeshStandardMaterial({ color: 0xc9d1d9, metalness: 0.85, roughness: 0.25 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x24282d, metalness: 0.5, roughness: 0.45 }),
  wood: new THREE.MeshStandardMaterial({ color: 0x7a4b2a, roughness: 0.7 }),
  gold: new THREE.MeshStandardMaterial({ color: 0xd4a017, metalness: 0.8, roughness: 0.3 }),
  leather: new THREE.MeshStandardMaterial({ color: 0x4a2f1d, roughness: 0.8 }),
  flash: new THREE.MeshBasicMaterial({ color: 0xffd36b, transparent: true, opacity: 0.95 }),
  fletch: new THREE.MeshStandardMaterial({ color: 0xd94b3b, roughness: 0.6, side: THREE.DoubleSide }),
};

function box(w, h, d, mat, y = 0, x = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  return m;
}
function cyl(r, h, mat, y = 0, seg = 10) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), mat);
  m.position.y = y;
  m.castShadow = true;
  return m;
}

// Every weapon is modelled with its length along +Y from the grip at the origin.
function makeWeapon(type) {
  const g = new THREE.Group();
  if (type === "gun") {
    g.add(box(0.035, 0.2, 0.05, MAT.dark, 0.1), box(0.03, 0.05, 0.1, MAT.dark, 0.0, 0, -0.04));
  } else if (type === "rifle") {
    g.add(box(0.04, 0.95, 0.06, MAT.dark, 0.3), box(0.045, 0.32, 0.09, MAT.wood, -0.25), box(0.03, 0.06, 0.12, MAT.dark, 0.1, 0, -0.06));
  } else if (type === "sword") {
    g.add(cyl(0.018, 0.2, MAT.leather, 0.02), box(0.17, 0.025, 0.035, MAT.gold, 0.13), box(0.048, 0.84, 0.012, MAT.steel, 0.56));
    const tip = new THREE.Mesh(new THREE.ConeGeometry(0.034, 0.1, 4), MAT.steel);
    tip.position.y = 1.03; tip.scale.z = 0.25; g.add(tip);
  } else if (type === "mace") { // gadha: long handle with a heavy round head
    g.add(cyl(0.024, 0.95, MAT.gold, 0.3), cyl(0.04, 0.06, MAT.gold, 0.74));
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.15, 18, 14), MAT.gold);
    head.position.y = 0.9; head.castShadow = true; g.add(head);
    for (let i = 0; i < 8; i++) {
      const rib = new THREE.Mesh(new THREE.TorusGeometry(0.15, 0.012, 6, 20, Math.PI), MAT.gold);
      rib.position.y = 0.9; rib.rotation.y = (i / 8) * Math.PI; rib.rotation.z = Math.PI / 2; g.add(rib);
    }
  } else if (type === "bow") {
    // limbs bend toward +Z (the target); string is drawn separately
    const arc = new THREE.Mesh(new THREE.TorusGeometry(0.75, 0.014, 6, 30, 1.9), MAT.wood);
    arc.rotation.z = Math.PI / 2 - 0.95;
    arc.rotation.y = -Math.PI / 2;
    arc.position.z = -0.62;
    g.add(arc, cyl(0.022, 0.16, MAT.leather, 0));
  }
  g.userData.type = type;
  return g;
}

function makeArrow() {
  const g = new THREE.Group();
  g.add(cyl(0.006, 0.78, MAT.wood, 0.39, 6));
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.014, 0.06, 6), MAT.steel);
  tip.position.y = 0.8; g.add(tip);
  for (let i = 0; i < 3; i++) {
    const f = new THREE.Mesh(new THREE.PlaneGeometry(0.035, 0.1), MAT.fletch);
    f.position.y = 0.05; f.rotation.y = (i / 3) * Math.PI; g.add(f);
  }
  return g;
}

export class Armory {
  constructor(world, store, scene, probe) {
    this.world = world;
    this.store = store;
    this.items = [];
    const J = store.ji;
    for (const ch of scene.characters) {
      if (ch.kind && ch.kind !== "human") continue;
      const beats = scene.beats.filter((b) => b.character === ch.id).sort((a, b) => a.start - b.start);
      const events = [{ t: -1, w: ch.weapon || null }];
      const temp = [];
      for (const b of beats) {
        const meta = store.meta[b.clip] || {};
        if (b.clip === "draw_weapon") events.push({ t: b.start + 0.45 * b.duration, w: b.weapon && b.weapon !== "none" ? b.weapon : "gun" });
        else if (b.clip === "holster") events.push({ t: b.start + 0.5 * b.duration, w: null });
        else if (b.weapon === "none") events.push({ t: b.start, w: null });
        else if (b.weapon) events.push({ t: b.start, w: b.weapon });
        const clipW = meta.weapon || LEGACY[b.clip];
        if (clipW) temp.push({ a: b.start, b: b.start + b.duration, w: clipW });
      }
      events.sort((x, y) => x.t - y.t);
      if (!events.some((e) => e.w) && !temp.length) continue;
      const meshes = {};
      for (const w of new Set([...events.map((e) => e.w), ...temp.map((x) => x.w)].filter(Boolean))) {
        meshes[w] = makeWeapon(w);
        meshes[w].visible = false;
        world.add(meshes[w]);
      }
      const string = new THREE.Line(new THREE.BufferGeometry().setFromPoints([V(), V(), V()]),
        new THREE.LineBasicMaterial({ color: 0xe8e2d0 }));
      string.visible = false;
      world.add(string);
      const flash = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), MAT.flash);
      flash.visible = false;
      world.add(flash);
      // gunshots: every 0.7 s while the shooting arm is extended
      const shots = [];
      const arrows = [];
      for (const b of beats) {
        if (b.clip === "shoot_gun" || b.clip === "shoot_rifle") {
          let last = -9;
          for (let t = b.start; t < b.start + b.duration; t += 1 / 30) {
            const p = probe(ch.id, t);
            const ext = dist(p, J.RightHand, J.RightArm);
            if (ext > 0.5 && t - last > 0.7) { shots.push(t); last = t; }
          }
        }
        if (b.clip === "shoot_bow") {
          let best = -1, rel = b.start + 1;
          for (let t = b.start; t < b.start + b.duration; t += 1 / 30) {
            const d = dist(probe(ch.id, t), J.LeftHand, J.RightHand);
            if (d > best) { best = d; rel = t; }
          }
          const p = probe(ch.id, rel);
          const from = V(p[J.LeftHand * 3], p[J.LeftHand * 3 + 1], p[J.LeftHand * 3 + 2]);
          let to;
          if (b.face) {
            const q = probe(b.face, rel + 0.25);
            to = V(q[J.Spine1 * 3], q[J.Spine1 * 3 + 1], q[J.Spine1 * 3 + 2]);
          } else {
            const back = V(p[J.RightHand * 3], p[J.RightHand * 3 + 1], p[J.RightHand * 3 + 2]);
            to = from.clone().add(from.clone().sub(back).normalize().multiplyScalar(25));
          }
          const len = from.distanceTo(to);
          const mesh = makeArrow();
          mesh.visible = false;
          world.add(mesh);
          arrows.push({ t: rel + 0.05, from, dir: to.clone().sub(from).normalize(), len, flight: len / 38, mesh });
        }
      }
      this.items.push({ ch, events, temp, meshes, string, flash, shots, arrows });
    }
  }

  weaponOf(cid, t) {
    const it = this.items.find((x) => x.ch.id === cid);
    return it ? this.weaponAt(it, t) : null;
  }

  weaponAt(item, t) {
    for (const x of item.temp) if (t >= x.a && t <= x.b) return x.w;
    let w = null;
    for (const e of item.events) if (e.t <= t) w = e.w;
    return w;
  }

  update(t, poses) {
    const J = this.store.ji;
    for (const it of this.items) {
      const p = poses[it.ch.id];
      if (!p) continue;
      const w = this.weaponAt(it, t);
      const P = (j) => V(p[j * 3], p[j * 3 + 1], p[j * 3 + 2]);
      for (const [k, m] of Object.entries(it.meshes)) m.visible = k === w;
      it.string.visible = w === "bow";
      it.flash.visible = false;
      if (w) {
        const m = it.meshes[w];
        const rh = P(J.RightHand), rf = P(J.RightForeArm), lh = P(J.LeftHand), lf = P(J.LeftForeArm);
        const fore = rh.clone().sub(rf).normalize();
        if (w === "bow") {
          const arrowDir = lh.clone().sub(lf).normalize();
          const axis = UP.clone().addScaledVector(arrowDir, -UP.dot(arrowDir)).normalize();
          const x = new THREE.Vector3().crossVectors(axis, arrowDir).normalize();
          m.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, axis, arrowDir));
          m.position.copy(lh);
          const top = lh.clone().addScaledVector(axis, 0.72).addScaledVector(arrowDir, -0.12);
          const bot = lh.clone().addScaledVector(axis, -0.72).addScaledVector(arrowDir, -0.12);
          const drawn = rh.distanceTo(lh) < 0.9 && rh.clone().sub(lh).dot(arrowDir) < 0;
          const mid = drawn ? rh : top.clone().lerp(bot, 0.5);
          it.string.geometry.setFromPoints([top, mid, bot]);
        } else if (w === "rifle" && lh.distanceTo(rh) > 0.18) {
          const dir = lh.clone().sub(rh).normalize();
          m.position.copy(rh);
          m.quaternion.setFromUnitVectors(UP, dir);
        } else {
          m.position.copy(rh);
          m.quaternion.setFromUnitVectors(UP, fore);
          if (w === "mace") m.position.addScaledVector(fore, -0.18);
        }
        if (w === "gun" || w === "rifle") {
          for (const s of it.shots) {
            if (t >= s && t < s + 0.07) {
              const tipLen = w === "rifle" ? 0.78 : 0.22;
              const dir = w === "rifle" && lh.distanceTo(rh) > 0.18 ? lh.clone().sub(rh).normalize() : fore;
              it.flash.position.copy(rh).addScaledVector(dir, tipLen);
              it.flash.scale.setScalar(0.8 + 0.6 * Math.random());
              it.flash.visible = true;
            }
          }
        }
      }
      for (const a of it.arrows) {
        const ts = t - a.t;
        a.mesh.visible = ts >= 0;
        if (ts < 0) continue;
        const d = Math.min(ts / a.flight, 1) * a.len;
        a.mesh.position.copy(a.from).addScaledVector(a.dir, d - 0.6);
        a.mesh.quaternion.setFromUnitVectors(UP, a.dir);
      }
    }
  }

  dispose() {
    for (const it of this.items) {
      for (const m of Object.values(it.meshes)) this.world.remove(m);
      this.world.remove(it.string, it.flash);
      for (const a of it.arrows) this.world.remove(a.mesh);
    }
    this.items = [];
  }
}

function dist(p, a, b) {
  return Math.hypot(p[a * 3] - p[b * 3], p[a * 3 + 1] - p[b * 3 + 1], p[a * 3 + 2] - p[b * 3 + 2]);
}
