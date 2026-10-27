// Animal bodies (dog, horse) and procedural tails, driven by the same 20-joint poses as people.
// Quadruped joint mapping (see tools/synth_motion.py): legs = hind legs, arms = front legs,
// Hips = pelvis, Spine1 = withers, Neck1 = poll, Head_End = nose.
import * as THREE from "three";

const V = () => new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

const LOOKS = {
  dog: { coat: 0xb27b4b, dark: 0x6e4a2c, body: 0.14, neck: 0.07, head: 0.075, leg: 0.032, tail: 0.38, ear: 0.07 },
  horse: { coat: 0x7b4a2a, dark: 0x2e1d12, body: 0.34, neck: 0.15, head: 0.12, leg: 0.06, tail: 0.75, ear: 0.1 },
};

export class Tail {
  // A chain of segments from the pelvis that hangs, sways with time and trails the body's motion.
  constructor(mat, length, radius, segments = 6) {
    this.group = new THREE.Group();
    this.n = segments;
    this.len = length;
    this.r = radius;
    this.segs = [];
    const geo = new THREE.CylinderGeometry(1, 1, 1, 8);
    geo.translate(0, 0.5, 0);
    for (let i = 0; i < segments; i++) {
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = true;
      this.group.add(m);
      this.segs.push(m);
    }
    this.pts = Array.from({ length: segments + 1 }, () => V());
    this.prevRoot = null;
    this.vel = V();
  }

  update(root, back, side, t, droop = 0.6, wag = 0.35, freq = 1.4) {
    // back: unit vector pointing out of the body behind the pelvis; side: body's left
    const segLen = this.len / this.n;
    if (this.prevRoot) this.vel.lerp(V().subVectors(root, this.prevRoot).multiplyScalar(30), 0.2);
    this.prevRoot = root.clone();
    this.pts[0].copy(root);
    let dir = back.clone().multiplyScalar(Math.cos(droop)).addScaledVector(UP, -Math.sin(droop) * 0.5).normalize();
    for (let i = 1; i <= this.n; i++) {
      const u = i / this.n;
      const sway = Math.sin(t * freq * Math.PI * 2 - u * 2.2) * wag * u;
      const d = dir.clone().addScaledVector(side, sway).addScaledVector(UP, -u * droop * 0.6)
        .addScaledVector(this.vel, -0.04 * u).normalize();
      this.pts[i].copy(this.pts[i - 1]).addScaledVector(d, segLen);
      if (this.pts[i].y < 0.03) this.pts[i].y = 0.03;
    }
    this.segs.forEach((m, i) => {
      const a = this.pts[i], b = this.pts[i + 1];
      const d = V().subVectors(b, a);
      const len = d.length();
      m.position.copy(a);
      m.quaternion.setFromUnitVectors(UP, d.normalize());
      const r = this.r * (1 - 0.7 * (i / this.n));
      m.scale.set(r, len, r);
    });
  }
}

export class AnimalRig {
  constructor(character, joints, makeLabel) {
    const L = LOOKS[character.kind];
    this.kind = character.kind;
    this.look = L;
    this.ji = Object.fromEntries(joints.map((n, i) => [n, i]));
    this.group = new THREE.Group();
    const coat = new THREE.MeshStandardMaterial({ color: L.coat, roughness: 0.85 });
    const dark = new THREE.MeshStandardMaterial({ color: L.dark, roughness: 0.8 });
    const accent = new THREE.MeshStandardMaterial({ color: new THREE.Color(character.color), roughness: 0.5 });
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 10);
    cyl.translate(0, 0.5, 0);
    this.body = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), coat);
    this.chest = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), coat);
    this.neck = new THREE.Mesh(cyl, coat);
    this.skull = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), coat);
    this.muzzle = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 1, 1, 12), coat);
    this.muzzle.geometry.translate(0, 0.5, 0);
    this.nose = new THREE.Mesh(new THREE.SphereGeometry(1, 10, 8), dark);
    this.ears = [0, 1].map(() => new THREE.Mesh(new THREE.ConeGeometry(0.5, 1, 8), character.kind === "dog" ? dark : coat));
    this.legs = [];
    const legBones = [["LeftUpLeg", "LeftLeg"], ["LeftLeg", "LeftFoot"], ["LeftFoot", "LeftToeBase"],
      ["RightUpLeg", "RightLeg"], ["RightLeg", "RightFoot"], ["RightFoot", "RightToeBase"],
      ["LeftArm", "LeftForeArm"], ["LeftForeArm", "LeftHand"], ["RightArm", "RightForeArm"], ["RightForeArm", "RightHand"]];
    for (const [a, b] of legBones) {
      const m = new THREE.Mesh(cyl, coat);
      const upper = /UpLeg|Arm$/.test(a);
      m.userData = { a: this.ji[a], b: this.ji[b], r: L.leg * (upper ? 1.7 : 1) };
      this.legs.push(m);
    }
    this.feet = ["LeftToeBase", "RightToeBase", "LeftHand", "RightHand"].map((n) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 10, 8), dark);
      m.userData.j = this.ji[n];
      return m;
    });
    // identification in the character's colour: a collar on the dog, a saddle blanket on the horse
    this.tag = character.kind === "dog"
      ? new THREE.Mesh(new THREE.TorusGeometry(L.neck * 1.25, L.neck * 0.25, 8, 20), accent)
      : new THREE.Mesh(new THREE.BoxGeometry(L.body * 1.9, 0.06, L.body * 1.6), accent);
    this.mane = character.kind === "horse" ? new THREE.Mesh(new THREE.BoxGeometry(0.05, 1, 0.12), dark) : null;
    this.tail = new Tail(character.kind === "horse" ? dark : coat, L.tail, character.kind === "horse" ? 0.07 : 0.035);
    const parts = [this.body, this.chest, this.neck, this.skull, this.muzzle, this.nose, ...this.ears, ...this.legs,
      ...this.feet, this.tag, this.tail.group, ...(this.mane ? [this.mane] : [])];
    parts.forEach((p) => { this.group.add(p); p.traverse?.((o) => { if (o.isMesh) o.castShadow = true; }); });
    this.label = makeLabel(character.name, character.color);
    this.group.add(this.label);
  }

  update(pose, t) {
    const L = this.look, J = this.ji;
    const P = (j) => new THREE.Vector3(pose[j * 3], pose[j * 3 + 1], pose[j * 3 + 2]);
    const hips = P(J.Hips), withers = P(J.Spine1), poll = P(J.Neck1), head = P(J.Head), nose = P(J.Head_End);
    const fwd = V().subVectors(withers, hips);
    const bodyLen = fwd.length();
    fwd.normalize();
    const left = V().subVectors(P(J.LeftUpLeg), P(J.RightUpLeg)).normalize();
    // trunk: an ellipsoid from pelvis to withers, slightly below the spine line
    const mid = V().addVectors(hips, withers).multiplyScalar(0.5).addScaledVector(UP, -L.body * 0.25);
    this.body.position.copy(mid);
    this.body.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), fwd);
    this.body.scale.set(L.body * 0.92, L.body, bodyLen * 0.62);
    this.chest.position.copy(withers).addScaledVector(UP, -L.body * 0.35).addScaledVector(fwd, -L.body * 0.1);
    this.chest.scale.set(L.body * 0.85, L.body * 0.95, L.body * 0.9);
    // neck and head
    const nb = V().copy(withers).addScaledVector(fwd, L.body * 0.25).addScaledVector(UP, -L.body * 0.05);
    const nd = V().subVectors(poll, nb);
    this.neck.position.copy(nb);
    this.neck.quaternion.setFromUnitVectors(UP, nd.clone().normalize());
    this.neck.scale.set(L.neck, nd.length(), L.neck * 1.25);
    this.skull.position.copy(poll).lerp(head, 0.5);
    this.skull.scale.set(L.head * 1.05, L.head * 1.15, L.head * 1.3);
    const md = V().subVectors(nose, head);
    this.muzzle.position.copy(head);
    this.muzzle.quaternion.setFromUnitVectors(UP, md.clone().normalize());
    this.muzzle.scale.set(L.head * 0.7, md.length(), L.head * 0.75);
    this.nose.position.copy(nose);
    this.nose.scale.setScalar(L.head * 0.38);
    this.ears.forEach((e, i) => {
      const s = i ? -1 : 1;
      e.position.copy(poll).addScaledVector(left, s * L.head * 0.6).addScaledVector(UP, L.head * 0.9);
      e.quaternion.setFromUnitVectors(UP, V().copy(UP).addScaledVector(fwd, -0.3).addScaledVector(left, s * 0.2).normalize());
      e.scale.set(L.ear * 0.6, L.ear * (this.kind === "dog" ? 1.1 : 1.4), L.ear * 0.4);
    });
    for (const m of this.legs) {
      const a = P(m.userData.a), b = P(m.userData.b);
      const d = V().subVectors(b, a);
      m.position.copy(a);
      m.quaternion.setFromUnitVectors(UP, d.clone().normalize());
      m.scale.set(m.userData.r, d.length(), m.userData.r);
    }
    for (const f of this.feet) { f.position.copy(P(f.userData.j)); f.scale.setScalar(L.leg * 1.25); }
    if (this.kind === "dog") {
      this.tag.position.copy(nb).lerp(poll, 0.35);
      this.tag.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), nd.clone().normalize());
    } else {
      this.tag.position.copy(P(J.Spine)).addScaledVector(UP, 0.02);
      this.tag.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), fwd);
      const mm = V().addVectors(nb, poll).multiplyScalar(0.5).addScaledVector(fwd, -L.neck * 0.6).addScaledVector(UP, L.neck * 0.5);
      this.mane.position.copy(mm);
      this.mane.quaternion.setFromUnitVectors(UP, nd.clone().normalize());
      this.mane.scale.set(1, nd.length() * 0.95, 1);
    }
    const back = fwd.clone().negate();
    const root = V().copy(hips).addScaledVector(back, L.body * 0.55).addScaledVector(UP, L.body * 0.25);
    const dog = this.kind === "dog";
    this.tail.update(root, back, left, t, dog ? -0.35 : 0.9, dog ? 0.45 : 0.15, dog ? 2.2 : 0.6);
    this.label.position.copy(poll).addScaledVector(UP, L.head * 2 + 0.25);
  }
}
