// Cars and motorbikes. They move like any other actor (rigid-body "skeleton" clips: Hips = chassis
// centre, legs = rear axle, arms = front axle, Spine = nose), so blocking, approach and timing all
// work unchanged. Riders sit in seats and lean / steer with the path (see motion.js seatInVehicle).
import * as THREE from "three";

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const M = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.5, ...o });

function box(w, h, d, mat, x, y, z) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  return m;
}

function wheel(r, w, mat, hub) {
  const g = new THREE.Group();
  const tyre = new THREE.Mesh(new THREE.CylinderGeometry(r, r, w, 20), mat);
  tyre.rotation.z = Math.PI / 2;
  tyre.castShadow = true;
  const cap = new THREE.Mesh(new THREE.BoxGeometry(w + 0.01, r * 1.2, r * 0.25), hub);
  g.add(tyre, cap);
  return g;
}

export class VehicleRig {
  constructor(character, joints, makeLabel) {
    this.kind = character.kind;
    this.ji = Object.fromEntries(joints.map((n, i) => [n, i]));
    this.group = new THREE.Group();
    this.body = new THREE.Group();
    this.group.add(this.body);
    const paint = M(new THREE.Color(character.color).multiplyScalar(0.9), { metalness: 0.35, roughness: 0.35 });
    const dark = M(0x1b1f24, { roughness: 0.9 });
    const chrome = M(0xc9ced4, { metalness: 0.9, roughness: 0.25 });
    const glass = M(0x9ec4dd, { transparent: true, opacity: 0.28, roughness: 0.05, metalness: 0.2 });
    this.wheels = [];
    if (this.kind === "car") {
      this.body.add(box(1.82, 0.55, 4.3, paint, 0, 0.62, 0));
      this.body.add(box(1.7, 0.08, 4.36, dark, 0, 0.36, 0));
      // open-sided glass cabin so the seated actors stay visible
      this.body.add(box(1.62, 0.5, 2.1, glass, 0, 1.15, -0.25));
      this.body.add(box(1.62, 0.05, 2.1, paint, 0, 1.42, -0.25));
      for (const [x, z] of [[0.8, 0.8], [-0.8, 0.8], [0.8, -1.3], [-0.8, -1.3]]) {
        this.body.add(box(0.06, 0.5, 0.06, paint, x, 1.15, z));
      }
      this.wheelBase = [[0.86, 1.35], [-0.86, 1.35], [0.86, -1.35], [-0.86, -1.35]];
      this.r = 0.34;
      // steering wheel in front of the driver's seat
      this.wheelSteer = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.025, 8, 24), dark);
      this.wheelSteer.position.set(0.38, 1.0, 0.68);
      this.wheelSteer.rotation.x = -1.1;
      this.body.add(this.wheelSteer);
      this.body.add(box(0.12, 0.1, 0.05, chrome, 0.62, 0.66, 2.16), box(0.12, 0.1, 0.05, chrome, -0.62, 0.66, 2.16));
    } else {
      this.body.add(box(0.26, 0.3, 1.1, paint, 0, 0.62, 0));            // tank / frame
      this.body.add(box(0.28, 0.1, 0.62, dark, 0, 0.84, -0.25));         // seat
      this.body.add(box(0.06, 0.6, 0.06, chrome, 0, 0.75, 0.62));        // forks
      const bar = box(0.7, 0.04, 0.04, chrome, 0, 1.05, 0.55);
      this.body.add(bar);
      this.wheelBase = [[0, 0.7], [0, -0.7]];
      this.r = 0.33;
    }
    for (const [x, z] of this.wheelBase) {
      const w = wheel(this.r, this.kind === "car" ? 0.24 : 0.12, dark, chrome);
      w.position.set(x, this.r, z);
      this.body.add(w);
      this.wheels.push(w);
    }
    this.label = makeLabel(character.name, character.color);
    this.group.add(this.label);
    this.prev = null;
    this.spin = 0;
    this.lean = 0;
  }

  // see-through when it blocks the director's camera (or the lens is inside it)
  ghost(cam, look) {
    this.ray ||= new THREE.Raycaster();
    const dir = look.clone().sub(cam);
    const len = dir.length();
    this.ray.set(cam, dir.normalize());
    this.ray.far = Math.max(len - 0.5, 0.1);
    this.group.updateMatrixWorld();
    const box = new THREE.Box3().setFromObject(this.body).expandByScalar(0.3);
    const hide = box.containsPoint(cam) || this.ray.intersectObject(this.body, true).length > 0;
    this.body.traverse((m) => {
      if (!m.isMesh) return;
      m.userData.mat ||= m.material;
      m.userData.ghost ||= Object.assign(m.material.clone(), { transparent: true, opacity: 0.14, depthWrite: false });
      m.material = hide ? m.userData.ghost : m.userData.mat;
    });
  }

  update(pose, t, info = {}) {
    const J = this.ji;
    const P = (j) => V(pose[j * 3], pose[j * 3 + 1], pose[j * 3 + 2]);
    const hips = P(J.Hips), l = P(J.LeftUpLeg), r = P(J.RightUpLeg);
    const left = l.clone().sub(r); left.y = 0; left.normalize();
    const yaw = Math.atan2(-left.z, left.x);
    this.group.position.set(hips.x, hips.y - (this.kind === "car" ? 0.6 : 0.55), hips.z);
    this.group.rotation.set(0, yaw, 0);
    // wheels roll with the distance travelled (sampled per frame; scrubbing just re-rolls them)
    if (this.prev) {
      const d = hips.clone().sub(this.prev); d.y = 0;
      const fwd = V(Math.sin(yaw), 0, Math.cos(yaw));
      const s = d.dot(fwd);
      if (Math.abs(s) < 3) this.spin += s / this.r;
    }
    this.prev = hips.clone();
    for (const w of this.wheels) w.rotation.x = this.spin;
    this.lean = info.lean || 0;
    if (this.kind === "bike") this.body.rotation.z = -this.lean;
    else if (this.wheelSteer) this.wheelSteer.rotation.z = info.steer || 0;
    if (this.kind === "car") for (let i = 0; i < 2; i++) this.wheels[i].rotation.y = (info.steer || 0) * 0.35;
    this.label.position.set(0, this.kind === "car" ? 2.0 : 1.7, 0);
  }
}
