// Virtual cinematographer: turns a shot description into a live camera.
import * as THREE from "three";

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
const lerp = (a, b, u) => a + (b - a) * u;

function segDist(p, a, b) {
  const ab = b.clone().sub(a).setY(0), ap = p.clone().sub(a).setY(0);
  const u = THREE.MathUtils.clamp(ap.dot(ab) / Math.max(ab.lengthSq(), 1e-6), 0, 1);
  return ap.sub(ab.multiplyScalar(u)).length();
}

export const lensToFov = (mm) => THREE.MathUtils.radToDeg(2 * Math.atan(12 / mm)); // full-frame, vertical

export class Cinematographer {
  constructor(camera, joints) {
    this.camera = camera;
    this.ji = Object.fromEntries(joints.map((n, i) => [n, i]));
    this.shot = null;
    this.frozen = null; // azimuth etc. decided at the start of each shot
    this.side = 1;      // 180-degree rule: which side of the action line
    this.pos = V();
    this.look = V();
  }

  setScene(scene) {
    this.scene = scene;
    this.shot = null;
    this.side = 1;
  }

  joint(pose, name) {
    const k = this.ji[name] * 3;
    return V(pose[k], pose[k + 1], pose[k + 2]);
  }

  facing(pose) {
    const l = this.joint(pose, "LeftUpLeg"), r = this.joint(pose, "RightUpLeg");
    return V(-(l.z - r.z), 0, l.x - r.x).normalize();
  }

  shotAt(t) {
    const shots = this.scene?.shots || [];
    let idx = 0;
    for (let i = 0; i < shots.length; i++) if (shots[i].start <= t + 1e-6) idx = i;
    return { shot: shots[idx], index: idx };
  }

  update(t, poses, dt, lowActors = new Set()) {
    const { shot, index } = this.shotAt(t);
    if (!shot) return null;
    const cut = this.shot !== shot;
    const subj = (shot.subjects.length ? shot.subjects : Object.keys(poses)).filter((id) => poses[id]);
    const ps = subj.map((id) => poses[id]);
    if (!ps.length) return null;
    const chest = ps.map((p) => this.joint(p, "Spine1"));
    const heads = ps.map((p) => this.joint(p, "Head"));
    const center = chest.reduce((a, b) => a.add(b), V()).multiplyScalar(1 / chest.length);
    const spread = chest.length > 1 ? Math.max(...chest.map((c) => c.distanceTo(center))) * 2 : 0;
    const u = THREE.MathUtils.clamp((t - shot.start) / Math.max(shot.duration, 0.01), 0, 1);
    const e = ease(u);

    if (cut) {
      // Decide the shot's framing once, at the cut.
      let az;
      if (chest.length >= 2) {
        const line = chest[1].clone().sub(chest[0]).setY(0).normalize();
        let perp = V(-line.z, 0, line.x);
        if (this.shot === null) this.side = 1; // first shot sets the side of the line
        az = Math.atan2(perp.x * this.side, perp.z * this.side);
      } else {
        // Three-quarter front view, nudged so no other actor blocks the lens.
        const f = this.facing(ps[0]);
        const base = Math.atan2(f.x, f.z);
        const others = Object.entries(poses).filter(([id]) => !subj.includes(id)).map(([, p]) => this.joint(p, "Spine1"));
        let best = -Infinity;
        // An actor low to the floor (pushups, crawling, lying) reads best from the side.
        const lowActor = heads[0].y < 1.0 || lowActors.has(subj[0]);
        const offsets = lowActor ? [1.45, -1.45, 1.1, -1.1, 0.7, -0.7] : [0.55, -0.55, 0.95, -0.95, 0.2, -0.2, 1.4, -1.4];
        for (const off of offsets) {
          const a = base + off;
          const cam = chest[0].clone().add(V(Math.sin(a) * 3, 0, Math.cos(a) * 3));
          let clear = 2;
          for (const o of others) clear = Math.min(clear, segDist(o, cam, chest[0]));
          const score = Math.min(clear, 1.2) * 10 - Math.abs(off);
          if (score > best) { best = score; az = a; }
        }
      }
      this.frozen = { az, center: center.clone() };
      this.shot = shot;
    }
    const az = this.frozen.az;
    const dir = V(Math.sin(az), 0, Math.cos(az));
    let pos, look;
    const head = heads[0];
    switch (shot.type) {
      case "wide": {
        const cap = (this.scene?.characters?.length || 0) > 4 ? 21 : 9.4; // big casts get a longer throw
        const d = Math.min(Math.max(7.5, spread * 1.3 + 5.5), cap);
        pos = center.clone().addScaledVector(dir, d).setY(2.7);
        look = center.clone().setY(1.0);
        break;
      }
      case "close_up": {
        pos = head.clone().addScaledVector(dir, 1.35).setY(head.y + 0.02);
        look = head.clone().add(V(0, -0.04, 0));
        break;
      }
      case "two_shot": {
        const d = spread * 1.15 + 2.8;
        pos = center.clone().addScaledVector(dir, d).setY(1.65);
        look = center.clone().setY(1.3);
        break;
      }
      case "over_shoulder": {
        const a = chest[0], b = chest[1] || a.clone().add(this.facing(ps[0]));
        const back = a.clone().sub(b).setY(0).normalize();
        const gap = a.clone().setY(0).distanceTo(b.clone().setY(0));
        const side = V(-back.z, 0, back.x).multiplyScalar(0.7 * this.side);
        pos = heads[0].clone().addScaledVector(back, 1.6 + Math.max(0, 1.6 - gap)).add(side).add(V(0, 0.2, 0));
        look = (chest[1] || b).clone().lerp(heads[1] || b, 0.6);
        break;
      }
      case "low_angle": {
        pos = center.clone().addScaledVector(dir, 3.0).setY(0.32);
        look = center.clone().add(V(0, 0.45, 0));
        break;
      }
      case "high_angle": {
        pos = center.clone().addScaledVector(dir, 3.6).setY(5.8);
        look = center.clone().setY(0.7);
        break;
      }
      case "tracking": {
        const f = this.facing(ps[0]);
        const sideV = V(f.z, 0, -f.x).multiplyScalar(this.side);
        pos = center.clone().addScaledVector(sideV, 4.2).addScaledVector(f, 0.8).setY(1.45);
        look = center.clone().addScaledVector(f, 0.4).setY(1.2);
        break;
      }
      case "pov": {
        const f = this.facing(ps[0]);
        pos = head.clone().addScaledVector(f, 0.12).add(V(0, 0.02, 0));
        const other = Object.entries(poses).find(([id]) => id !== subj[0]);
        look = other ? this.joint(other[1], "Head") : head.clone().addScaledVector(f, 5);
        break;
      }
      default: { // medium
        pos = chest[0].clone().addScaledVector(dir, 2.9).setY(head.y < 1.0 ? head.y + 0.55 : 1.5);
        look = center.clone().setY(chest.length > 1 ? 1.25 : chest[0].y + 0.05);
      }
    }
    // Camera moves
    const off = pos.clone().sub(look);
    if (shot.move === "push_in") off.multiplyScalar(lerp(1.3, 0.72, e));
    if (shot.move === "pull_out") off.multiplyScalar(lerp(0.75, 1.55, e));
    if (shot.move === "orbit") off.applyAxisAngle(V(0, 1, 0), lerp(-0.7, 0.7, e));
    if (shot.move === "crane_up") { off.y += lerp(-1.0, 3.4, e); off.multiplyScalar(lerp(0.9, 1.25, e)); }
    pos = look.clone().add(off);
    if (shot.move === "handheld") {
      pos.add(V(Math.sin(t * 2.3) * 0.035 + Math.sin(t * 5.1) * 0.012, Math.sin(t * 3.7) * 0.025, Math.cos(t * 2.9) * 0.03));
    }
    pos.y = Math.max(pos.y, 0.18);

    // Smooth follow inside a shot, hard cut between shots.
    if (cut) { this.pos.copy(pos); this.look.copy(look); }
    else {
      const k = 1 - Math.exp(-dt * (shot.type === "tracking" ? 6 : 3.5));
      this.pos.lerp(pos, k);
      this.look.lerp(look, Math.min(1, k * 1.6));
    }
    this.camera.position.copy(this.pos);
    this.camera.lookAt(this.look);
    const fov = lensToFov(shot.lens_mm || 35);
    if (Math.abs(this.camera.fov - fov) > 0.01) { this.camera.fov = fov; this.camera.updateProjectionMatrix(); }
    return { shot, index, cut };
  }
}
