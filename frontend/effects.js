// Visual effects and director utilities, all a pure function of scene time:
//   teleport  - the actor dissolves out and back in with a flash ring at both ends
//   explosion - fireball, light flash, smoke column, scorch mark and camera shake
//   scale_ref - a measuring pole with the actor's height beside them
//   eyeline   - a sight-line ray from the eyes; it lights up when it crosses the lens
import * as THREE from "three";

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const smooth = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };

function textSprite(text, color = "#ffffff", scale = 0.45) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(10,14,18,0.78)";
  g.fillRect(0, 0, 256, 64);
  g.fillStyle = color;
  g.font = "600 34px Archivo, system-ui, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 128, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(scale * 4, scale, 1);
  s.renderOrder = 10;
  return s;
}

export class Effects {
  constructor(world, scene, store) {
    this.world = world;
    this.store = store;
    this.items = [];
    const add = (o) => { world.add(o); this.items.push(o); return o; };
    const J = store.ji;
    const byChar = {};
    for (const b of scene.beats) (byChar[b.character] ||= []).push(b);
    Object.values(byChar).forEach((l) => l.sort((a, b) => a.start - b.start));
    this.byChar = byChar;
    // ---- teleports
    this.teleports = [];
    const tmp = new Float32Array(store.nj * 3);
    for (const [cid, list] of Object.entries(byChar)) {
      list.forEach((b, i) => {
        if (!b.teleport) return;
        const prev = list[i - 1];
        let from = null;
        if (prev && store.clips.has(prev.clip)) {
          store.sampleBeat(prev, prev.duration, tmp);
          from = V(tmp[J.Hips * 3], 0, tmp[J.Hips * 3 + 2]);
        }
        const ring = (pos) => {
          const g = new THREE.Group();
          const r = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.035, 8, 40),
            new THREE.MeshBasicMaterial({ color: 0x9fe8ff, transparent: true, opacity: 0.9 }));
          r.rotation.x = Math.PI / 2;
          const col = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.45, 2.2, 24, 1, true),
            new THREE.MeshBasicMaterial({ color: 0xcff6ff, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }));
          col.position.y = 1.1;
          g.add(r, col);
          g.position.copy(pos);
          g.visible = false;
          return add(g);
        };
        this.teleports.push({ cid, t: b.start, out: from ? ring(from) : null, in: ring(V(b.x, 0, b.z)) });
      });
    }
    // ---- explosions
    this.blasts = (scene.layers || []).filter((l) => l.type === "fx" && l.point).map((l) => {
      const g = new THREE.Group();
      g.position.set(l.point[0], 0, l.point[1]);
      const fire = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16),
        new THREE.MeshBasicMaterial({ color: 0xffa23a, transparent: true, opacity: 0.95 }));
      const core = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xfff2b0, transparent: true }));
      const light = new THREE.PointLight(0xffa040, 0, 30, 1.6);
      light.position.y = 1.2;
      const smokeMat = new THREE.MeshStandardMaterial({ color: 0x3a3836, transparent: true, opacity: 0.0, roughness: 1, depthWrite: false });
      const puffs = Array.from({ length: 9 }, (_, i) => {
        const m = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 10), smokeMat.clone());
        m.userData = { a: (i / 9) * Math.PI * 2, r: 0.4 + (i % 3) * 0.35, rise: 1.2 + (i % 4) * 0.5 };
        g.add(m);
        return m;
      });
      const scorch = new THREE.Mesh(new THREE.CircleGeometry(2.4, 32),
        new THREE.MeshBasicMaterial({ color: 0x0c0b0a, transparent: true, opacity: 0, depthWrite: false }));
      scorch.rotation.x = -Math.PI / 2;
      scorch.position.y = 0.012;
      g.add(fire, core, light, scorch);
      add(g);
      return { t: l.start, g, fire, core, light, puffs, scorch };
    });
    // ---- scale reference poles and eye-line rays
    this.poles = [];
    this.rays = [];
    for (const [cid, list] of Object.entries(byChar)) {
      for (const b of list) {
        if (b.clip === "scale_ref") {
          const g = new THREE.Group();
          const mat = new THREE.MeshBasicMaterial({ color: 0xf2b33d });
          const pole = new THREE.Mesh(new THREE.BoxGeometry(0.03, 2.5, 0.03), mat);
          pole.position.y = 1.25;
          g.add(pole);
          for (let h = 0; h <= 2.5001; h += 0.25) {
            const tick = new THREE.Mesh(new THREE.BoxGeometry(h % 0.5 < 0.01 ? 0.22 : 0.12, 0.012, 0.012), mat);
            tick.position.y = h;
            g.add(tick);
          }
          const label = textSprite("", "#f2b33d", 0.32);
          g.add(label);
          g.visible = false;
          add(g);
          this.poles.push({ cid, b, g, label, text: "" });
        }
        if (b.clip === "eyeline_sweep") {
          const geo = new THREE.BufferGeometry().setFromPoints([V(), V(0, 0, 1)]);
          const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x4ac6ff }));
          line.visible = false;
          add(line);
          const tag = textSprite("ON LENS", "#4affa8", 0.28);
          tag.visible = false;
          add(tag);
          this.rays.push({ cid, b, line, tag });
        }
      }
    }
  }

  standinAt(cid, t) {
    return (this.byChar[cid] || []).some((b) => b.clip === "standin_turn" && t >= b.start && t <= b.start + b.duration);
  }

  opacity(cid, t) {
    let a = 1;
    for (const tp of this.teleports) {
      if (tp.cid !== cid) continue;
      if (t >= tp.t - 0.3 && t < tp.t) a = Math.min(a, 1 - smooth((t - (tp.t - 0.3)) / 0.3));
      else if (t >= tp.t && t < tp.t + 0.4) a = Math.min(a, smooth((t - tp.t) / 0.4));
    }
    return a;
  }

  shake(t) {
    let s = 0;
    for (const b of this.blasts) if (t >= b.t && t < b.t + 1.6) s = Math.max(s, 0.18 * Math.exp(-(t - b.t) * 2.6));
    return s;
  }

  update(t, poses, gaze, camera) {
    const J = this.store.ji;
    for (const tp of this.teleports) {
      for (const [ring, t0] of [[tp.out, tp.t - 0.3], [tp.in, tp.t - 0.05]]) {
        if (!ring) continue;
        const u = (t - t0) / 0.5;
        ring.visible = u >= 0 && u <= 1;
        if (!ring.visible) continue;
        ring.scale.set(0.6 + u * 0.8, 1 - 0.6 * u, 0.6 + u * 0.8);
        ring.children.forEach((m) => (m.material.opacity = (m.geometry.type === "TorusGeometry" ? 0.9 : 0.35) * (1 - u)));
      }
    }
    for (const b of this.blasts) {
      const u = t - b.t;
      const on = u >= 0;
      b.g.visible = on;
      if (!on) continue;
      const grow = smooth(u / 0.35);
      b.fire.visible = b.core.visible = u < 1.3;
      b.fire.scale.setScalar(0.3 + 2.4 * grow);
      b.fire.position.y = 0.8 + 1.0 * grow;
      b.fire.material.opacity = 0.95 * (1 - smooth((u - 0.35) / 0.9));
      b.core.scale.setScalar(0.2 + 1.3 * grow);
      b.core.position.y = 0.8 + 0.9 * grow;
      b.core.material.opacity = 1 - smooth((u - 0.15) / 0.5);
      b.light.intensity = 260 * Math.exp(-u * 4);
      for (const m of b.puffs) {
        const k = smooth((u - 0.25) / 3.5);
        const { a, r, rise } = m.userData;
        m.position.set(Math.sin(a) * r * (1 + k), 0.6 + rise * k * 2.2, Math.cos(a) * r * (1 + k));
        m.scale.setScalar(0.5 + 1.4 * k);
        m.material.opacity = u < 0.25 ? 0 : 0.55 * (1 - smooth((u - 2.5) / 2.5));
      }
      b.scorch.material.opacity = 0.6 * smooth((u - 0.2) / 0.6);
    }
    for (const p of this.poles) {
      const on = t >= p.b.start && t <= p.b.start + p.b.duration && poses[p.cid];
      p.g.visible = !!on;
      if (!on) continue;
      const q = poses[p.cid];
      const P = (j) => V(q[j * 3], q[j * 3 + 1], q[j * 3 + 2]);
      const left = P(J.LeftUpLeg).sub(P(J.RightUpLeg)); left.y = 0; left.normalize();
      const fwd = new THREE.Vector3().crossVectors(left, V(0, 1, 0));
      const hips = P(J.Hips);
      p.g.position.set(hips.x, 0, hips.z).addScaledVector(left, -1.05).addScaledVector(fwd, 0.25);
      const h = P(J.Head_End).y + 0.06;
      const text = `${h.toFixed(2)} m`;
      if (text !== p.text) {
        const c = p.label.material.map.image, g = c.getContext("2d");
        g.clearRect(0, 0, 256, 64); g.fillStyle = "rgba(10,14,18,0.78)"; g.fillRect(0, 0, 256, 64);
        g.fillStyle = "#f2b33d"; g.font = "600 34px Archivo, system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
        g.fillText(text, 128, 34);
        p.label.material.map.needsUpdate = true;
        p.text = text;
      }
      p.label.position.set(0, h + 0.05, 0);
    }
    for (const r of this.rays) {
      const on = t >= r.b.start && t <= r.b.start + r.b.duration && poses[r.cid] && gaze[r.cid];
      r.line.visible = !!on;
      r.tag.visible = false;
      if (!on) continue;
      const q = poses[r.cid];
      const head = V(q[J.Head * 3], q[J.Head * 3 + 1] + 0.03, q[J.Head * 3 + 2]);
      const g = gaze[r.cid];
      const dir = V(Math.sin(g.yaw) * Math.cos(g.pitch), Math.sin(g.pitch), Math.cos(g.yaw) * Math.cos(g.pitch));
      const toCam = camera.position.clone().sub(head);
      const along = toCam.dot(dir);
      const miss = toCam.clone().addScaledVector(dir, -along).length();
      const hit = along > 0 && miss < Math.max(0.35, along * 0.06);
      r.line.geometry.setFromPoints([head, head.clone().addScaledVector(dir, hit ? along : 30)]);
      r.line.material.color.set(hit ? 0x4affa8 : 0x4ac6ff);
      if (hit) { r.tag.visible = true; r.tag.position.copy(head).add(V(0, 0.45, 0)); }
    }
  }

  dispose() {
    for (const o of this.items) this.world.remove(o);
    this.items = [];
  }
}
