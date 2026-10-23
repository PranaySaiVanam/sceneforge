// Set dressing: furniture and hand props for everyday actions (eating, cooking, desk work).
// Each prop is sized and placed from the clip's own mocap: the surface sits just under the
// actor's lowest working hand height, in front of where the hands actually are.
import * as THREE from "three";

// clip id -> surface kind + items on it
export const PROP_SPECS = {
  eat_dinner: { surface: "table", items: ["plate", "fork"] },
  eat_casual: { surface: "table", items: ["plate"] },
  eat_soup: { surface: "table", items: ["bowl", "spoon"] },
  eat_sandwich: { surface: "table", items: ["plate", "sandwich"] },
  set_glass: { surface: "table", items: ["plate", "glass"] },
  chop: { surface: "counter", items: ["board", "onions", "knife"] },
  slice: { surface: "counter", items: ["board", "loaf"] },
  carve: { surface: "counter", items: ["board", "roast"] },
  mix: { surface: "counter", items: ["mixing_bowl"] },
  knead: { surface: "counter", items: ["board", "dough"] },
  make_coffee: { surface: "counter", items: ["kettle", "mug"] },
  typing: { surface: "desk", items: ["laptop"] },
};

const M = (color, opts = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.6, ...opts });
const MAT = {
  wood: M(0x8a5a3b, { roughness: 0.7 }),
  counter: M(0xd9d4cc, { roughness: 0.45 }),
  cabinet: M(0x5f6f7a, { roughness: 0.8 }),
  steel: M(0xc9cfd4, { roughness: 0.35, metalness: 0.3 }),
  ceramic: M(0xf4f1ea, { roughness: 0.3 }),
  board: M(0xc49a6c, { roughness: 0.8 }),
  onion: M(0xd9b38c, { roughness: 0.5 }),
  bread: M(0xc7843f, { roughness: 0.9 }),
  roast: M(0x8b3a2a, { roughness: 0.7 }),
  dough: M(0xefe0c2, { roughness: 0.95 }),
  glass: M(0xbfe3ff, { roughness: 0.05, transparent: true, opacity: 0.45 }),
  dark: M(0x2b3138, { roughness: 0.4 }),
  screen: M(0x1d3a52, { emissive: 0x16324a, emissiveIntensity: 0.8 }),
  filling: M(0x6fa85a, { roughness: 0.8 }),
};

function mesh(geo, mat, x, y, z) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = true; m.receiveShadow = true;
  return m;
}

// Measure where the hands work in a clip (clip-local frame: root starts at origin facing +Z).
export function handStats(clip, ji, nj) {
  const ys = [], xs = [], zs = [];
  for (let f = 0; f < clip.n; f += 2) {
    for (const j of [ji.LeftHand, ji.RightHand]) {
      const k = f * nj * 3 + j * 3;
      xs.push(clip.frames[k]); ys.push(clip.frames[k + 1]); zs.push(clip.frames[k + 2]);
    }
  }
  const q = (a, p) => { const s = [...a].sort((u, v) => u - v); return s[Math.floor(p * (s.length - 1))]; };
  return { low: q(ys, 0.1), x: q(xs, 0.5), z: q(zs, 0.5) };
}

function buildSurface(kind, h) {
  const g = new THREE.Group();
  if (kind === "counter") {
    const w = 1.5, d = 0.62;
    g.add(mesh(new THREE.BoxGeometry(w, h - 0.05, d), MAT.cabinet, 0, (h - 0.05) / 2, 0));
    g.add(mesh(new THREE.BoxGeometry(w + 0.04, 0.05, d + 0.04), MAT.counter, 0, h - 0.025, 0));
  } else {
    const w = kind === "desk" ? 1.2 : 1.0, d = kind === "desk" ? 0.6 : 0.7;
    g.add(mesh(new THREE.BoxGeometry(w, 0.04, d), MAT.wood, 0, h - 0.02, 0));
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      g.add(mesh(new THREE.CylinderGeometry(0.025, 0.025, h - 0.04, 8), MAT.wood, sx * (w / 2 - 0.06), (h - 0.04) / 2, sz * (d / 2 - 0.06)));
    }
  }
  return g;
}

function buildItem(name, top, hx) {
  const g = new THREE.Group();
  const y = top;
  switch (name) {
    case "plate": g.add(mesh(new THREE.CylinderGeometry(0.13, 0.11, 0.02, 24), MAT.ceramic, hx, y + 0.01, 0)); break;
    case "fork": g.add(mesh(new THREE.BoxGeometry(0.015, 0.006, 0.18), MAT.steel, hx + 0.18, y + 0.003, 0)); break;
    case "spoon": g.add(mesh(new THREE.BoxGeometry(0.02, 0.006, 0.16), MAT.steel, hx + 0.16, y + 0.003, 0)); break;
    case "bowl": {
      const b = mesh(new THREE.SphereGeometry(0.09, 20, 10, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), MAT.ceramic, hx, y + 0.09, 0);
      b.material = MAT.ceramic.clone(); b.material.side = THREE.DoubleSide; g.add(b); break;
    }
    case "mixing_bowl": {
      const b = mesh(new THREE.SphereGeometry(0.15, 24, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), MAT.steel.clone(), hx, y + 0.15, 0);
      b.material.side = THREE.DoubleSide; g.add(b);
      g.add(mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.01, 24), MAT.dough, hx, y + 0.06, 0)); break;
    }
    case "sandwich":
      g.add(mesh(new THREE.BoxGeometry(0.12, 0.025, 0.1), MAT.bread, hx, y + 0.035, 0));
      g.add(mesh(new THREE.BoxGeometry(0.125, 0.012, 0.105), MAT.filling, hx, y + 0.054, 0));
      g.add(mesh(new THREE.BoxGeometry(0.12, 0.025, 0.1), MAT.bread, hx, y + 0.073, 0)); break;
    case "glass": g.add(mesh(new THREE.CylinderGeometry(0.035, 0.03, 0.12, 16), MAT.glass, hx + 0.2, y + 0.06, -0.05)); break;
    case "board": g.add(mesh(new THREE.BoxGeometry(0.42, 0.025, 0.28), MAT.board, hx, y + 0.0125, 0)); break;
    case "onions":
      for (const [dx, dz] of [[-0.12, -0.06], [-0.08, 0.07], [0.1, 0.05]]) g.add(mesh(new THREE.SphereGeometry(0.035, 12, 10), MAT.onion, hx + dx, y + 0.06, dz));
      break;
    case "knife": g.add(mesh(new THREE.BoxGeometry(0.22, 0.004, 0.03), MAT.steel, hx + 0.32, y + 0.002, 0.08)); break;
    case "loaf": g.add(mesh(new THREE.CapsuleGeometry(0.06, 0.16, 6, 12).rotateZ(Math.PI / 2), MAT.bread, hx, y + 0.085, 0)); break;
    case "roast": g.add(mesh(new THREE.SphereGeometry(0.09, 16, 12).scale(1.3, 0.7, 1), MAT.roast, hx, y + 0.085, 0)); break;
    case "dough": g.add(mesh(new THREE.SphereGeometry(0.08, 16, 12).scale(1.3, 0.45, 1.1), MAT.dough, hx, y + 0.055, 0)); break;
    case "kettle":
      g.add(mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.16, 20), MAT.steel, hx - 0.18, y + 0.08, 0.02));
      g.add(mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.1, 8).rotateZ(-0.9), MAT.steel, hx - 0.09, y + 0.12, 0.02)); break;
    case "mug": g.add(mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.09, 16), MAT.ceramic, hx + 0.12, y + 0.045, 0)); break;
    case "laptop": {
      g.add(mesh(new THREE.BoxGeometry(0.34, 0.015, 0.23), MAT.dark, hx, y + 0.008, 0.02));
      const lid = mesh(new THREE.BoxGeometry(0.34, 0.22, 0.012), MAT.screen, hx, y + 0.12, 0.14);
      lid.rotation.x = -0.25; g.add(lid); break;
    }
  }
  return g;
}

export class PropSet {
  constructor(world, scene, store) {
    this.world = world;
    this.items = [];
    const ji = store.ji, nj = store.nj;
    const lastSurface = {}; // character -> {entry, x, z, kind}
    const beats = [...scene.beats].sort((a, b) => a.start - b.start);
    for (const b of beats) {
      const spec = PROP_SPECS[b.clip];
      if (!spec || b.auto) continue;
      const st = handStats(store.clips.get(b.clip), ji, nj);
      const top = THREE.MathUtils.clamp(st.low - 0.04, 0.62, 1.15);
      const depth = spec.surface === "counter" ? 0.62 : spec.surface === "desk" ? 0.6 : 0.7;
      const cz = Math.max(st.z, 0.25) + depth / 2 - 0.12; // front edge just inside the hands
      const prev = lastSurface[b.character];
      const sameSpot = prev && Math.hypot(prev.x - b.x, prev.z - b.z) < 0.4 && b.start - prev.entry.end < 0.6;
      let surfaceTop = top;
      if (sameSpot) {
        // Back-to-back actions at one spot share the first piece of furniture.
        prev.entry.end = b.start + b.duration + 0.4;
        surfaceTop = prev.top;
      } else {
        const sg = new THREE.Group();
        const surf = buildSurface(spec.surface, top);
        surf.position.set(st.x * 0.5, 0, cz);
        sg.add(surf);
        const entry = this.place(world, sg, b, b.start - 0.4, b.start + b.duration + 0.4);
        lastSurface[b.character] = { entry, x: b.x, z: b.z, top };
      }
      const ig = new THREE.Group();
      for (const it of spec.items) {
        const item = buildItem(it, surfaceTop, st.x);
        item.position.z += Math.max(st.z, 0.25) + 0.02; // under the working hands
        ig.add(item);
      }
      this.place(world, ig, b, b.start - (sameSpot ? 0 : 0.4), b.start + b.duration);
    }
  }

  place(world, g, b, start, end) {
    g.position.set(b.x, 0, b.z);
    g.rotation.y = b.yaw;
    g.visible = false;
    world.add(g);
    const entry = { g, start, end };
    this.items.push(entry);
    return entry;
  }

  update(t) {
    for (const p of this.items) p.g.visible = t >= p.start && t <= p.end;
  }

  dispose() {
    for (const p of this.items) this.world.remove(p.g);
    this.items = [];
  }
}
