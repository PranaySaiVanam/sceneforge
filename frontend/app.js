import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { MotionStore, Performer } from "./motion.js";
import { Cinematographer } from "./camera.js";
import { PropSet } from "./props.js";

const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  { title: "Fight", hint: "Circle, punch, dodge, fall", text: "Two fighters circle each other. One lunges with a punch, the other dodges and falls. Slow push-in on the winner." },
  { title: "Stealth", hint: "Night, close-up, chase", text: "Ravi sneaks into the warehouse at night while a guard paces. Close-up on Ravi as he looks around. The guard turns and runs toward him." },
  { title: "Greeting", hint: "Wave, handshake, orbit", text: "Priya waves at Arjun. Arjun walks to her and they shake hands, then both laugh. Wide shot, orbit around them." },
  { title: "Zombie", hint: "Retreat, fall, crawl", text: "A zombie shambles toward a survivor. The survivor backs away, trips and falls, then crawls. Low angle on the zombie." },
  { title: "Chef", hint: "Chop, stir, taste", text: "A chef chops vegetables in the kitchen, stirs the curry, then tastes it. Close-up on the chef." },
  { title: "Family dinner", hint: "Knead, set table, eat", text: "Mom kneads dough for chapati while her son sets the table, then eats soup." },
  { title: "Soldier", hint: "Limp, sit, crane up", text: "A tired soldier limps forward, stops and sits down. Close-up as he mopes. Crane up to end the scene." },
  { title: "Bootcamp", hint: "Shout, pushups", text: "The sergeant shouts and the recruit drops and does pushups." },
  { title: "Gym", hint: "Jumping jacks, squats", text: "A coach does jumping jacks while an athlete does squats, then lifts weights." },
];
const ASPECT = 1.85; // theatrical flat widescreen
const WIDE_SHOTS = new Set(["wide", "high_angle"]);
const TIME_LOOKS = {
  day:   { sky: 0x9fb2c2, fog: 0x9fb2c2, floor: 0x5d646b, hemi: 1.15, sun: 2.3, sunColor: 0xfff3e0 },
  dusk:  { sky: 0x6a5a6e, fog: 0x7a6670, floor: 0x4f4a50, hemi: 0.8, sun: 2.0, sunColor: 0xffb27a },
  night: { sky: 0x0f1622, fog: 0x121a26, floor: 0x2b3036, hemi: 0.35, sun: 1.1, sunColor: 0x9fb7ff },
};

// ------------------------------------------------------------------ state
const stageRigs = []; // IR camera stands around the volume
const state = {
  library: null, store: null, scene: null, result: null, performers: [], rigs: [],
  t: 0, playing: false, camMode: "director", recording: null, lastTs: 0,
};

// ------------------------------------------------------------------ renderer
const canvas = $("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
const world = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
camera.position.set(6, 3, 8);
const orbit = new OrbitControls(camera, canvas);
orbit.target.set(0, 1, 0);
orbit.enableDamping = true;
orbit.enabled = false;
const hemi = new THREE.HemisphereLight(0xdfe8f0, 0x3a3f45, 1.1);
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(6, 10, 4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12, near: 1, far: 40 });
world.add(hemi, sun, sun.target);
buildStage();

function buildStage() {
  // Capture-volume floor: carpet, taped boundary, centre cross and IR camera rigs.
  const floorMat = new THREE.MeshStandardMaterial({ color: 0x5d646b, roughness: 0.95 });
  const floor = new THREE.Mesh(new THREE.CircleGeometry(60, 64), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  floor.name = "floor";
  world.add(floor);
  const grid = new THREE.GridHelper(24, 24, 0x7a838c, 0x6a7279);
  grid.material.transparent = true; grid.material.opacity = 0.35;
  grid.position.y = 0.002;
  world.add(grid);
  const tape = new THREE.MeshBasicMaterial({ color: 0xe9e4d6 });
  const strip = (w, d, x, z) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), tape); m.rotation.x = -Math.PI / 2; m.position.set(x, 0.004, z); world.add(m); };
  const S = 5; // 10 x 10 m volume
  strip(2 * S, 0.06, 0, S); strip(2 * S, 0.06, 0, -S); strip(0.06, 2 * S, S, 0); strip(0.06, 2 * S, -S, 0);
  strip(0.6, 0.05, 0, 0); strip(0.05, 0.6, 0, 0);
  for (const [x, z] of [[-2, 0], [2, 0]]) { strip(0.3, 0.05, x, z); strip(0.05, 0.3, x, z); }
  const metal = new THREE.MeshStandardMaterial({ color: 0x2b3138, roughness: 0.4, metalness: 0.6 });
  const ring = new THREE.MeshStandardMaterial({ color: 0xff3b30, emissive: 0xff2a1f, emissiveIntensity: 1.4 });
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * Math.PI * 2 + 0.3;
    const r = 10.5, x = Math.cos(a) * r, z = Math.sin(a) * r;
    const rig = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.04, 3.2, 8), metal); pole.position.y = 1.6;
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.16, 0.26), metal); body.position.y = 3.25;
    const lens = new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.018, 8, 20), ring); lens.position.set(0, 3.25, 0.14);
    rig.add(pole, body, lens);
    rig.position.set(x, 0, z);
    rig.lookAt(0, 0, 0);
    rig.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    world.add(rig);
    stageRigs.push(rig);
  }
}

// Characters who spend this shot near the floor (pushups, crawling): the camera favours a side view.
function lowActorsInShot() {
  const { shot } = cine.shotAt(state.t);
  const low = new Set();
  if (!shot) return low;
  const end = shot.start + shot.duration;
  for (const b of state.scene.beats) {
    if (b.start < end && b.start + b.duration > shot.start && (state.store.meta[b.clip]?.height ?? 2) < 1.0) low.add(b.character);
  }
  return low;
}

// Like moving a stand out of shot: hide camera rigs between the lens and the action.
function clearSightline(cam, look) {
  const ax = look.x - cam.x, az = look.z - cam.z, len2 = ax * ax + az * az || 1;
  for (const rig of stageRigs) {
    const px = rig.position.x - cam.x, pz = rig.position.z - cam.z;
    const u = (px * ax + pz * az) / len2;
    const dx = px - u * ax, dz = pz - u * az;
    rig.visible = !(u > -0.05 && u < 1 && Math.hypot(dx, dz) < 1.2);
  }
}

function applyLook(tod) {
  const L = TIME_LOOKS[tod] || TIME_LOOKS.day;
  world.background = new THREE.Color(L.sky);
  world.fog = new THREE.Fog(L.fog, 16, 55);
  hemi.intensity = L.hemi;
  sun.intensity = L.sun;
  sun.color.set(L.sunColor);
  world.getObjectByName("floor").material.color.set(L.floor);
}

// ------------------------------------------------------------------ rigs
class Rig {
  constructor(store, character, joints, bones) {
    this.group = new THREE.Group();
    this.ji = Object.fromEntries(joints.map((n, i) => [n, i]));
    this.bones = bones;
    const col = new THREE.Color(character.color);
    const skin = new THREE.MeshStandardMaterial({ color: col, roughness: 0.55, metalness: 0.05 });
    const suit = new THREE.MeshStandardMaterial({ color: col.clone().multiplyScalar(0.42), roughness: 0.75 });
    const marker = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.6, roughness: 0.2 });
    const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 10);
    unitCyl.translate(0, 0.5, 0);
    this.limbs = bones.map(([a, b]) => {
      const name = joints[b];
      const thick = /Spine|Neck|Hips/.test(name) ? 0.085 : /UpLeg|Leg$|Arm$/.test(name) ? 0.058 : 0.045;
      const torso = /Spine|LeftUpLeg|RightUpLeg|Arm$/.test(name) && /Spine|Hips/.test(joints[a]);
      const m = new THREE.Mesh(unitCyl, torso ? suit : skin);
      m.userData = { a, b, r: torso ? 0.07 : thick };
      m.castShadow = true;
      this.group.add(m);
      return m;
    });
    const jointGeo = new THREE.SphereGeometry(1, 12, 10);
    this.joints = joints.map((n, i) => {
      if (n === "Head_End") return null;
      const r = n === "Hips" ? 0.11 : /Spine1/.test(n) ? 0.12 : /Hand|Foot|Toe/.test(n) ? 0.045 : 0.055;
      const m = new THREE.Mesh(jointGeo, n.startsWith("Spine") || n === "Hips" ? suit : skin);
      m.scale.setScalar(r); m.castShadow = true;
      this.group.add(m);
      return m;
    });
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.115, 18, 14), skin);
    this.head.castShadow = true;
    this.group.add(this.head);
    // retroreflective mocap markers
    const mgeo = new THREE.SphereGeometry(0.018, 8, 6);
    this.markers = ["Head_End", "LeftHand", "RightHand", "LeftForeArm", "RightForeArm", "LeftFoot", "RightFoot",
                    "LeftLeg", "RightLeg", "Spine1", "Hips", "LeftArm", "RightArm"].map((n) => {
      const m = new THREE.Mesh(mgeo, marker); this.group.add(m); return { m, j: this.ji[n] };
    });
    this.label = makeLabel(character.name, character.color);
    this.group.add(this.label);
    this.tmpA = new THREE.Vector3(); this.tmpB = new THREE.Vector3();
    this.up = new THREE.Vector3(0, 1, 0);
  }

  update(pose) {
    const P = (j, v) => v.set(pose[j * 3], pose[j * 3 + 1], pose[j * 3 + 2]);
    for (const m of this.limbs) {
      const { a, b, r } = m.userData;
      P(a, this.tmpA); P(b, this.tmpB);
      const dir = this.tmpB.sub(this.tmpA);
      const len = dir.length();
      m.position.copy(this.tmpA);
      m.quaternion.setFromUnitVectors(this.up, dir.normalize());
      m.scale.set(r, len, r);
    }
    this.joints.forEach((m, i) => m && P(i, m.position));
    const h = P(this.ji.Head, this.tmpA), he = P(this.ji.Head_End, this.tmpB);
    this.head.position.copy(h).lerp(he, 0.45);
    for (const { m, j } of this.markers) P(j, m.position);
    P(this.ji.Head_End, this.label.position).add(new THREE.Vector3(0, 0.32, 0));
  }
}

function makeLabel(text, color) {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  const font = "600 44px Archivo, system-ui, sans-serif";
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 48;
  c.width = w; c.height = 68;
  ctx.font = font;
  ctx.fillStyle = "rgba(20,26,32,.82)";
  ctx.beginPath(); ctx.roundRect(0, 0, w, 68, 14); ctx.fill();
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(26, 34, 9, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(text, 44, 36);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set(0.0034 * w, 0.0034 * 68, 1);
  s.renderOrder = 10;
  return s;
}

// ------------------------------------------------------------------ scene load
async function loadScene(result) {
  const scene = result.scene;
  await state.store.load(scene.beats.map((b) => b.clip));
  state.rigs.forEach((r) => world.remove(r.group));
  state.props?.dispose();
  state.props = new PropSet(world, scene, state.store);
  state.result = result;
  state.scene = scene;
  state.performers = scene.characters.map((c) => new Performer(state.store, c, scene.beats));
  state.rigs = scene.characters.map((c) => {
    const r = new Rig(state.store, c, state.library.joints, state.library.bones);
    r.label.visible = $("labels").checked;
    world.add(r.group); return r;
  });
  cine.setScene(scene);
  applyLook(scene.setting.time_of_day);
  $("sceneTitle").textContent = scene.title;
  $("empty").hidden = true;
  state.t = 0;
  renderTimeline();
  renderTrace();
  renderShots();
  $("json").textContent = JSON.stringify(scene, null, 2);
  setPlaying(true);
}

// ------------------------------------------------------------------ playback
let cine = new Cinematographer(camera, []);
function setPlaying(on) {
  state.playing = on;
  $("playIcon").setAttribute("d", on ? "M7 5h4v14H7zM13 5h4v14h-4z" : "M8 5v14l11-7z");
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
}

function poses() {
  const out = {};
  state.performers.forEach((p, i) => { out[p.character.id] = p.evaluate(state.t); state.rigs[i].update(out[p.character.id]); });
  return out;
}

function frame(ts) {
  const dt = Math.min(0.05, (ts - (state.lastTs || ts)) / 1000);
  state.lastTs = ts;
  resize();
  if (state.scene) {
    if (state.playing) {
      state.t += dt;
      if (state.t >= state.scene.duration) {
        if (state.recording) stopRecording();
        if ($("loop").checked && !state.recording) state.t = 0;
        else { state.t = state.scene.duration; setPlaying(false); }
      }
    }
    const ps = poses();
    state.props?.update(state.t);
    if (state.camMode === "director") {
      const info = cine.update(state.t, ps, dt, lowActorsInShot());
      clearSightline(camera.position, cine.look);
      if (info) updateHud(info);
      const show = $("labels").checked && info && WIDE_SHOTS.has(info.shot.type);
      state.rigs.forEach((r) => (r.label.visible = show));
    } else {
      orbit.update();
      $("hudShot").textContent = "Free camera";
      state.rigs.forEach((r) => (r.label.visible = $("labels").checked));
      stageRigs.forEach((r) => (r.visible = true));
    }
    updatePlayhead();
  }
  draw();
  requestAnimationFrame(frame);
}

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, w, h);
  if (state.camMode === "director") {
    // 1.85:1 letterbox rendered into the canvas so recordings carry it too
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    let vw = w, vh = Math.round(w / ASPECT);
    if (vh > h) { vh = h; vw = Math.round(h * ASPECT); }
    const x = Math.round((w - vw) / 2), y = Math.round((h - vh) / 2);
    renderer.setViewport(x, y, vw, vh);
    renderer.setScissor(x, y, vw, vh);
    renderer.setScissorTest(true);
    camera.aspect = vw / vh;
  } else {
    camera.aspect = w / h;
  }
  camera.updateProjectionMatrix();
  renderer.render(world, camera);
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pr = renderer.getPixelRatio();
  if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) renderer.setSize(w, h, false);
}

const SHOT_NAMES = { wide: "Wide", medium: "Medium", close_up: "Close-up", two_shot: "Two-shot", over_shoulder: "Over the shoulder",
  low_angle: "Low angle", high_angle: "High angle", tracking: "Tracking", pov: "POV" };
const MOVE_NAMES = { static: "", push_in: "push in", pull_out: "pull out", orbit: "orbit", crane_up: "crane up", handheld: "handheld" };
function updateHud({ shot, index }) {
  const mv = MOVE_NAMES[shot.move];
  $("hudShot").innerHTML = `<b>Shot ${index + 1}/${state.scene.shots.length}</b>${SHOT_NAMES[shot.type]}${mv ? ", " + mv : ""}, ${Math.round(shot.lens_mm)} mm`;
  document.querySelectorAll(".tl-block.shot").forEach((el, i) => el.classList.toggle("active", i === index));
  document.querySelectorAll("#tab-shots tr[data-t]").forEach((el, i) => el.classList.toggle("active", i === index));
}

function timecode(t, fps = 24) {
  const f = Math.floor(t * fps);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(Math.floor(f / (3600 * fps)))}:${p(Math.floor(f / (60 * fps)) % 60)}:${p(Math.floor(f / fps) % 60)}:${p(f % fps)}`;
}

// ------------------------------------------------------------------ timeline
function renderTimeline() {
  const tl = $("timeline");
  const sc = state.scene;
  const D = sc.duration;
  const pct = (t) => `${(t / D) * 100}%`;
  const step = D > 30 ? 5 : D > 12 ? 2 : 1;
  let html = `<div class="tl-ruler">`;
  for (let t = 0; t <= D + 1e-6; t += step) html += `<span style="left:${pct(t)}">${t}s</span>`;
  html += `</div><div class="tl-track"><div class="tl-label">Camera</div>`;
  sc.shots.forEach((s, i) => {
    html += `<div class="tl-block shot" style="left:${pct(s.start)};width:${pct(s.duration)}" title="${esc(s.note || "")}">${i + 1}. ${SHOT_NAMES[s.type]}</div>`;
  });
  html += `</div>`;
  for (const c of sc.characters) {
    html += `<div class="tl-track"><div class="tl-label" title="${esc(c.name)}"><i style="background:${c.color}"></i>${esc(c.name)}</div>`;
    for (const b of sc.beats.filter((b) => b.character === c.id)) {
      const kf = state.store.meta[b.clip]?.source === "keyframed" ? ", keyframed" : "";
      const tip = b.auto ? "idle (added by continuity)" : `${b.action}\nclip: ${b.clip}${kf} (confidence ${Math.round(b.confidence * 100)}%)`;
      html += `<div class="tl-block${b.auto ? " auto" : ""}" style="left:${pct(b.start)};width:${pct(b.duration)};background:${c.color}" title="${esc(tip)}">${b.auto ? "idle" : esc(b.clip.replace(/_/g, " "))}</div>`;
    }
    html += `</div>`;
  }
  html += `<div class="tl-head" id="tlHead"></div>`;
  tl.innerHTML = html;
}

function updatePlayhead() {
  const tl = $("timeline"), head = $("tlHead");
  if (!head) return;
  const pl = parseFloat(getComputedStyle(tl).paddingLeft), pr = parseFloat(getComputedStyle(tl).paddingRight);
  const w = tl.clientWidth - pl - pr;
  head.style.left = `${pl + (state.t / state.scene.duration) * w}px`;
  $("hudTc").textContent = timecode(state.t);
}

function scrubFromEvent(e) {
  const tl = $("timeline");
  const r = tl.getBoundingClientRect();
  const pl = parseFloat(getComputedStyle(tl).paddingLeft), pr = parseFloat(getComputedStyle(tl).paddingRight);
  const u = (e.clientX - r.left - pl) / (r.width - pl - pr);
  state.t = Math.max(0, Math.min(1, u)) * state.scene.duration;
}
let scrubbing = false;
$("timeline").addEventListener("pointerdown", (e) => { if (!state.scene) return; scrubbing = true; setPlaying(false); scrubFromEvent(e); $("timeline").setPointerCapture(e.pointerId); });
$("timeline").addEventListener("pointermove", (e) => scrubbing && scrubFromEvent(e));
$("timeline").addEventListener("pointerup", () => { scrubbing = false; });

// ------------------------------------------------------------------ inspector
function renderTrace() {
  const r = state.result;
  const meta = r.trace.find((s) => s.agent === "Director");
  const fixCount = r.trace.reduce((n, s) => n + (s.fixes || []).length, 0);
  let html = `<div class="summary"><div><b>${r.total_ms} ms</b><span>end to end</span></div>` +
    `<div><b>${state.scene.beats.filter((b) => !b.auto).length}</b><span>beats</span></div>` +
    `<div><b>${fixCount}</b><span>auto-fix${fixCount === 1 ? "" : "es"}</span></div></div>`;
  if (meta?.tokens) html += `<p class="step-note">${meta.tokens.input + meta.tokens.output} tokens</p>`;
  r.trace.forEach((s, i) => {
    const fb = s.fallback_from || (s.agent === "Director" && s.provider === "offline");
    html += `<div class="step${fb ? " fallback" : ""}"><span class="num">${i + 1}</span><h4>${s.agent}<small>${s.ms} ms</small></h4><p>${esc(s.summary)}</p>`;
    if (s.matches) {
      html += `<div class="match">`;
      for (const m of s.matches) html += `<span title="${esc(m.action)}">${esc(m.action)}</span><span><b>${m.clip}</b><span class="conf"><i style="width:${Math.round(m.confidence * 100)}%"></i></span></span>`;
      html += `</div>`;
    }
    const fixes = (s.fixes || []).concat(s.errors || []);
    if (fixes.length) html += `<ul>${fixes.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>`;
    html += `</div>`;
  });
  $("tab-trace").innerHTML = html;
}

function renderShots() {
  const sc = state.scene;
  const name = (id) => sc.characters.find((c) => c.id === id)?.name || id;
  let html = `<table><thead><tr><th>#</th><th>Time</th><th>Shot</th><th>Lens</th><th>On</th></tr></thead><tbody>`;
  sc.shots.forEach((s, i) => {
    html += `<tr data-t="${s.start}"><td>${i + 1}</td><td>${s.start.toFixed(1)}s<br><small>${s.duration.toFixed(1)}s</small></td>` +
      `<td>${SHOT_NAMES[s.type]}${MOVE_NAMES[s.move] ? `<br><small>${MOVE_NAMES[s.move]}</small>` : ""}</td>` +
      `<td>${Math.round(s.lens_mm)} mm</td><td>${s.subjects.map(name).map(esc).join(", ")}</td></tr>`;
  });
  html += `</tbody></table>`;
  $("tab-shots").innerHTML = html;
  $("tab-shots").querySelectorAll("tr[data-t]").forEach((tr) => tr.addEventListener("click", () => { state.t = +tr.dataset.t + 0.01; }));
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${b.dataset.tab}`; });
}));

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const slug = () => (state.scene?.title || "scene").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
$("dlJson").addEventListener("click", () => state.scene && download(`${slug()}.json`, JSON.stringify(state.scene, null, 2), "application/json"));
$("dlCsv").addEventListener("click", () => {
  if (!state.scene) return;
  const sc = state.scene;
  const rows = [["shot", "start_s", "duration_s", "type", "move", "lens_mm", "subjects", "note"]];
  sc.shots.forEach((s, i) => rows.push([i + 1, s.start, s.duration, s.type, s.move, s.lens_mm,
    s.subjects.map((id) => sc.characters.find((c) => c.id === id)?.name || id).join(" & "), s.note]));
  download(`${slug()}-shots.csv`, rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv");
});

// ------------------------------------------------------------------ recording
$("record").addEventListener("click", () => (state.recording ? stopRecording() : startRecording()));
function startRecording() {
  if (!state.scene || !canvas.captureStream || !window.MediaRecorder) return setStatus("This browser can't record the canvas.", true);
  setCam("director");
  const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"].find((t) => MediaRecorder.isTypeSupported(t));
  const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: type, videoBitsPerSecond: 8e6 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    const ext = (type || "").includes("mp4") ? "mp4" : "webm";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(chunks, { type }));
    a.download = `${slug()}-previs.${ext}`; a.click();
    setStatus("Video saved.");
  };
  state.recording = rec;
  state.t = 0;
  setPlaying(true);
  rec.start();
  $("record").classList.add("on"); $("record").querySelector("span").textContent = "Stop recording";
  $("hudRec").hidden = false;
}
function stopRecording() {
  state.recording?.stop();
  state.recording = null;
  $("record").classList.remove("on"); $("record").querySelector("span").textContent = "Record video";
  $("hudRec").hidden = true;
}

// ------------------------------------------------------------------ controls
function setCam(mode) {
  state.camMode = mode;
  document.querySelectorAll(".seg button").forEach((b) => { const on = b.dataset.cam === mode; b.classList.toggle("on", on); b.setAttribute("aria-checked", on); });
  orbit.enabled = mode === "free";
  if (mode === "free") { camera.fov = 40; orbit.target.copy(cine.look); cine.shot = null; }
}
document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setCam(b.dataset.cam)));
$("play").addEventListener("click", () => { if (state.scene && state.t >= state.scene.duration) state.t = 0; setPlaying(!state.playing); });
$("restart").addEventListener("click", () => { state.t = 0; cine.shot = null; });
$("labels").addEventListener("change", (e) => state.rigs.forEach((r) => (r.label.visible = e.target.checked)));
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && document.activeElement.tagName !== "TEXTAREA" && document.activeElement.tagName !== "BUTTON") {
    e.preventDefault(); $("play").click();
  }
});

function setStatus(msg, err = false, busy = false, warn = false) {
  const s = $("status");
  s.classList.toggle("warn", warn);
  s.querySelector("span").textContent = msg;
  s.title = msg;
  s.classList.toggle("err", err);
  s.classList.toggle("busy", busy);
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function directScene() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return setStatus("Write a scene first.", true);
  const provider = $("provider").value;
  if (/[^\x00-\x7F]/.test(prompt) && (provider === "offline" || (provider === "auto" && state.defaultProvider === "offline"))) {
    setStatus("Hindi and Telugu scenes need an LLM planner. Add an API key on the server.", true);
    return;
  }
  $("go").disabled = true;
  $("loading").hidden = false;
  setStatus("Directing scene", false, true);
  try {
    const res = await fetch("/api/direct", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, provider }) });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || body.error || `HTTP ${res.status}`);
    await loadScene(body);
    const d = body.trace.find((s) => s.agent === "Director");
    const missing = body.scene.unmatched || [];
    if (missing.length) setStatus(`Not in the motion library yet: ${missing.map((m) => `"${m}"`).join(", ")}`, false, false, true);
    else setStatus(`Directed in ${body.total_ms} ms, ${d?.model ? d.model : "rule-based planner"}`);
  } catch (err) {
    setStatus(`Couldn't direct the scene: ${err.message}`, true);
  } finally {
    $("go").disabled = false;
    $("loading").hidden = true;
  }
}
$("go").addEventListener("click", directScene);
function updateCount() { $("count").textContent = `${$("prompt").value.length}/2000`; }
$("prompt").addEventListener("input", updateCount);
$("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) directScene(); });

// Voice dictation (Web Speech API; Chrome/Edge)
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SR) { $("mic").disabled = true; $("mic").title = "Dictation needs Chrome or Edge"; }
let recog = null;
$("mic").addEventListener("click", () => {
  if (recog) { recog.stop(); return; }
  recog = new SR();
  recog.lang = $("lang").value;
  recog.interimResults = true;
  recog.continuous = true;
  const base = $("prompt").value.trim();
  let finalText = "";
  recog.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) finalText += e.results[i][0].transcript + " ";
      else interim += e.results[i][0].transcript;
    }
    $("prompt").value = (base && finalText ? "" : base ? base + " " : "") + finalText + interim;
  };
  recog.onend = () => { recog = null; $("mic").setAttribute("aria-pressed", "false"); $("mic").querySelector("span").textContent = "Dictate"; updateCount(); };
  recog.onerror = (e) => setStatus(`Dictation stopped: ${e.error}`, true);
  $("prompt").value = "";
  recog.start();
  $("mic").setAttribute("aria-pressed", "true");
  $("mic").querySelector("span").textContent = "Stop";
});

// ------------------------------------------------------------------ boot
async function boot() {
  const ex = $("examples");
  EXAMPLES.forEach((e) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button"; b.title = e.text;
    b.innerHTML = `<b>${esc(e.title)}</b><small>${esc(e.hint)}</small>`;
    b.addEventListener("click", () => { $("prompt").value = e.text; updateCount(); directScene(); });
    li.append(b); ex.append(li);
  });
  updateCount();
  try {
    const [lib, prov] = await Promise.all([fetch("/api/library").then((r) => r.json()), fetch("/api/providers").then((r) => r.json())]);
    state.library = lib;
    state.store = new MotionStore(lib);
    cine = new Cinematographer(camera, lib.joints);
    state.defaultProvider = prov.default;
    const sel = $("provider");
    const NAMES = { offline: "Rule-based (no API key)", anthropic: "Claude", gemini: "Gemini", openai: "OpenAI" };
    sel.innerHTML = `<option value="auto">Automatic (${prov.default === "offline" ? "rule-based" : NAMES[prov.default]})</option>` +
      prov.available.map((p) => `<option value="${p}">${p === "offline" ? NAMES.offline : `${NAMES[p]} (${prov.models[p]})`}</option>`).join("");
    applyLook("day");
    await directScene();
  } catch (err) {
    setStatus(`Couldn't reach the SceneForge server: ${err.message}`, true);
    $("empty").hidden = false;
  }
  requestAnimationFrame(frame);
}
// test/automation hook
window.__sf = { get scene() { return state.scene; }, seek(t) { state.t = t; cine.shot = null; }, state };
boot();
