import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { MotionStore, Performer } from "./motion.js";
import { Cinematographer } from "./camera.js";
import { PropSet } from "./props.js";
import { AnimalRig, Tail } from "./creatures.js";
import { Armory } from "./weapons.js";
import { StageProps } from "./stageprops.js";
import { Overlays } from "./layers.js";
import { VehicleRig } from "./vehicles.js";
import { Effects } from "./effects.js";

// Stage grid (mirrors sceneforge/stage.py): 20 x 20 cells of 2 m, columns A-T along X, rows 1-20 along Z.
const GRID = { cell: 2, n: 20, cols: "ABCDEFGHIJKLMNOPQRST" };
const HALF = (GRID.cell * GRID.n) / 2;

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
const SCRIPTS = [
  { title: "Zombie siege", hint: "5 zombies, sword, bow", text: `Cast: Ravi [At=Grid 10J, Weapon=Sword], Kira [Tail=Yes, Weapon=Bow, At=Grid 9H], 5 Zombies [At=Grid 14J]
Camera: Wide shot, slow push in [At=0s]
Zombies: Approach Ravi [Style=Zombie]
Kira: Shoot Zombie 1 [With=prev, Delay=4]
Ravi: Attack Zombie 2 [Weapon=Sword, Result=Fall]
Ravi: Takedown Zombie 3
Kira: Retreat from Zombie 4 [Distance=3]
Ravi: Finish Zombie 5 [Weapon=Mace]` },
  { title: "Horse ride", hint: "Mount, gallop, dog runs", text: `Cast: Arjun [At=Grid 9J], Storm [Kind=Horse, At=Grid 11J, Facing=West], Bruno [Kind=Dog, At=Grid 8L]
Arjun: Approach Storm
Arjun: Mount Storm
Storm: Move to Grid 5C [Speed=Fast]
Bruno: Move to Grid 6C [Speed=Fast, With=prev]
Camera: Tracking shot [On=Storm, At=7s]` },
  { title: "Loading the van", hint: "Lift, carry, stow", text: `Cast: Ravi [At=Grid 10J], Priya [At=Grid 10M]
Set: Van at Grid 14J [Facing=North]
Ravi: Load the luggage into the van
Priya: Lift weights [With=prev]
Ravi: Wave at Priya` },
  { title: "Bar fight", hint: "Block, roundhouse, fall", text: `Cast: Vikram [At=Grid 10H], Sam [At=Grid 10L]
Vikram: Approach Sam [Style=Angry]
Vikram: Attack Sam [Weapon=Fists, Result=Block]
Sam: Roundhouse Vikram [Result=Hit]
Vikram: Kick Sam [Result=Fall]
Vikram: Cheer
Camera: Low angle on Vikram [At=12s]` },
  { title: "Dialogue blocking", hint: "Argue, listen, pace", text: `Cast: Meera [At=Grid 10I], Kiran [At=Grid 10L]
Meera: Argue with Kiran
Kiran: Listen [With=prev]
Kiran: Say "Fine. We do it your way."
Meera: Pace between Grid 9H and Grid 11H [Times=2]
Kiran: Cower
Meera: Turn to Kiran` },
  { title: "Stealth op", hint: "Crouch, door, gun", text: `Cast: Agent [At=Grid 6F], Guard [At=Grid 10K, Weapon=Rifle]
Agent: Move to Grid 9H [Style=Crouched]
Guard: Pace between Grid 10K and Grid 10N [Times=2, With=prev]
Agent: Open door
Agent: Equip gun
Agent: Shoot Guard [Result=Fall]
Agent: Prone` },
  { title: "Obstacle run", hint: "Pivots, curb, gap, beam, wall", text: `Cast: Ravi [At=Grid 10C, Facing=North], Kira [At=Grid 13H], Meera [At=Grid 7K]
Set: Curb at Grid 10F, Gap at Grid 10I, Beam at Grid 10L, Wall at Grid 10P [Length=6, Facing=West]
Camera: Wide shot, slow push in [At=0s]
Ravi: Run to Grid 10E
Ravi: Step over curb
Kira: Look at Ravi [With=prev, For=8s]
Ravi: Squeeze through gap
Ravi: Duck under beam
Ravi: Run to Grid 10T
Meera: Hold cup
Meera: Look at Camera [With=prev]
Ravi: Turn to Meera` },
  { title: "Firefight", hint: "Strafe, startle, collapse", text: `Cast: Agent [At=Grid 8E, Weapon=Rifle], Guard [At=Grid 12K, Weapon=Gun], Clerk [At=Grid 9L]
Camera: Wide shot [At=0s]
Agent: Strafe left facing Guard [Distance=3]
Guard: Startle [From=Grid 8E, With=prev]
Clerk: Hold suitcase
Clerk: Flee from Guard [With=prev]
Guard: Strafe right facing Agent [Distance=3]
Agent: Shoot Guard [Result=Fall]
Clerk: Faint
Guard: Crawl` },
  { title: "Busy street", hint: "Crowd, chat, wallflowers, moods", text: `Cast: Ravi [At=Grid 10C], Meera [At=Grid 10Q]
Spawn Crowd [Count=6, Path=Grid 13A to Grid 13T]
Spawn 2 Patrons [Activity=Chat, At=Grid 6H]
Spawn 2 Bystanders [Activity=Idle, At=Grid 7O]
Camera: Wide shot, slow push in [At=0s]
Ravi: Walk to Grid 10I [Mood=Sad]
Meera: Check phone [With=prev]
Meera: Walk to Grid 10K [Mood=Anxious]
Ravi: Chat with Meera [For=8s]
Meera: Think [Style=Pace]` },
  { title: "Getaway", hint: "Car, bike, explosion, bail-out", text: `Cast: Ravi [At=Grid 10H], Kira [At=Grid 10J], Jeep [Kind=Car, At=Grid 8E, Facing=East], Arjun [At=Grid 14D], Moto [Kind=Bike, At=Grid 14C, Facing=East]
Camera: Wide shot [At=0s]
Ravi: Get in Jeep
Kira: Get in Jeep [Seat=Passenger, With=prev]
Arjun: Get on Moto [With=prev]
Jeep: Drive to Grid 8P [Speed=Fast]
Moto: Move to Grid 14Q [Speed=Fast, With=prev]
Explosion at Grid 10M
Ravi: Jump out
Ravi: Exhausted
Ravi: Wipe sweat` },
  { title: "Set check", hint: "T-pose, stand-in, scale, eye-line", text: `Cast: Ravi [At=Grid 10H], Kira [At=Grid 10L]
Set: Van at Grid 13J [Facing=East]
Camera: Medium shot [At=0s]
Ravi: T-Pose
Kira: A-Pose [With=prev]
Ravi: Stand-in
Kira: Scale reference next to Van [With=prev]
Ravi: Eye-line calibration
Kira: Teleport to Grid 10F
Kira: High fall [Land=Grid 8H]
Kira: Faint
Kira: Wake up` },
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
Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, near: 1, far: 45 });
world.add(hemi, sun, sun.target);
buildStage();

function buildStage() {
  // Capture-volume floor: carpet, taped boundary, centre cross and IR camera rigs.
  const floorMat = new THREE.MeshStandardMaterial({ color: 0x5d646b, roughness: 0.95 });
  const floor = new THREE.Mesh(new THREE.CircleGeometry(95, 72), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  floor.name = "floor";
  world.add(floor);
  const grid = new THREE.GridHelper(2 * HALF, GRID.n, 0x7a838c, 0x6a7279);
  grid.material.transparent = true; grid.material.opacity = 0.42;
  grid.position.y = 0.002;
  world.add(grid);
  const major = new THREE.GridHelper(2 * HALF, 4, 0x9aa3ab, 0x9aa3ab);
  major.material.transparent = true; major.material.opacity = 0.5;
  major.position.y = 0.003;
  world.add(major);
  // grid labels: letters along the near edge, numbers along the left edge (both edges for readability)
  for (let i = 0; i < GRID.n; i++) {
    const c = -HALF + (i + 0.5) * GRID.cell;
    for (const z of [-HALF - 0.9, HALF + 0.9]) world.add(floorLabel(GRID.cols[i], c, z));
    for (const x of [-HALF - 0.9, HALF + 0.9]) world.add(floorLabel(String(i + 1), x, c));
  }
  const tape = new THREE.MeshBasicMaterial({ color: 0xe9e4d6 });
  const strip = (w, d, x, z) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), tape); m.rotation.x = -Math.PI / 2; m.position.set(x, 0.004, z); world.add(m); };
  const S = 5; // 10 x 10 m inner capture volume
  strip(2 * S, 0.06, 0, S); strip(2 * S, 0.06, 0, -S); strip(0.06, 2 * S, S, 0); strip(0.06, 2 * S, -S, 0);
  strip(2 * HALF, 0.08, 0, HALF); strip(2 * HALF, 0.08, 0, -HALF); strip(0.08, 2 * HALF, HALF, 0); strip(0.08, 2 * HALF, -HALF, 0);
  strip(0.6, 0.05, 0, 0); strip(0.05, 0.6, 0, 0);
  for (const [x, z] of [[-2, 0], [2, 0]]) { strip(0.3, 0.05, x, z); strip(0.05, 0.3, x, z); }
  const metal = new THREE.MeshStandardMaterial({ color: 0x2b3138, roughness: 0.4, metalness: 0.6 });
  const ring = new THREE.MeshStandardMaterial({ color: 0xff3b30, emissive: 0xff2a1f, emissiveIntensity: 1.4 });
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2 + 0.2;
    const r = HALF + 3.5, x = Math.cos(a) * r, z = Math.sin(a) * r;
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

function floorLabel(text, x, z) {
  const c = document.createElement("canvas");
  c.width = 128; c.height = 128;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgba(233,228,214,.9)";
  ctx.font = "700 84px Archivo, system-ui, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, 64, 68);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.1), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2;
  m.position.set(x, 0.006, z);
  return m;
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
  world.fog = new THREE.Fog(L.fog, 30, 120);
  hemi.intensity = L.hemi;
  sun.intensity = L.sun;
  sun.color.set(L.sunColor);
  world.getObjectByName("floor").material.color.set(L.floor);
}

// ------------------------------------------------------------------ rigs
class Rig {
  constructor(store, character, joints, bones) {
    this.group = new THREE.Group();
    const sc = character.scale || 1;
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
      m.userData = { a, b, r: (torso ? 0.07 : thick) * sc };
      m.castShadow = true;
      this.group.add(m);
      return m;
    });
    const jointGeo = new THREE.SphereGeometry(1, 12, 10);
    this.joints = joints.map((n, i) => {
      if (n === "Head_End") return null;
      const r = n === "Hips" ? 0.11 : /Spine1/.test(n) ? 0.12 : /Hand|Foot|Toe/.test(n) ? 0.045 : 0.055;
      const m = new THREE.Mesh(jointGeo, n.startsWith("Spine") || n === "Hips" ? suit : skin);
      m.scale.setScalar(r * sc); m.castShadow = true;
      this.group.add(m);
      return m;
    });
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.115 * sc, 18, 14), skin);
    this.head.castShadow = true;
    this.group.add(this.head);
    // eye visor: shows where the actor is looking (gaze comes from the look / mood layers)
    this.visor = new THREE.Mesh(new THREE.BoxGeometry(0.15 * sc, 0.038 * sc, 0.05 * sc),
      new THREE.MeshStandardMaterial({ color: 0x14181d, roughness: 0.2, metalness: 0.6 }));
    this.visor.position.set(0, 0.018 * sc, 0.09 * sc);
    this.face = new THREE.Group();
    this.face.add(this.visor);
    this.group.add(this.face);
    this.mats = [skin, suit, marker, this.visor.material];
    this.standinMat = new THREE.MeshStandardMaterial({ color: 0x9aa1a8, roughness: 0.95, flatShading: true });
    this.standin = false;
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
    if (character.tail) { // a long, swaying tail for tailed characters
      this.tail = new Tail(skin, 1.05 * sc, 0.05 * sc, 8);
      this.group.add(this.tail.group);
    }
  }

  // Grey low-poly stand-in for lighting / lens checks (no colour, markers or visor)
  setStandin(on) {
    if (on === this.standin) return;
    this.standin = on;
    this.group.traverse((m) => {
      if (!m.isMesh || m === this.label) return;
      if (on) { m.userData.mat ||= m.material; m.material = this.standinMat; } else if (m.userData.mat) m.material = m.userData.mat;
    });
    for (const { m } of this.markers) m.visible = !on;
    this.face.visible = !on;
  }

  // Teleport dissolve
  setOpacity(a) {
    const mats = [...this.mats, this.standinMat];
    for (const m of mats) { m.transparent = a < 0.999; m.opacity = a; m.depthWrite = a > 0.5; }
    this.group.visible = a > 0.01;
  }

  update(pose, t = 0, gaze = null) {
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
    this.face.position.copy(this.head.position);
    if (gaze) this.face.rotation.set(-gaze.pitch, gaze.yaw, 0, "YXZ");
    for (const { m, j } of this.markers) P(j, m.position);
    P(this.ji.Head_End, this.label.position).add(new THREE.Vector3(0, 0.32, 0));
    if (this.tail) {
      const hips = P(this.ji.Hips, new THREE.Vector3());
      const l = P(this.ji.LeftUpLeg, new THREE.Vector3()), r = P(this.ji.RightUpLeg, new THREE.Vector3());
      const left = l.sub(r).normalize();
      const back = new THREE.Vector3(left.z, 0, -left.x).normalize(); // behind the pelvis
      this.tail.update(hips.addScaledVector(back, 0.1).add(new THREE.Vector3(0, -0.05, 0)), back, left, t, 0.75, 0.3, 0.7);
    }
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
  state.armory?.dispose();
  state.stage?.dispose();
  state.overlays?.dispose();
  state.props = new PropSet(world, scene, state.store);
  state.result = result;
  state.scene = scene;
  state.performers = scene.characters.map((c) => new Performer(state.store, c, scene.beats));
  const byId = Object.fromEntries(state.performers.map((p) => [p.character.id, p]));
  state.performers.forEach((p) => (p.others = byId));
  // evaluation order: horses before their riders
  state.order = [...state.performers.keys()].sort((i, j) => Number(state.performers[i].rides) - Number(state.performers[j].rides));
  // probes: independent performers for looking up any actor's pose at any time (props, arrows)
  const probes = Object.fromEntries(scene.characters.map((c) => [c.id, new Performer(state.store, c, scene.beats)]));
  Object.values(probes).forEach((p) => (p.others = probes));
  const probe = (cid, t) => {
    for (const p of Object.values(probes)) if (!p.rides) p.evaluate(t);
    return Float32Array.from(probes[cid].evaluate(t));
  };
  state.rigs = scene.characters.map((c) => {
    const r = c.kind === "car" || c.kind === "bike" ? new VehicleRig(c, state.library.joints, makeLabel)
      : c.kind && c.kind !== "human" ? new AnimalRig(c, state.library.joints, makeLabel)
      : new Rig(state.store, c, state.library.joints, state.library.bones);
    if (c.standin && r.setStandin) r.setStandin(true);
    r.label.visible = $("labels").checked;
    world.add(r.group); return r;
  });
  state.armory = new Armory(world, state.store, scene, probe);
  state.stage = new StageProps(world, scene, state.store, probe);
  state.overlays = new Overlays(world, state.store, scene, camera);
  state.overlays.weaponOf = (cid, t) => state.armory.weaponOf(cid, t);
  state.overlays.ladders = (t) => state.stage.laddersAt(t);
  state.effects?.dispose();
  state.effects = new Effects(world, scene, state.store);
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
  for (const i of state.order) {
    const p = state.performers[i];
    out[p.character.id] = p.evaluate(state.t);
  }
  const gaze = state.overlays?.apply(state.t, out) || {}; // mood, grip, look and IK-snap layers on top of the body
  const standins = $("standins")?.checked;
  for (const i of state.order) {
    const p = state.performers[i], r = state.rigs[i], id = p.character.id;
    if (r instanceof VehicleRig) r.update(out[id], state.t, p.motionInfo(state.t));
    else r.update(out[id], state.t, gaze[id]);
    if (r.setStandin) r.setStandin(!!(standins || p.character.standin || state.effects?.standinAt(id, state.t)));
    if (r.setOpacity) r.setOpacity(state.effects ? state.effects.opacity(id, state.t) : 1);
  }
  return out;
}

function updateSubtitles() {
  const lines = [];
  for (const b of state.scene.beats) {
    if (b.say && state.t >= b.start && state.t <= b.start + b.duration) {
      const c = state.scene.characters.find((x) => x.id === b.character);
      lines.push(`<span style="color:${c?.color || "#fff"}">${esc(c?.name || b.character)}</span> ${esc(b.say)}`);
    }
  }
  const el = $("hudSay");
  el.innerHTML = lines.join("<br>");
  el.hidden = !lines.length;
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
    state.armory?.update(state.t, ps);
    state.stage?.update(state.t, ps);
    state.effects?.update(state.t, ps, state.overlays?.gaze || {}, camera);
    updateSubtitles();
    if (state.camMode === "director") {
      const info = cine.update(state.t, ps, dt, lowActorsInShot());
      const shake = state.effects?.shake(state.t) || 0;
      if (shake) camera.position.add(new THREE.Vector3(Math.sin(state.t * 91) * shake, Math.sin(state.t * 77 + 1) * shake, 0));
      clearSightline(camera.position, cine.look);
      state.stage?.ghost(camera.position, cine.look);
      state.rigs.forEach((r) => r.ghost?.(camera.position, cine.look));
      sun.position.copy(cine.look).add(new THREE.Vector3(6, 10, 4));
      sun.target.position.copy(cine.look);
      if (info) updateHud(info);
      const show = $("labels").checked && info && WIDE_SHOTS.has(info.shot.type);
      state.rigs.forEach((r) => (r.label.visible = show));
    } else {
      orbit.update();
      state.stage?.ghost(camera.position, camera.position);
      state.rigs.forEach((r) => r.ghost?.(camera.position, camera.position.clone().add(new THREE.Vector3(0, 0, 0.01))));
      $("hudShot").textContent = "Free camera";
      state.rigs.forEach((r) => (r.label.visible = $("labels").checked));
      stageRigs.forEach((r) => (r.visible = true));
      sun.position.copy(orbit.target).add(new THREE.Vector3(6, 10, 4));
      sun.target.position.copy(orbit.target);
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
      const src = state.store.meta[b.clip]?.source;
      const kf = src === "keyframed" ? ", keyframed" : src === "studio" ? ", studio mocap" : "";
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
  const sx = $("scripts");
  SCRIPTS.forEach((e) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button"; b.title = e.text;
    b.innerHTML = `<b>${esc(e.title)}</b><small>${esc(e.hint)}</small>`;
    b.addEventListener("click", () => { $("prompt").value = e.text; $("prompt").rows = 9; updateCount(); directScene(); });
    li.append(b); sx.append(li);
  });
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
