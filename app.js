// 北京微缩城市 — MuJoCo WASM (official google-deepmind mujoco-js) + Three.js
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import loadMujoco from "./mujoco_wasm.js";

const $ = (s) => document.querySelector(s);
const sub = $("#loadsub");

// MuJoCo geom type ids
const T = { PLANE: 0, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6 };

async function main() {
  // ---- load the official MuJoCo WebAssembly module ----
  const mujoco = await loadMujoco();
  sub.textContent = "加载模型 beijing.xml…";
  const xml = await (await fetch("./beijing.xml")).text();
  mujoco.FS.writeFile("/model.xml", xml);
  const model = mujoco.MjModel.loadFromXML("/model.xml");
  const data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);
  const names = await (await fetch("./names.json")).json();

  const ngeom = model.ngeom;
  const gtype = model.geom_type, gsize = model.geom_size, gmatid = model.geom_matid;
  const grgba = model.geom_rgba, gbody = model.geom_bodyid, matrgba = model.mat_rgba;
  const gxpos = data.geom_xpos, gxmat = data.geom_xmat;

  // ---- three.js scene (MuJoCo is z-up; rotate a root group so +z -> +y) ----
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xbcd6f0);
  scene.fog = new THREE.Fog(0xbcd6f0, 650, 1900);

  const camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 1, 5000);
  camera.position.set(360, 300, 470);

  const renderer = new THREE.WebGLRenderer({ canvas: $("#c"), antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 18, 30);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 25;
  controls.maxDistance = 1800;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x6a7a88, 1.05));
  const sun = new THREE.DirectionalLight(0xfff4e0, 1.5);
  sun.position.set(280, 420, 180);
  scene.add(sun);
  scene.add(new THREE.AmbientLight(0xffffff, 0.25));

  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;     // mujoco z-up -> three y-up
  scene.add(root);

  // ---- ground plane (the mujoco plane geom) ----
  const tex = (() => {
    const cv = document.createElement("canvas"); cv.width = cv.height = 256;
    const x = cv.getContext("2d");
    x.fillStyle = "#cdc8bc"; x.fillRect(0, 0, 256, 256);
    x.strokeStyle = "#bdb7a8"; x.lineWidth = 2;
    for (let i = 0; i <= 256; i += 32) {
      x.beginPath(); x.moveTo(i, 0); x.lineTo(i, 256); x.stroke();
      x.beginPath(); x.moveTo(0, i); x.lineTo(256, i); x.stroke();
    }
    const t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(60, 60);
    return t;
  })();
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(2400, 2400),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 1, metalness: 0 })
  );
  root.add(ground);  // PlaneGeometry normal is +z, matches mujoco plane

  // ---- bucket geoms by render primitive ----
  const buckets = { box: [], sph: [], cyl: [] };  // sph=sphere+ellipsoid, cyl=cyl+capsule
  for (let i = 0; i < ngeom; i++) {
    const t = gtype[i];
    if (t === T.BOX) buckets.box.push(i);
    else if (t === T.SPHERE || t === T.ELLIPSOID) buckets.sph.push(i);
    else if (t === T.CYLINDER || t === T.CAPSULE) buckets.cyl.push(i);
    // plane handled above; others ignored
  }

  const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 18);
  cylGeo.rotateX(Math.PI / 2);  // axis -> z (matches mujoco cylinder local z)
  const bases = {
    box: new THREE.BoxGeometry(1, 1, 1),
    sph: new THREE.SphereGeometry(1, 16, 12),
    cyl: cylGeo,
  };
  const mat = () => new THREE.MeshStandardMaterial({ roughness: 0.82, metalness: 0.05 });

  const m4 = new THREE.Matrix4(), sc = new THREE.Matrix4(), col = new THREE.Color();
  const meshes = {};
  for (const key of ["box", "sph", "cyl"]) {
    const list = buckets[key];
    const im = new THREE.InstancedMesh(bases[key], mat(), list.length);
    im.userData.geom = list;  // instanceId -> geom index
    for (let k = 0; k < list.length; k++) {
      const gi = list[k], p = gi * 3, r = gi * 9, s = gi * 3;
      m4.set(
        gxmat[r + 0], gxmat[r + 1], gxmat[r + 2], gxpos[p + 0],
        gxmat[r + 3], gxmat[r + 4], gxmat[r + 5], gxpos[p + 1],
        gxmat[r + 6], gxmat[r + 7], gxmat[r + 8], gxpos[p + 2],
        0, 0, 0, 1);
      let sx, sy, sz;
      const t = gtype[gi];
      if (t === T.BOX) { sx = 2 * gsize[s]; sy = 2 * gsize[s + 1]; sz = 2 * gsize[s + 2]; }
      else if (t === T.SPHERE) { sx = sy = sz = gsize[s]; }
      else if (t === T.ELLIPSOID) { sx = gsize[s]; sy = gsize[s + 1]; sz = gsize[s + 2]; }
      else { sx = sy = gsize[s]; sz = 2 * gsize[s + 1]; }  // cyl/capsule: r,r,length
      sc.makeScale(sx, sy, sz);
      im.setMatrixAt(k, m4.clone().multiply(sc));
      const mi = gmatid[gi];
      if (mi >= 0) col.setRGB(matrgba[mi * 4], matrgba[mi * 4 + 1], matrgba[mi * 4 + 2]);
      else col.setRGB(grgba[gi * 4], grgba[gi * 4 + 1], grgba[gi * 4 + 2]);
      im.setColorAt(k, col);
    }
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    root.add(im);
    meshes[key] = im;
  }

  // ---- click -> landmark name (bodies 1..52 are the named landmarks) ----
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  let downXY = null;
  const info = $("#info");
  renderer.domElement.addEventListener("pointerdown", (e) => { downXY = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downXY || Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]) > 6) return;
    ptr.x = (e.clientX / innerWidth) * 2 - 1;
    ptr.y = -(e.clientY / innerHeight) * 2 + 1;
    ray.setFromCamera(ptr, camera);
    const hits = ray.intersectObjects([meshes.box, meshes.sph, meshes.cyl], false);
    let found = null;
    for (const h of hits) {
      const gi = h.object.userData.geom[h.instanceId];
      const bid = gbody[gi];
      if (bid >= 1 && bid <= names.order.length) { found = names.order[bid - 1]; break; }
    }
    if (found) {
      info.querySelector(".zh").textContent = found.zh;
      info.querySelector(".en").textContent = found.key;
      info.style.opacity = 1;
    } else { info.style.opacity = 0; }
  });

  // ---- run ----
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  $("#stat").innerHTML = `${names.order.length} 地标 · ${ngeom.toLocaleString()} geoms · MuJoCo ${"WASM"}`;
  const load = $("#load");
  load.style.opacity = 0; setTimeout(() => load.remove(), 700);

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();
}

main().catch((e) => {
  console.error(e);
  $("#loadsub").textContent = "加载失败: " + (e.message || e);
  $("#loadsub").style.color = "#ff8a8a";
});
