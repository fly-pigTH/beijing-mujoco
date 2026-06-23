// 北京微缩城市 — MuJoCo WASM (official google-deepmind mujoco-js) + Three.js
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";
import loadMujoco from "./mujoco_wasm.js";

const $ = (s) => document.querySelector(s);
const sub = $("#loadsub");

// MuJoCo geom type ids
const T = { PLANE: 0, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6 };

// device tier -> quality / perf trade-offs (mobile-friendly)
const MOBILE = matchMedia("(pointer:coarse)").matches ||
  /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
const SHADOWS = !MOBILE;
// curve tessellation — high facet counts for smooth domes / towers / shells.
// (one shared base geometry per primitive, so cost stays low even at 12k geoms)
const SEG = MOBILE ? { s: [24, 16], c: 28 } : { s: [40, 28], c: 48 };

async function main() {
  // ---- load the official MuJoCo WebAssembly module ----
  const mujoco = await loadMujoco();
  sub.textContent = "加载模型 beijing.xml…";
  // revalidate the small model files every load so redeploys show immediately
  // (the 10 MB WASM is imported normally and stays cached)
  const xml = await (await fetch("./beijing.xml", { cache: "no-cache" })).text();
  mujoco.FS.writeFile("/model.xml", xml);
  const model = mujoco.MjModel.loadFromXML("/model.xml");
  const data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);
  const names = await (await fetch("./names.json", { cache: "no-cache" })).json();
  const NB = names.bodies || {};

  const ngeom = model.ngeom;
  const gtype = model.geom_type, gsize = model.geom_size, gmatid = model.geom_matid;
  const grgba = model.geom_rgba, gbody = model.geom_bodyid, matrgba = model.mat_rgba;
  const gxpos = data.geom_xpos, gxmat = data.geom_xmat;

  // ---- three.js scene (MuJoCo is z-up; rotate a root group so +z -> +y) ----
  const scene = new THREE.Scene();
  const sky = new THREE.Color(0xbcd6f0);
  scene.background = sky;
  scene.fog = new THREE.Fog(0xc3d8ef, 760, 2200);

  const camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 2, 6000);
  camera.position.set(255, 215, 345);

  const renderer = new THREE.WebGLRenderer({ canvas: $("#c"), antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, MOBILE ? 1.6 : 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  if (SHADOWS) {
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;   // static scene -> render shadows once
    renderer.shadowMap.needsUpdate = true;
  }

  // soft image-based lighting for nicer glass / metal
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 8, 8);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 25;
  controls.maxDistance = 2000;
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };

  scene.add(new THREE.HemisphereLight(0xdcebff, 0x6b7480, 0.7));
  scene.add(new THREE.AmbientLight(0xffffff, 0.14));   // low — GTAO supplies depth
  const sun = new THREE.DirectionalLight(0xffeccf, 2.6);
  sun.position.set(340, 360, 250);   // lower angle -> longer, readable shadows
  const fill = new THREE.DirectionalLight(0xbcd2ec, 0.5);
  fill.position.set(-260, 200, -200);
  scene.add(fill);
  if (SHADOWS) {
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const d = 460;
    const sc = sun.shadow.camera;
    sc.left = -d; sc.right = d; sc.top = d; sc.bottom = -d;
    sc.near = 80; sc.far = 1500;
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.6;
  }
  scene.add(sun);

  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;     // mujoco z-up -> three y-up
  scene.add(root);

  // ---- ground: a clean soft-vignette plane (no grid lines) ----
  const gtex = (() => {
    const cv = document.createElement("canvas"); cv.width = cv.height = 512;
    const x = cv.getContext("2d");
    const grad = x.createRadialGradient(256, 256, 30, 256, 256, 360);
    grad.addColorStop(0, "#cbc5b5"); grad.addColorStop(1, "#aaa493");
    x.fillStyle = grad; x.fillRect(0, 0, 512, 512);
    const t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  })();
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(2600, 2600),
    new THREE.MeshStandardMaterial({ map: gtex, roughness: 1, metalness: 0 })
  );
  ground.receiveShadow = SHADOWS;
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

  const cylGeo = new THREE.CylinderGeometry(1, 1, 1, SEG.c);
  cylGeo.rotateX(Math.PI / 2);  // axis -> z (matches mujoco cylinder local z)
  const bases = {
    box: new THREE.BoxGeometry(1, 1, 1),
    sph: new THREE.SphereGeometry(1, SEG.s[0], SEG.s[1]),
    cyl: cylGeo,
  };
  const mat = () => new THREE.MeshStandardMaterial({
    roughness: 0.68, metalness: 0.14, envMapIntensity: 0.9,
  });

  const m4 = new THREE.Matrix4(), sc = new THREE.Matrix4(), col = new THREE.Color();
  const meshes = {};
  for (const key of ["box", "sph", "cyl"]) {
    const list = buckets[key];
    const im = new THREE.InstancedMesh(bases[key], mat(), list.length);
    im.userData.geom = list;  // instanceId -> geom index
    im.castShadow = SHADOWS;
    im.receiveShadow = SHADOWS;
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
      col.convertSRGBToLinear();
      im.setColorAt(k, col);
    }
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    root.add(im);
    meshes[key] = im;
  }

  // ---- post-processing: ground-contact ambient occlusion (desktop) ----
  // GTAO darkens crevices, building bases and where volumes meet, which makes
  // the blocky primitives read with real depth/detail. SMAA gives clean edges.
  let composer = null;
  if (!MOBILE) {
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const gtao = new GTAOPass(scene, camera, innerWidth, innerHeight);
    gtao.output = GTAOPass.OUTPUT.Default;
    gtao.updateGtaoMaterial({
      radius: 4.5, distanceExponent: 1.0, thickness: 1.4,
      scale: 1.4, samples: 16, distanceFallOff: 1.0, screenSpaceRadius: false,
    });
    gtao.updatePdMaterial({
      lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 4, radiusExponent: 1,
      rings: 2, samples: 16,
    });
    composer.addPass(gtao);
    composer.addPass(new OutputPass());
    composer.addPass(new SMAAPass(innerWidth, innerHeight));
  }

  // ---- click / tap -> landmark name + intro (with highlight) ----
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  const info = $("#info");
  let downXY = null, downT = 0;
  let sel = null;  // { mesh, id, color:THREE.Color }

  function clearSel() {
    if (!sel) return;
    sel.mesh.setColorAt(sel.id, sel.color);
    sel.mesh.instanceColor.needsUpdate = true;
    sel = null;
  }
  function highlight(mesh, id) {
    clearSel();
    const c = new THREE.Color();
    mesh.getColorAt(id, c);
    sel = { mesh, id, color: c.clone() };
    mesh.setColorAt(id, c.lerp(new THREE.Color(0xffd66b), 0.55));
    mesh.instanceColor.needsUpdate = true;
  }
  function hideInfo() {
    info.classList.remove("show");
    document.body.classList.remove("sheet");
    clearSel();
  }

  function pick(e) {
    ptr.x = (e.clientX / innerWidth) * 2 - 1;
    ptr.y = -(e.clientY / innerHeight) * 2 + 1;
    ray.setFromCamera(ptr, camera);
    const hits = ray.intersectObjects([meshes.box, meshes.sph, meshes.cyl], false);
    for (const h of hits) {
      const gi = h.object.userData.geom[h.instanceId];
      const rec = NB[String(gbody[gi])];
      if (rec) {
        info.querySelector(".tag").textContent = rec.tag || "北京";
        info.querySelector(".zh").textContent = rec.zh;
        info.querySelector(".intro").textContent = rec.intro || "";
        info.classList.add("show");
        document.body.classList.add("sheet");
        highlight(h.object, h.instanceId);
        return;
      }
    }
    hideInfo();
  }

  renderer.domElement.addEventListener("pointerdown", (e) => {
    downXY = [e.clientX, e.clientY]; downT = performance.now();
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downXY) return;
    const moved = Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]);
    const tol = e.pointerType === "touch" ? 12 : 6;
    if (moved <= tol && performance.now() - downT < 600) pick(e);
    downXY = null;
  });
  $("#infoclose").addEventListener("click", hideInfo);

  // ---- run ----
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    if (composer) composer.setSize(innerWidth, innerHeight);
  });

  const recs = Object.values(NB);
  const nLand = recs.filter((r) => r.tag === "北京地标").length;
  const nNamed = recs.length;
  $("#stat").innerHTML =
    `${nLand} 地标 · ${nNamed} 可点建筑 · ${ngeom.toLocaleString()} geoms · MuJoCo WASM`;
  $("#help").innerHTML = MOBILE
    ? `<b>单指</b> 旋转 · <b>双指</b> 缩放/平移 · <b>点按</b> 看介绍`
    : `<b>拖拽</b> 旋转 &nbsp;·&nbsp; <b>滚轮</b> 缩放 &nbsp;·&nbsp; <b>右键</b> 平移 &nbsp;·&nbsp; <b>点击建筑</b> 看介绍`;

  const load = $("#load");
  load.style.opacity = 0; setTimeout(() => load.remove(), 700);

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    if (composer) composer.render();
    else renderer.render(scene, camera);
  })();
}

main().catch((e) => {
  console.error(e);
  $("#loadsub").textContent = "加载失败: " + (e.message || e);
  $("#loadsub").style.color = "#ff8a8a";
});
