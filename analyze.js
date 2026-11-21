// FoodScan analysis: tap-seeded GrabCut segmentation in every keyframe + voxel carving on the card frame.
// Needs window.FoodCore and OpenCV.js (grabCut only). Exposes window.FoodAnalyze.
(function (root) {
  'use strict';
  const Core = root.FoodCore;
  const OCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js';
  const WORK_MAX = 640; // segmentation resolution (long side)
  let cvP = null;

  function loadCV(onStatus) {
    if (cvP) return cvP;
    cvP = new Promise((res, rej) => {
      onStatus && onStatus('OpenCV betöltése (~13 MB, csak először)…');
      const s = document.createElement('script'); s.src = OCV_URL; s.async = true;
      s.onerror = () => { cvP = null; rej(new Error('Az OpenCV nem töltődött be.')); };
      s.onload = async () => {
        try {
          let cv = root.cv;
          if (cv && typeof cv.then === 'function') cv = await cv;
          if (!cv.Mat) await new Promise(r => { cv.onRuntimeInitialized = r; });
          root.cvReady = cv; res(cv);
        } catch (e) { cvP = null; rej(e); }
      };
      document.head.appendChild(s);
    });
    return cvP;
  }

  // ---------- helpers
  const scaleK = (K, k) => ({ f: K.f * k, cx: K.cx * k, cy: K.cy * k });
  function rasterPoly(w, h, polys) { // polys: [[[x,y],...],...] -> Uint8Array mask
    const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d');
    x.fillStyle = '#fff'; polys.forEach(p => { if (p.length < 3) return; x.beginPath(); p.forEach((q, i) => i ? x.lineTo(q[0], q[1]) : x.moveTo(q[0], q[1])); x.closePath(); x.fill(); });
    const d = x.getImageData(0, 0, w, h).data, m = new Uint8Array(w * h);
    for (let i = 0; i < m.length; i++) m[i] = d[4 * i + 3] > 127 ? 1 : 0;
    return m;
  }
  function convexHull(pts) { // monotone chain
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (p.length < 3) return p;
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }
  function erode(m, w, h, r) {
    let a = m;
    for (let it = 0; it < r; it++) {
      const b = new Uint8Array(w * h);
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) { const i = y * w + x; b[i] = a[i] && a[i - 1] && a[i + 1] && a[i - w] && a[i + w] ? 1 : 0; }
      a = b;
    }
    return a;
  }
  function largestComponent(m, w, h, prefer) {
    const lab = new Int32Array(w * h); let best = 0, bestN = 0, cur = 0, preferLab = 0; const st = [];
    for (let i = 0; i < m.length; i++) {
      if (!m[i] || lab[i]) continue;
      cur++; let n = 0; st.push(i); lab[i] = cur;
      while (st.length) { const j = st.pop(); n++; const x = j % w, y = (j / w) | 0;
        if (x > 0 && m[j - 1] && !lab[j - 1]) { lab[j - 1] = cur; st.push(j - 1); }
        if (x < w - 1 && m[j + 1] && !lab[j + 1]) { lab[j + 1] = cur; st.push(j + 1); }
        if (y > 0 && m[j - w] && !lab[j - w]) { lab[j - w] = cur; st.push(j - w); }
        if (y < h - 1 && m[j + w] && !lab[j + w]) { lab[j + w] = cur; st.push(j + w); } }
      if (n > bestN) { bestN = n; best = cur; }
    }
    if (prefer != null && lab[prefer]) preferLab = lab[prefer];
    const keep = preferLab || best, out = new Uint8Array(w * h);
    for (let i = 0; i < m.length; i++) out[i] = lab[i] === keep ? 1 : 0;
    return out;
  }
  function cardPoly(pose, K, scale, pad) {
    const hw = Core.CARD.w / 2 * scale + pad, hh = Core.CARD.h / 2 * scale + pad;
    return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => Core.project(pose.R, pose.t, K, [x, y, 0]).slice(0, 2));
  }
  function grabcut(cv, rgba, w, h, init) { // init: Uint8Array of GC labels
    const src = cv.matFromImageData(new ImageData(rgba, w, h)), img = new cv.Mat();
    cv.cvtColor(src, img, cv.COLOR_RGBA2RGB);
    const mask = new cv.Mat(h, w, cv.CV_8UC1); mask.data.set(init);
    const bg = new cv.Mat(), fg = new cv.Mat(), rect = new cv.Rect(0, 0, 1, 1);
    try { cv.grabCut(img, mask, rect, bg, fg, 5, cv.GC_INIT_WITH_MASK); }
    finally { src.delete(); img.delete(); bg.delete(); fg.delete(); }
    const out = new Uint8Array(w * h); const d = mask.data;
    for (let i = 0; i < out.length; i++) out[i] = d[i] === 1 || d[i] === 3 ? 1 : 0;
    mask.delete(); return out;
  }
  async function decode(kf) {
    const bm = await createImageBitmap(kf.blob);
    const s = Math.min(1, WORK_MAX / Math.max(bm.width, bm.height)), w = Math.round(bm.width * s), h = Math.round(bm.height * s);
    const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d', { willReadFrequently: true });
    x.drawImage(bm, 0, 0, w, h); bm.close && bm.close();
    return { w, h, rgba: x.getImageData(0, 0, w, h).data, k: kf.capScale * s }; // k: native px -> work px
  }

  // ---------- main
  // kfs: [{blob, capScale, pose}], K: native intrinsics, seed: {idx, x, y} in native px. Returns result object.
  async function run(kfs, K, cardScale, seed, onStatus) {
    const cv = await loadCV(onStatus);
    const imgs = [];
    for (let i = 0; i < kfs.length; i++) { onStatus && onStatus(`Képek dekódolása ${i + 1}/${kfs.length}`); imgs.push(await decode(kfs[i])); }
    const GC = { BGD: 0, FGD: 1, PR_BGD: 2, PR_FGD: 3 };
    // 1) seed frame
    const s0 = imgs[seed.idx], p0 = kfs[seed.idx].pose, K0 = scaleK(K, s0.k);
    const tx = seed.x * s0.k, ty = seed.y * s0.k;
    const R = p0.R, Cc = p0.C;
    const toPlane = (u, v, Kk) => { const xc = (u - Kk.cx) / Kk.f, yc = (v - Kk.cy) / Kk.f; const d = [R[0] * xc + R[3] * yc + R[6], R[1] * xc + R[4] * yc + R[7], R[2] * xc + R[5] * yc + R[8]]; const s = -Cc[2] / d[2]; return [Cc[0] + s * d[0], Cc[1] + s * d[1]]; };
    const P0 = toPlane(tx, ty, K0);
    const cyl = (c, r, hgt, n) => { const pts = []; for (let i = 0; i < n; i++) { const a = 2 * Math.PI * i / n; for (const z of [0, hgt]) pts.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), z]); } return pts; };
    const projHull = (pose, Kk, pts) => convexHull(pts.map(P => Core.project(pose.R, pose.t, Kk, P)).filter(q => q[2] > 0).map(q => [q[0], q[1]]));
    onStatus && onStatus('Szegmentálás: kiinduló kocka');
    let init = new Uint8Array(s0.w * s0.h).fill(GC.BGD);
    const reg0 = rasterPoly(s0.w, s0.h, [projHull(p0, K0, cyl(P0, 110 * cardScale, 120 * cardScale, 48))]);
    for (let i = 0; i < init.length; i++) if (reg0[i]) init[i] = GC.PR_BGD;
    const rr = Math.max(6, 0.06 * Math.max(s0.w, s0.h));
    for (let y = Math.max(0, ty - rr | 0); y < Math.min(s0.h, ty + rr); y++) for (let x = Math.max(0, tx - rr | 0); x < Math.min(s0.w, tx + rr); x++) { const d = Math.hypot(x - tx, y - ty); if (d < rr && reg0[y * s0.w + x]) init[y * s0.w + x] = d < 4 ? GC.FGD : GC.PR_FGD; }
    const card0 = rasterPoly(s0.w, s0.h, [cardPoly(p0, K0, cardScale, 2)]);
    for (let i = 0; i < init.length; i++) if (card0[i]) init[i] = GC.BGD;
    let m0 = grabcut(cv, s0.rgba, s0.w, s0.h, init);
    m0 = largestComponent(m0, s0.w, s0.h, (ty | 0) * s0.w + (tx | 0));
    // footprint from seed mask back-projected onto the table plane (upper bound, food top lands behind)
    const pl = []; for (let i = 0; i < m0.length; i += 3) if (m0[i]) pl.push(toPlane(i % s0.w, (i / s0.w) | 0, K0));
    if (pl.length < 20) throw new Error('Nem sikerült kivágni az ételt a koppintás körül — koppints az étel közepére.');
    const med = a => a.slice().sort((x, y) => x - y)[a.length >> 1];
    const ctr = [med(pl.map(p => p[0])), med(pl.map(p => p[1]))];
    const dists = pl.map(p => Math.hypot(p[0] - ctr[0], p[1] - ctr[1])).sort((a, b) => a - b);
    const rad = Math.min(160 * cardScale, dists[Math.floor(dists.length * 0.95)]);
    // 2) search volume and per-frame masks
    const R3 = 1.3 * rad, Hmax = 1.6 * rad;
    const masks = [];
    for (let i = 0; i < kfs.length; i++) {
      if (i === seed.idx) { masks.push(m0); continue; }
      onStatus && onStatus(`Szegmentálás ${i + 1}/${kfs.length}`);
      const im = imgs[i], p = kfs[i].pose, Kk = scaleK(K, im.k);
      const reg = rasterPoly(im.w, im.h, [projHull(p, Kk, cyl(ctr, R3, Hmax, 48))]);
      const core = erode(reg, im.w, im.h, Math.max(3, Math.round(0.012 * Math.max(im.w, im.h)) * 2));
      const card = rasterPoly(im.w, im.h, [cardPoly(p, Kk, cardScale, 2)]);
      const ini = new Uint8Array(im.w * im.h);
      for (let j = 0; j < ini.length; j++) ini[j] = card[j] ? GC.BGD : core[j] ? GC.PR_FGD : reg[j] ? GC.PR_BGD : GC.BGD;
      let m = grabcut(cv, im.rgba, im.w, im.h, ini);
      masks.push(largestComponent(m, im.w, im.h));
      await new Promise(r => setTimeout(r, 0));
    }
    // 3) voxel carving
    onStatus && onStatus('Térfogat: voxel-faragás');
    const vox = Math.max(1, R3 / 45);
    const V = [];
    for (let x = ctr[0] - R3; x <= ctr[0] + R3; x += vox) for (let y = ctr[1] - R3; y <= ctr[1] + R3; y += vox) {
      if (Math.hypot(x - ctr[0], y - ctr[1]) > R3) continue;
      for (let z = vox / 2; z < Hmax; z += vox) V.push(x, y, z);
    }
    const nV = V.length / 3, votes = new Uint16Array(nV), seen = new Uint16Array(nV);
    kfs.forEach((kf, i) => {
      const im = imgs[i], p = kf.pose, Kk = scaleK(K, im.k), m = masks[i], Rm = p.R, t = p.t;
      for (let v = 0; v < nV; v++) {
        const X = V[3 * v], Y = V[3 * v + 1], Z = V[3 * v + 2];
        const xc = Rm[0] * X + Rm[1] * Y + Rm[2] * Z + t[0], yc = Rm[3] * X + Rm[4] * Y + Rm[5] * Z + t[1], zc = Rm[6] * X + Rm[7] * Y + Rm[8] * Z + t[2];
        if (zc <= 0) continue;
        const u = Math.round(Kk.f * xc / zc + Kk.cx), w = Math.round(Kk.f * yc / zc + Kk.cy);
        if (u < 0 || w < 0 || u >= im.w || w >= im.h) continue;
        seen[v]++; if (m[w * im.w + u]) votes[v]++;
      }
    });
    const vv = vox ** 3 / 1000; let n0 = 0, n1 = 0; const pts = [];
    for (let v = 0; v < nV; v++) {
      if (seen[v] < 3) continue;
      if (votes[v] >= seen[v]) { n0++; pts.push(V[3 * v], V[3 * v + 1], V[3 * v + 2]); }
      if (votes[v] >= seen[v] - 1) n1++;
    }
    let zmax = 0; for (let i = 2; i < pts.length; i += 3) zmax = Math.max(zmax, pts[i]);
    return {
      volume: n0 * vv, volumeLoose: n1 * vv, voxelMm: vox, points: pts, footprint: { ctr, rad }, heightMm: zmax + vox / 2,
      masks: masks.map((m, i) => ({ w: imgs[i].w, h: imgs[i].h, k: imgs[i].k, data: m })), seedIdx: seed.idx,
      seedCrop: cropAround(imgs[seed.idx], m0)
    };
  }
  // crop of the seed frame around the segmented food (for recognition), as a JPEG Blob promise
  function cropAround(im, m) {
    let x0 = im.w, y0 = im.h, x1 = 0, y1 = 0;
    for (let i = 0; i < m.length; i++) if (m[i]) { const x = i % im.w, y = (i / im.w) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const pad = 0.35 * Math.max(x1 - x0, y1 - y0, 20); x0 = Math.max(0, x0 - pad | 0); y0 = Math.max(0, y0 - pad | 0); x1 = Math.min(im.w, x1 + pad | 0); y1 = Math.min(im.h, y1 + pad | 0);
    const c = document.createElement('canvas'); c.width = im.w; c.height = im.h; c.getContext('2d').putImageData(new ImageData(im.rgba, im.w, im.h), 0, 0);
    const o = document.createElement('canvas'); o.width = x1 - x0; o.height = y1 - y0; o.getContext('2d').drawImage(c, x0, y0, o.width, o.height, 0, 0, o.width, o.height);
    return new Promise(r => o.toBlob(r, 'image/jpeg', 0.9));
  }


  // ---------- automatic mode: no tap. Off-plane objects are found by multi-view consistency.
  function toLab(r, g, b) { // sRGB 0..255 -> CIE Lab (D65)
    const f = c => { c /= 255; return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92; };
    const R = f(r), G = f(g), B = f(b);
    let X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
    const g3 = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
    X = g3(X); Y = g3(Y); Z = g3(Z);
    return [116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)];
  }
  function kmeans(pts, k, it) {
    let c = []; for (let i = 0; i < k; i++) c.push(pts[Math.floor((i + 0.5) * pts.length / k)].slice());
    let lab = new Int32Array(pts.length);
    for (let t = 0; t < it; t++) {
      const sum = c.map(() => [0, 0, 0, 0]);
      pts.forEach((p, i) => { let b = 0, bd = 1e9; c.forEach((q, j) => { const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2; if (d < bd) { bd = d; b = j; } }); lab[i] = b; const s2 = sum[b]; s2[0] += p[0]; s2[1] += p[1]; s2[2] += p[2]; s2[3]++; });
      c = c.map((q, j) => sum[j][3] ? [sum[j][0] / sum[j][3], sum[j][1] / sum[j][3], sum[j][2] / sum[j][3]] : q);
    }
    const share = c.map((q, j) => 0); lab.forEach(l => share[l]++);
    return c.map((q, j) => ({ c: q, share: share[j] / pts.length }));
  }
  // per frame: background (table/plate) colour modes from a ring on the plane around the card, then a candidate mask
  function candidateMask(im, pose, Kk, cardScale) {
    const pts = [], hw = Core.CARD.w / 2 * cardScale + 6, hh = Core.CARD.h / 2 * cardScale + 6;
    for (let r = 60; r <= 240; r += 12) for (let a = 0; a < 360; a += 6) {
      const x = r * cardScale * Math.cos(a * Math.PI / 180), y = r * cardScale * Math.sin(a * Math.PI / 180);
      if (Math.abs(x) < hw && Math.abs(y) < hh) continue;
      const q = Core.project(pose.R, pose.t, Kk, [x, y, 0]); const u = q[0] | 0, v = q[1] | 0;
      if (q[2] <= 0 || u < 0 || v < 0 || u >= im.w || v >= im.h) continue;
      const i = 4 * (v * im.w + u); pts.push(toLab(im.rgba[i], im.rgba[i + 1], im.rgba[i + 2]));
    }
    if (pts.length < 30) return null;
    const modes = kmeans(pts, 4, 8).filter(m => m.share >= 0.18).map(m => m.c);
    const m = new Uint8Array(im.w * im.h), lab = new Float32Array(im.w * im.h * 3);
    for (let p = 0, i = 0; p < m.length; p++, i += 4) {
      const L = toLab(im.rgba[i], im.rgba[i + 1], im.rgba[i + 2]); lab[3 * p] = L[0]; lab[3 * p + 1] = L[1]; lab[3 * p + 2] = L[2];
      let best = 1e9, dL = 0;
      for (const c of modes) { const dc = Math.hypot(L[1] - c[1], L[2] - c[2]), dl = L[0] - c[0], sc = dc + Math.max(0, Math.abs(dl) - 30) * 0.5; if (sc < best) { best = sc; dL = dl; } }
      const dc = best; // chroma-led distance; mild darkening (shadow) is tolerated
      m[p] = dc > 12 || dL > 22 || dL < -50 ? 1 : 0;
    }
    return { m, lab };
  }
  function carveGrid(views, box, step, frac) { // views: [{pose, Kk, w, h, m}]
    const nx = Math.ceil((box.x1 - box.x0) / step), ny = Math.ceil((box.y1 - box.y0) / step), nz = Math.ceil((box.z1 - box.z0) / step);
    const occ = new Uint8Array(nx * ny * nz), seen = new Uint8Array(nx * ny * nz), vote = new Uint8Array(nx * ny * nz);
    for (const vw of views) {
      const R = vw.pose.R, t = vw.pose.t, K = vw.Kk, m = vw.m, w = vw.w, h = vw.h;
      let idx = 0;
      for (let ix = 0; ix < nx; ix++) { const X = box.x0 + (ix + 0.5) * step;
        for (let iy = 0; iy < ny; iy++) { const Y = box.y0 + (iy + 0.5) * step;
          const bx = R[0] * X + R[1] * Y + t[0], by = R[3] * X + R[4] * Y + t[1], bz = R[6] * X + R[7] * Y + t[2];
          for (let iz = 0; iz < nz; iz++, idx++) { const Z = box.z0 + (iz + 0.5) * step;
            const zc = bz + R[8] * Z; if (zc <= 0) continue;
            const u = (K.f * (bx + R[2] * Z) / zc + K.cx) | 0, v = (K.f * (by + R[5] * Z) / zc + K.cy) | 0;
            if (u < 0 || v < 0 || u >= w || v >= h) continue;
            seen[idx]++; if (m[v * w + u]) vote[idx]++; } } }
    }
    const minSeen = Math.max(3, Math.ceil(views.length * 0.5));
    for (let i = 0; i < occ.length; i++) occ[i] = seen[i] >= minSeen && vote[i] >= frac * seen[i] ? 1 : 0;
    return { occ, nx, ny, nz, step, box };
  }
  function components3d(g) { // 6-connected components -> [{n, vox:[indices], zmax, bbox}]
    const { occ, nx, ny, nz } = g, lab = new Int32Array(occ.length), comps = []; const st = [];
    const id = (x, y, z) => (x * ny + y) * nz + z;
    for (let i = 0; i < occ.length; i++) {
      if (!occ[i] || lab[i]) continue;
      const c = { vox: [], zmax: 0, x0: 1e9, x1: -1, y0: 1e9, y1: -1 }; lab[i] = comps.length + 1; st.push(i);
      while (st.length) { const j = st.pop(); c.vox.push(j);
        const z = j % nz, y = ((j / nz) | 0) % ny, x = (j / (nz * ny)) | 0;
        if (z > c.zmax) c.zmax = z; if (x < c.x0) c.x0 = x; if (x > c.x1) c.x1 = x; if (y < c.y0) c.y0 = y; if (y > c.y1) c.y1 = y;
        const nb = [[x - 1, y, z], [x + 1, y, z], [x, y - 1, z], [x, y + 1, z], [x, y, z - 1], [x, y, z + 1]];
        for (const [a, b, cc] of nb) { if (a < 0 || b < 0 || cc < 0 || a >= nx || b >= ny || cc >= nz) continue; const k = id(a, b, cc); if (occ[k] && !lab[k]) { lab[k] = lab[i]; st.push(k); } } }
      comps.push(c);
    }
    return comps;
  }
  const centreOf = (g, c) => { let sx = 0, sy = 0; for (const j of c.vox) { sx += ((j / (g.nz * g.ny)) | 0); sy += ((j / g.nz) | 0) % g.ny; } return [g.box.x0 + (sx / c.vox.length + 0.5) * g.step, g.box.y0 + (sy / c.vox.length + 0.5) * g.step]; };

  // tap (optional): {idx, x, y} native px — picks the candidate object nearest to that viewing ray
  async function runAuto(kfs, K, cardScale, onStatus, tap) {
    const cv = await loadCV(onStatus);
    const imgs = [];
    for (let i = 0; i < kfs.length; i++) { onStatus && onStatus(`Képek dekódolása ${i + 1}/${kfs.length}`); imgs.push(await decode(kfs[i])); }
    // 1) candidate (non-background) masks
    const views = [];
    for (let i = 0; i < kfs.length; i++) {
      onStatus && onStatus(`Háttér-modell ${i + 1}/${kfs.length}`);
      const im = imgs[i], Kk = scaleK(K, im.k), cm = candidateMask(im, kfs[i].pose, Kk, cardScale);
      const cm0 = cm ? cm.m : new Uint8Array(im.w * im.h), cardM = rasterPoly(im.w, im.h, [cardPoly(kfs[i].pose, Kk, cardScale, 3)]);
      for (let j = 0; j < cm0.length; j++) if (cardM[j]) cm0[j] = 0; // the card is flat printed paper, never the food
      views.push({ i, pose: kfs[i].pose, Kk, w: im.w, h: im.h, m: cm0 });
      await new Promise(r => setTimeout(r, 0));
    }
    // 2) coarse carve in a cylinder around the card
    onStatus && onStatus('Tárgyak keresése 3D-ben…');
    const RR = 280 * cardScale, coarse = carveGrid(views, { x0: -RR, x1: RR, y0: -RR, y1: RR, z0: 0, z1: 220 * cardScale }, 5 * cardScale, 0.8);
    let comps = components3d(coarse).map(c => ({ ...c, ctr: centreOf(coarse, c), vol: c.vox.length * coarse.step ** 3 / 1000, hgt: (c.zmax + 1) * coarse.step }));
    const onCard = c => Math.abs(c.ctr[0]) < Core.CARD.w / 2 * cardScale + 10 && Math.abs(c.ctr[1]) < Core.CARD.h / 2 * cardScale + 10;
    comps = comps.filter(c => c.hgt >= 18 * cardScale && c.vol >= 3 && Math.hypot(...c.ctr) <= RR && !onCard(c));
    if (!comps.length) throw new Error('Nem találtam a kártya körül az asztalból kiemelkedő tárgyat. Legyen az étel 25 cm-en belül a kártyától, és több szögből is látsszon.');
    let pick;
    if (tap) {
      const p = kfs[tap.idx].pose, R = p.R, C0 = p.C, xc = (tap.x - K.cx) / K.f, yc = (tap.y - K.cy) / K.f;
      const d = [R[0] * xc + R[3] * yc + R[6], R[1] * xc + R[4] * yc + R[7], R[2] * xc + R[5] * yc + R[8]], dn = Math.hypot(...d);
      const distRay = c => { const P = [c.ctr[0] - C0[0], c.ctr[1] - C0[1], c.hgt / 2 - C0[2]], tt = (P[0] * d[0] + P[1] * d[1] + P[2] * d[2]) / dn; return Math.hypot(P[0] - tt * d[0] / dn, P[1] - tt * d[1] / dn, P[2] - tt * d[2] / dn); };
      pick = comps.reduce((a, b) => distRay(a) <= distRay(b) ? a : b);
    } else pick = comps.reduce((a, b) => Math.hypot(...a.ctr) <= Math.hypot(...b.ctr) ? a : b);
    // 3) refine masks with GrabCut, seeded by the coarse object's projection
    const pad = 3 * coarse.step;
    const bb = { x0: coarse.box.x0 + pick.x0 * coarse.step - pad, x1: coarse.box.x0 + (pick.x1 + 1) * coarse.step + pad, y0: coarse.box.y0 + pick.y0 * coarse.step - pad, y1: coarse.box.y0 + (pick.y1 + 1) * coarse.step + pad, z0: 0, z1: pick.hgt + pad };
    const pts3 = pick.vox.map(j => [coarse.box.x0 + (((j / (coarse.nz * coarse.ny)) | 0) + 0.5) * coarse.step, coarse.box.y0 + ((((j / coarse.nz) | 0) % coarse.ny) + 0.5) * coarse.step, ((j % coarse.nz) + 0.5) * coarse.step]);
    const GC = { BGD: 0, FGD: 1, PR_BGD: 2, PR_FGD: 3 };
    const masks = [];
    for (const vw of views) {
      onStatus && onStatus(`Szegmentálás ${vw.i + 1}/${views.length}`);
      const im = imgs[vw.i], proj = new Uint8Array(im.w * im.h);
      const rad = Math.max(1, Math.round(vw.Kk.f * coarse.step / Math.max(200, vw.pose.dist) * 0.6));
      for (const P of pts3) { const q = Core.project(vw.pose.R, vw.pose.t, vw.Kk, P); const u = q[0] | 0, v = q[1] | 0; if (q[2] <= 0) continue;
        for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) { const x = u + dx, y = v + dy; if (x >= 0 && y >= 0 && x < im.w && y < im.h) proj[y * im.w + x] = 1; } }
      const card = rasterPoly(im.w, im.h, [cardPoly(vw.pose, vw.Kk, cardScale, 2)]);
      const grow = Math.round(0.03 * Math.max(im.w, im.h));
      const near = dilate(proj, im.w, im.h, grow), core = erode(proj, im.w, im.h, Math.max(2, grow >> 1));
      const ini = new Uint8Array(im.w * im.h);
      for (let j = 0; j < ini.length; j++) ini[j] = card[j] ? GC.BGD : core[j] && vw.m[j] ? GC.FGD : proj[j] && vw.m[j] ? GC.PR_FGD : near[j] ? GC.PR_BGD : GC.BGD;
      let m;
      let nF = 0, nB = 0; for (let j = 0; j < ini.length; j++) { if (ini[j] === 1 || ini[j] === 3) nF++; else nB++; }
      if (nF < 30 || nB < 30) m = new Uint8Array(im.w * im.h);
      else { try { m = grabcut(cv, im.rgba, im.w, im.h, ini); } catch (e) { m = new Uint8Array(im.w * im.h); for (let j = 0; j < m.length; j++) m[j] = proj[j] && vw.m[j] ? 1 : 0; console.warn('grabCut failed on frame', vw.i, e); } }
      // colour-distinct food: drop table-coloured pixels (cast shadows) that GrabCut may have kept
      let inP = 0, inPC = 0; for (let j = 0; j < proj.length; j++) if (proj[j]) { inP++; if (vw.m[j]) inPC++; }
      if (inP && inPC / inP > 0.4) { const cd = dilate(vw.m, im.w, im.h, 2); for (let j = 0; j < m.length; j++) if (!cd[j]) m[j] = 0; }
      masks.push(m); vw.m = m;
      await new Promise(r => setTimeout(r, 0));
    }
    // 4) fine carve in the object's box
    onStatus && onStatus('Térfogat: finom faragás');
    const span = Math.max(bb.x1 - bb.x0, bb.y1 - bb.y0, bb.z1);
    const step = Math.max(1, span / 110);
    const fine = carveGrid(views, bb, step, 0.95), fineL = carveGrid(views, bb, step, 0.85);
    const keepBig = g => { const cs = components3d(g); if (!cs.length) return []; return cs.reduce((a, b) => a.vox.length >= b.vox.length ? a : b).vox; };
    // drop the flat skirt (shadow / table texture consistent with the plane): columns lower than 6 mm
    const dropFlat = g => { const top = new Float32Array(g.nx * g.ny); for (let i = 0; i < g.occ.length; i++) if (g.occ[i]) { const c = (i / g.nz) | 0; top[c] = Math.max(top[c], (i % g.nz) + 1); }
      const lim = Math.max(1, Math.round(6 / g.step)); for (let i = 0; i < g.occ.length; i++) if (g.occ[i] && top[(i / g.nz) | 0] <= lim) g.occ[i] = 0; };
    dropFlat(fine); dropFlat(fineL);
    const vox = keepBig(fine), voxL = keepBig(fineL);
    const pts = []; let zmax = 0, sx = 0, sy = 0, rmax = 0;
    for (const j of vox) { const x = bb.x0 + (((j / (fine.nz * fine.ny)) | 0) + 0.5) * step, y = bb.y0 + ((((j / fine.nz) | 0) % fine.ny) + 0.5) * step, z = ((j % fine.nz) + 0.5) * step; pts.push(x, y, z); zmax = Math.max(zmax, z); sx += x; sy += y; }
    const ctr = vox.length ? [sx / vox.length, sy / vox.length] : pick.ctr;
    for (let i = 0; i < pts.length; i += 3) rmax = Math.max(rmax, Math.hypot(pts[i] - ctr[0], pts[i + 1] - ctr[1]));
    // best frame for recognition: largest projected object, moderately high view
    let best = 0, bs = -1; masks.forEach((m, i) => { let n = 0; for (let j = 0; j < m.length; j++) n += m[j]; const sc = n / m.length * (1 - Math.abs(kfs[i].pose.elev - 50) / 90); if (sc > bs) { bs = sc; best = i; } });
    return {
      volume: vox.length * step ** 3 / 1000, volumeLoose: voxL.length * step ** 3 / 1000, voxelMm: step, points: pts, footprint: { ctr, rad: rmax + step / 2 }, heightMm: zmax + step / 2,
      masks: masks.map((m, i) => ({ w: imgs[i].w, h: imgs[i].h, k: imgs[i].k, data: m })), seedIdx: best, seedCrop: cropAround(imgs[best], masks[best]),
      candidates: comps.length
    };
  }
  function dilate(m, w, h, r) { const inv = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) inv[i] = m[i] ? 0 : 1; const e = erode(inv, w, h, r); const o = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) o[i] = e[i] ? 0 : 1; return o; }


  // ---------- dish mode: multi-view plane sweep -> height map -> container (plate/bowl) -> volume
  function grayOf(im) { const g = new Float32Array(im.w * im.h); for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * im.rgba[j] + 0.587 * im.rgba[j + 1] + 0.114 * im.rgba[j + 2]; return g; }
  function bilin(g, w, h, u, v) { if (u < 0 || v < 0 || u >= w - 1 || v >= h - 1) return -1; const x = u | 0, y = v | 0, fx = u - x, fy = v - y, i = y * w + x; return (g[i] * (1 - fx) + g[i + 1] * fx) * (1 - fy) + (g[i + w] * (1 - fx) + g[i + w + 1] * fx) * fy; }
  async function heightMap(views, opts, onStatus) {
    const ext = opts.ext, cell = opts.cell, N = Math.round(2 * ext / cell), zs = []; for (let z = 0; z <= opts.zmax; z += opts.dz) zs.push(z);
    const off = []; for (const a of [-0.4, 0, 0.4]) for (const b of [-0.4, 0, 0.4]) off.push([a * cell, b * cell]);
    const best = new Float32Array(N * N).fill(-2), H = new Float32Array(N * N), S = new Float32Array(9), v9 = new Float32Array(9);
    for (let zi = 0; zi < zs.length; zi++) {
      const z = zs[zi];
      if (zi % 4 === 0) { onStatus && onStatus(`Felszínmérés (plane sweep) ${Math.round(100 * zi / zs.length)}%`); await new Promise(r => setTimeout(r, 0)); }
      for (let ix = 0; ix < N; ix++) { const X0 = -ext + (ix + 0.5) * cell;
        for (let iy = 0; iy < N; iy++) { const Y0 = -ext + (iy + 0.5) * cell;
          S.fill(0); let n = 0;
          for (const vw of views) {
            const R = vw.R, t = vw.t, K = vw.K; let ok = true, m = 0;
            for (let k = 0; k < 9; k++) { const X = X0 + off[k][0], Y = Y0 + off[k][1];
              const zc = R[6] * X + R[7] * Y + R[8] * z + t[2]; if (zc <= 0) { ok = false; break; }
              const val = bilin(vw.g, vw.w, vw.h, K.f * (R[0] * X + R[1] * Y + R[2] * z + t[0]) / zc + K.cx, K.f * (R[3] * X + R[4] * Y + R[5] * z + t[1]) / zc + K.cy);
              if (val < 0) { ok = false; break; } v9[k] = val; m += val; }
            if (!ok) continue;
            m /= 9; let sd = 0; for (let k = 0; k < 9; k++) sd += (v9[k] - m) ** 2; sd = Math.sqrt(sd / 9) + 1;
            for (let k = 0; k < 9; k++) S[k] += (v9[k] - m) / sd; n++;
          }
          if (n < 3) continue;
          let ss = 0; for (let k = 0; k < 9; k++) ss += S[k] * S[k];
          const sc = (ss / 9 - n) / (n * (n - 1)), c = ix * N + iy;
          if (sc > best[c]) { best[c] = sc; H[c] = z; }
        } }
    }
    return { H, C: best, N, cell, ext };
  }
  function labelGrid(m, N) { const lab = new Int32Array(N * N); const comps = []; const st = [];
    for (let i = 0; i < m.length; i++) { if (!m[i] || lab[i]) continue; const c = { cells: [] }; lab[i] = comps.length + 1; st.push(i);
      while (st.length) { const j = st.pop(); c.cells.push(j); const x = (j / N) | 0, y = j % N;
        for (const [a, b] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) { if (a < 0 || b < 0 || a >= N || b >= N) continue; const k = a * N + b; if (m[k] && !lab[k]) { lab[k] = lab[i]; st.push(k); } } }
      comps.push(c); }
    return { lab, comps }; }
  function morph(m, N, r, dil) { let a = m; for (let it = 0; it < r; it++) { const b = new Uint8Array(N * N);
      for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) { const i = x * N + y; let v = a[i];
        const nb = [x > 0 ? a[i - N] : 0, x < N - 1 ? a[i + N] : 0, y > 0 ? a[i - 1] : 0, y < N - 1 ? a[i + 1] : 0];
        b[i] = dil ? (v || nb.some(q => q) ? 1 : 0) : (v && nb.every(q => q) ? 1 : 0); }
      a = b; } return a; }
  const PROFILES = { flat: { hu: 'lapos tányér', z0: 4, p: 8, pr: [5, 12] }, soup: { hu: 'mélytányér', z0: 5, p: 3, pr: [2, 5] }, bowl: { hu: 'tál', z0: 8, p: 2, pr: [1.5, 3.5] } };
  // volume of the food above a rotationally symmetric container floor z_b(r) = z0 + (zw - z0) (r/Rw)^p
  function dishVolume(hm, ctr, Rw, zw, z0, p, cells) {
    let v = 0; const a = hm.cell * hm.cell;
    for (let c = 0; c < hm.N * hm.N; c++) { const x = -hm.ext + (((c / hm.N) | 0) + 0.5) * hm.cell, y = -hm.ext + ((c % hm.N) + 0.5) * hm.cell, r = Math.hypot(x - ctr[0], y - ctr[1]);
      if (r > Rw) continue; const zb = z0 + (zw - z0) * Math.pow(r / Rw, p); v += Math.max(0, hm.Hf[c] - zb) * a; }
    return v / 1000;
  }
  async function runDish(kfs, K, cardScale, onStatus, tap) {
    const imgs = [];
    for (let i = 0; i < kfs.length; i++) { onStatus && onStatus(`Képek dekódolása ${i + 1}/${kfs.length}`); imgs.push(await decode(kfs[i])); }
    // up to 14 views spread over azimuth
    const order = kfs.map((k, i) => i).sort((a, b) => kfs[a].pose.azim - kfs[b].pose.azim), nv = Math.min(14, order.length);
    const pick = []; for (let j = 0; j < nv; j++) pick.push(order[Math.round(j * (order.length - 1) / Math.max(1, nv - 1))]);
    const views = [...new Set(pick)].map(i => ({ i, R: kfs[i].pose.R, t: kfs[i].pose.t, K: scaleK(K, imgs[i].k), g: grayOf(imgs[i]), w: imgs[i].w, h: imgs[i].h }));
    const hm = await heightMap(views, { ext: 300 * cardScale, cell: 5 * cardScale, zmax: 140 * cardScale, dz: 2 * cardScale }, onStatus);
    const N = hm.N, valid = new Uint8Array(N * N); for (let i = 0; i < valid.length; i++) valid[i] = hm.C[i] > 0.3 ? 1 : 0;
    // fill unreliable cells with the median of reliable neighbours (a few passes)
    const Hf = Float32Array.from(hm.H); let known = Uint8Array.from(valid);
    for (let pass = 0; pass < 6; pass++) { const nk = Uint8Array.from(known);
      for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) { const i = x * N + y; if (known[i]) continue; const nb = [];
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) { const a = x + dx, b = y + dy; if (a >= 0 && b >= 0 && a < N && b < N && known[a * N + b]) nb.push(Hf[a * N + b]); }
        if (nb.length >= 3) { nb.sort((p, q) => p - q); Hf[i] = nb[nb.length >> 1]; nk[i] = 1; } }
      known = nk; }
    hm.Hf = Hf; hm.valid = valid;
    const cellXY = c => [-hm.ext + (((c / N) | 0) + 0.5) * hm.cell, -hm.ext + ((c % N) + 0.5) * hm.cell];
    const onCard = c => { const [x, y] = cellXY(c); return Math.abs(x) < Core.CARD.w / 2 * cardScale + 8 && Math.abs(y) < Core.CARD.h / 2 * cardScale + 8; };
    let m = new Uint8Array(N * N); for (let i = 0; i < m.length; i++) m[i] = valid[i] && hm.H[i] > 10 * cardScale && !onCard(i) ? 1 : 0;
    m = morph(morph(m, N, 1, false), N, 1, true); m = morph(morph(m, N, 2, true), N, 2, false);
    // fill holes: cells not reachable from the border through empty cells
    { const out = new Uint8Array(N * N), st = []; for (let k = 0; k < N; k++) for (const c of [k, (N - 1) * N + k, k * N, k * N + N - 1]) if (!m[c] && !out[c]) { out[c] = 1; st.push(c); }
      while (st.length) { const j = st.pop(), x = (j / N) | 0, y = j % N; for (const [a2, b2] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) { if (a2 < 0 || b2 < 0 || a2 >= N || b2 >= N) continue; const k2 = a2 * N + b2; if (!m[k2] && !out[k2]) { out[k2] = 1; st.push(k2); } } }
      for (let i = 0; i < m.length; i++) if (!out[i] && !onCard(i)) m[i] = 1; }
    const { comps } = labelGrid(m, N);
    let cand = comps.filter(c => c.cells.length * hm.cell * hm.cell >= 2000).map(c => { let sx = 0, sy = 0; for (const j of c.cells) { const [x, y] = cellXY(j); sx += x; sy += y; } return { ...c, ctr: [sx / c.cells.length, sy / c.cells.length] }; });
    if (!cand.length) return null;
    let dish;
    if (tap) { const p = kfs[tap.idx].pose, R = p.R, C0 = p.C, xc = (tap.x - K.cx) / K.f, yc = (tap.y - K.cy) / K.f, d = [R[0] * xc + R[3] * yc + R[6], R[1] * xc + R[4] * yc + R[7], R[2] * xc + R[5] * yc + R[8]];
      const s = -(C0[2] - 20) / d[2], P = [C0[0] + s * d[0], C0[1] + s * d[1]]; dish = cand.reduce((a, b) => Math.hypot(a.ctr[0] - P[0], a.ctr[1] - P[1]) <= Math.hypot(b.ctr[0] - P[0], b.ctr[1] - P[1]) ? a : b); }
    else dish = cand.reduce((a, b) => Math.hypot(...a.ctr) <= Math.hypot(...b.ctr) ? a : b);
    // circle fit (Kasa) on the region boundary
    const inR = new Uint8Array(N * N); dish.cells.forEach(c => inR[c] = 1);
    const bnd = dish.cells.filter(c => { const x = (c / N) | 0, y = c % N; return !(x > 0 && inR[c - N] && x < N - 1 && inR[c + N] && y > 0 && inR[c - 1] && y < N - 1 && inR[c + 1]); });
    let A = [0, 0, 0, 0, 0, 0, 0, 0, 0], bb = [0, 0, 0];
    for (const c of bnd) { const [x, y] = cellXY(c), row = [2 * x, 2 * y, 1], rhs = x * x + y * y; for (let i = 0; i < 3; i++) { bb[i] += row[i] * rhs; for (let j = 0; j < 3; j++) A[i * 3 + j] += row[i] * row[j]; } }
    const Ai = Core.inv3(A); if (!Ai) return null;
    const sol = [0, 1, 2].map(i => Ai[i * 3] * bb[0] + Ai[i * 3 + 1] * bb[1] + Ai[i * 3 + 2] * bb[2]);
    const ctr = [sol[0], sol[1]]; let Rw = Math.sqrt(sol[2] + ctr[0] ** 2 + ctr[1] ** 2);
    // radial profile (medians of reliable cells in 5 mm rings)
    const ringsN = Math.ceil((1.7 * Rw + 30) / 5), rings = Array.from({ length: ringsN }, () => []);
    for (let c = 0; c < N * N; c++) { if (!valid[c]) continue; const [x, y] = cellXY(c), r = Math.hypot(x - ctr[0], y - ctr[1]); const k = (r / 5) | 0; if (k < ringsN) rings[k].push(hm.H[c]); }
    const prof = rings.map(a => a.length >= 3 ? a.slice().sort((p, q) => p - q)[a.length >> 1] : null);
    const inner = prof.slice(0, Math.max(1, Math.floor(0.7 * Rw / 5))).filter(v => v != null), outer = prof.slice(Math.floor(0.7 * Rw / 5), Math.ceil((Rw + 12) / 5)).filter(v => v != null);
    const med = a => a.length ? a.slice().sort((p, q) => p - q)[a.length >> 1] : 0;
    const innerMed = med(inner), innerMin = inner.length ? Math.min(...inner) : 0, rimH = outer.length ? Math.max(...outer) : 0;
    const iq = (() => { const s2 = inner.slice().sort((p, q) => p - q); return s2.length ? s2[Math.floor(s2.length * 0.8)] - s2[Math.floor(s2.length * 0.2)] : 99; })();
    const isContainer = rimH >= innerMin + 4 * cardScale && (rimH >= innerMed - 2 || iq < 6);
    // food edge: walk out from the centre while the ring stays at the food level (it ends where the wall rises or the table starts)
    if (isContainer) { const lvl = innerMed, tol = Math.max(5, 0.25 * lvl); let k = 0;
      for (k = 0; k < prof.length; k++) { if (prof[k] == null) continue; if (Math.abs(prof[k] - lvl) > tol && k * 5 > 0.5 * Rw) break; }
      Rw = Math.max(Rw, k * 5); }
    const result = { hm, ctr, Rw, prof, innerMed, rimH, isContainer, views: views.length, dishCells: dish.cells, candidates: cand.length };
    if (!isContainer) return result;
    // container type from rim height above the table; floor model
    const type = rimH < 25 * cardScale ? 'flat' : rimH < 45 * cardScale ? 'soup' : 'bowl', P0 = PROFILES[type];
    // floor height where the food meets the wall = food surface in the outer ring of the region
    const edge = []; for (let c = 0; c < N * N; c++) { const [x, y] = cellXY(c), r = Math.hypot(x - ctr[0], y - ctr[1]); if (r > 0.8 * Rw && r <= Rw && valid[c]) edge.push(hm.H[c]); }
    const zw = edge.length ? med(edge) : innerMed, z0 = Math.min(P0.z0 * cardScale, innerMin, zw);
    // the food meets the wall where the region ends: use the fitted radius
    const V = dishVolume(hm, ctr, Rw, zw, z0, P0.p, dish.cells), Vlo = dishVolume(hm, ctr, Rw, zw, z0, P0.pr[0], dish.cells), Vhi = dishVolume(hm, ctr, Rw, zw, z0, P0.pr[1], dish.cells);
    // surface points for the 3D view
    const pts = []; for (let c = 0; c < N * N; c++) { const [x, y] = cellXY(c); if (Math.hypot(x - ctr[0], y - ctr[1]) <= Rw) pts.push(x, y, Hf[c]); }
    // display masks: the rim circle at the food level projected into each keyframe
    const masks = kfs.map((kf, i) => { const im = imgs[i], Kk = scaleK(K, im.k), poly = []; for (let a = 0; a < 64; a++) { const q = Core.project(kf.pose.R, kf.pose.t, Kk, [ctr[0] + Rw * Math.cos(a * Math.PI / 32), ctr[1] + Rw * Math.sin(a * Math.PI / 32), zw]); poly.push([q[0], q[1]]); } return { w: im.w, h: im.h, k: im.k, data: rasterPoly(im.w, im.h, [poly]) }; });
    let bestI = 0; kfs.forEach((k, i) => { if (k.pose.elev > kfs[bestI].pose.elev) bestI = i; });
    return Object.assign(result, { type, typeHu: P0.hu, z0, zw, p: P0.p, volume: V, volumeLo: Math.min(Vlo, Vhi), volumeHi: Math.max(Vlo, Vhi), points: pts, masks, seedIdx: bestI, seedCrop: cropAround(imgs[bestI], masks[bestI].data), footprint: { ctr, rad: Rw }, heightMm: Math.max(innerMed, rimH), voxelMm: hm.cell });
  }
  root.FoodAnalyze = { run, runAuto, runDish, loadCV };
})(window);
