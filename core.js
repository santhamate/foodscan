// FoodScan core geometry: card layout, homography, focal estimate, pose, coverage.
// Pure JS, no deps. Works in browser (window.FoodCore) and Node (module.exports).
(function (root) {
  'use strict';

  // ---------- Card layout (mm). Origin = card centre, x right, y up, z up (out of table).
  const CARD = {
    w: 85.6, h: 54.0,           // ISO/IEC 7810 ID-1
    marker: 20.0,               // black square side incl. 1-module border (8 modules -> 2.5 mm)
    cols: 3, rows: 2, gapX: 7.0, gapY: 6.0,
    dict: 'ARUCO_MIP_36h12', maxHamming: 5,
    ids: [0, 1, 2, 3, 4, 5],
  };
  // top-left (x,y in SVG-like coords, y down, mm from card top-left) of each marker
  function markerTopLeftSvg(i) {
    const totW = CARD.cols * CARD.marker + (CARD.cols - 1) * CARD.gapX;
    const totH = CARD.rows * CARD.marker + (CARD.rows - 1) * CARD.gapY;
    const x0 = (CARD.w - totW) / 2, y0 = (CARD.h - totH) / 2;
    const c = i % CARD.cols, r = Math.floor(i / CARD.cols);
    return [x0 + c * (CARD.marker + CARD.gapX), y0 + r * (CARD.marker + CARD.gapY)];
  }
  // Corners in card frame, order = js-aruco2 output order (TL, TR, BR, BL of printed marker)
  function markerCornersCard(id, scale) {
    const i = CARD.ids.indexOf(id);
    if (i < 0) return null;
    const s = scale || 1;
    const [x, y] = markerTopLeftSvg(i), m = CARD.marker;
    const svg = [[x, y], [x + m, y], [x + m, y + m], [x, y + m]];
    return svg.map(([u, v]) => [(u - CARD.w / 2) * s, (CARD.h / 2 - v) * s]);
  }

  // ---------- small linear algebra
  function jacobiEig(A, n) { // symmetric n×n (flat array) -> {vals, vecs (columns, flat)}
    const a = A.slice(), v = new Array(n * n).fill(0);
    for (let i = 0; i < n; i++) v[i * n + i] = 1;
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const th = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
    const vals = []; for (let i = 0; i < n; i++) vals.push(a[i * n + i]);
    return { vals, vecs: v };
  }
  function solve(A, b, n) { // Gaussian elimination with partial pivoting
    const M = A.slice(), x = b.slice();
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
      if (Math.abs(M[p * n + c]) < 1e-15) return null;
      if (p !== c) { for (let k = 0; k < n; k++) [M[c * n + k], M[p * n + k]] = [M[p * n + k], M[c * n + k]]; [x[c], x[p]] = [x[p], x[c]]; }
      for (let r = c + 1; r < n; r++) {
        const f = M[r * n + c] / M[c * n + c];
        if (!f) continue;
        for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k];
        x[r] -= f * x[c];
      }
    }
    for (let r = n - 1; r >= 0; r--) {
      let s = x[r];
      for (let k = r + 1; k < n; k++) s -= M[r * n + k] * x[k];
      x[r] = s / M[r * n + r];
    }
    return x;
  }
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = a => Math.hypot(a[0], a[1], a[2]);
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

  // ---------- homography (DLT, Hartley-normalised). src: [[X,Y]], dst: [[u,v]] -> H (flat 9), maps src->dst
  function normT(pts) {
    let mx = 0, my = 0; pts.forEach(p => { mx += p[0]; my += p[1]; }); mx /= pts.length; my /= pts.length;
    let d = 0; pts.forEach(p => { d += Math.hypot(p[0] - mx, p[1] - my); }); d /= pts.length;
    const s = Math.SQRT2 / (d || 1);
    return { T: [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1], s, mx, my };
  }
  function mul3(A, B) {
    const C = new Array(9).fill(0);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
    return C;
  }
  function inv3(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-300) return null;
    return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
      B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
      C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
  }
  function homography(src, dst) {
    if (src.length < 4) return null;
    const ns = normT(src), nd = normT(dst);
    const M = new Array(81).fill(0);
    for (let k = 0; k < src.length; k++) {
      const X = ns.s * (src[k][0] - ns.mx), Y = ns.s * (src[k][1] - ns.my);
      const u = nd.s * (dst[k][0] - nd.mx), v = nd.s * (dst[k][1] - nd.my);
      const r1 = [-X, -Y, -1, 0, 0, 0, u * X, u * Y, u];
      const r2 = [0, 0, 0, -X, -Y, -1, v * X, v * Y, v];
      for (const r of [r1, r2]) for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) M[i * 9 + j] += r[i] * r[j];
    }
    const { vals, vecs } = jacobiEig(M, 9);
    let mi = 0; for (let i = 1; i < 9; i++) if (vals[i] < vals[mi]) mi = i;
    const Hn = []; for (let i = 0; i < 9; i++) Hn.push(vecs[i * 9 + mi]);
    const H = mul3(mul3(inv3(nd.T), Hn), ns.T);
    const s = H[8] || 1;
    return H.map(x => x / s);
  }
  function applyH(H, p) {
    const w = H[6] * p[0] + H[7] * p[1] + H[8];
    return [(H[0] * p[0] + H[1] * p[1] + H[2]) / w, (H[3] * p[0] + H[4] * p[1] + H[5]) / w];
  }

  // ---------- focal from homographies (square pixels, known principal point). Returns f or null.
  // Each H gives 2 linear eqs in u = 1/f^2 (Zhang with K = diag(f,f,1) after centring).
  function focalFromHomographies(Hs, cx, cy) {
    let num = 0, den = 0;
    for (const H0 of Hs) {
      const H = mul3([1, 0, -cx, 0, 1, -cy, 0, 0, 1], H0);
      const h1 = [H[0], H[3], H[6]], h2 = [H[1], H[4], H[7]];
      const n = Math.max(norm(h1), norm(h2));
      const a = scale(h1, 1 / n), b = scale(h2, 1 / n);
      // eq1: (a0b0 + a1b1) u + a2b2 = 0 ; eq2: (a0²+a1²-b0²-b1²) u + (a2²-b2²) = 0
      const eqs = [[a[0] * b[0] + a[1] * b[1], a[2] * b[2]], [a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2, a[2] ** 2 - b[2] ** 2]];
      for (const [p, q] of eqs) { num += -p * q; den += p * p; }
    }
    if (den < 1e-30) return null;
    const u = num / den;
    if (!(u > 0)) return null;
    return 1 / Math.sqrt(u);
  }
  // Robust variant: only trust the joint estimate when per-frame estimates agree (else focal is unobservable,
  // e.g. small/low-res marker, near-frontal views). Returns {f, spread, n} or null.
  function focalConfident(Hs, cx, cy) {
    const each = Hs.map(H => focalFromHomographies([H], cx, cy)).filter(f => f > 0);
    if (each.length < 5) return null;
    const srt = each.slice().sort((a, b) => a - b), med = srt[srt.length >> 1];
    const mad = each.map(f => Math.abs(f - med)).sort((a, b) => a - b)[each.length >> 1];
    const spread = mad / med;
    if (spread > 0.12) return null;
    const f = focalFromHomographies(Hs, cx, cy);
    return f ? { f, spread, n: each.length } : null;
  }
  // single-H estimate, used for diagnostics / weighting
  function focalFromH(H, cx, cy) { return focalFromHomographies([H], cx, cy); }

  // ---------- rotations
  function rodrigues(r) {
    const th = norm(r);
    if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const k = scale(r, 1 / th), c = Math.cos(th), s = Math.sin(th), C = 1 - c;
    const [x, y, z] = k;
    return [c + x * x * C, x * y * C - z * s, x * z * C + y * s,
      y * x * C + z * s, c + y * y * C, y * z * C - x * s,
      z * x * C - y * s, z * y * C + x * s, c + z * z * C];
  }
  function rotToVec(R) {
    const tr = R[0] + R[4] + R[8];
    let th = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
    if (th < 1e-9) return [0, 0, 0];
    if (Math.PI - th < 1e-6) { // 180°
      const x = Math.sqrt(Math.max(0, (R[0] + 1) / 2)), y = Math.sqrt(Math.max(0, (R[4] + 1) / 2)) * Math.sign(R[1] || 1), z = Math.sqrt(Math.max(0, (R[8] + 1) / 2)) * Math.sign(R[2] || 1);
      return scale([x, y, z], th);
    }
    const k = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
    return scale(k, th / (2 * Math.sin(th)));
  }
  function orthonormalize(R) { // polar decomposition by iteration
    let M = R.slice();
    for (let it = 0; it < 30; it++) {
      const Mi = inv3(M); if (!Mi) break;
      const MiT = [Mi[0], Mi[3], Mi[6], Mi[1], Mi[4], Mi[7], Mi[2], Mi[5], Mi[8]];
      const N = M.map((x, i) => 0.5 * (x + MiT[i]));
      let d = 0; for (let i = 0; i < 9; i++) d += Math.abs(N[i] - M[i]);
      M = N; if (d < 1e-12) break;
    }
    return M;
  }

  // ---------- pose. K = {f, cx, cy}. Card frame -> camera frame: Xc = R X + t (camera: x right, y down, z forward)
  function project(R, t, K, P) {
    const X = R[0] * P[0] + R[1] * P[1] + R[2] * P[2] + t[0];
    const Y = R[3] * P[0] + R[4] * P[1] + R[5] * P[2] + t[1];
    const Z = R[6] * P[0] + R[7] * P[1] + R[8] * P[2] + t[2];
    return [K.f * X / Z + K.cx, K.f * Y / Z + K.cy, Z];
  }
  function poseFromH(H, K) {
    const Ki = [1 / K.f, 0, -K.cx / K.f, 0, 1 / K.f, -K.cy / K.f, 0, 0, 1];
    const B = mul3(Ki, H);
    let b1 = [B[0], B[3], B[6]], b2 = [B[1], B[4], B[7]], b3 = [B[2], B[5], B[8]];
    let lam = 2 / (norm(b1) + norm(b2));
    if (b3[2] * lam < 0) lam = -lam; // card in front of camera
    const r1 = scale(b1, lam), r2 = scale(b2, lam), t = scale(b3, lam), r3 = cross(r1, r2);
    const R = orthonormalize([r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]]);
    return { R, t };
  }
  function refinePose(R0, t0, K, obj, img, iters) {
    let p = rotToVec(R0).concat(t0);
    const resid = q => {
      const R = rodrigues(q.slice(0, 3)), t = q.slice(3), r = [];
      for (let k = 0; k < obj.length; k++) { const u = project(R, t, K, [obj[k][0], obj[k][1], 0]); r.push(u[0] - img[k][0], u[1] - img[k][1]); }
      return r;
    };
    let r = resid(p), cost = r.reduce((s, x) => s + x * x, 0), lam = 1e-3;
    for (let it = 0; it < (iters || 20); it++) {
      const J = [];
      for (let j = 0; j < 6; j++) {
        const e = j < 3 ? 1e-6 : 1e-4, q = p.slice(); q[j] += e;
        const rj = resid(q); J.push(rj.map((x, i) => (x - r[i]) / e));
      }
      const A = new Array(36).fill(0), g = new Array(6).fill(0);
      for (let i = 0; i < 6; i++) { for (let j = 0; j < 6; j++) { let s = 0; for (let k = 0; k < r.length; k++) s += J[i][k] * J[j][k]; A[i * 6 + j] = s; } let s = 0; for (let k = 0; k < r.length; k++) s += J[i][k] * r[k]; g[i] = -s; }
      let improved = false;
      for (let tries = 0; tries < 8; tries++) {
        const Al = A.slice(); for (let i = 0; i < 6; i++) Al[i * 6 + i] *= 1 + lam;
        const d = solve(Al, g, 6); if (!d) { lam *= 10; continue; }
        const q = p.map((x, i) => x + d[i]), rq = resid(q), cq = rq.reduce((s, x) => s + x * x, 0);
        if (cq < cost) { p = q; r = rq; const dc = cost - cq; cost = cq; lam = Math.max(lam / 10, 1e-9); improved = true; if (dc < 1e-10) it = 1e9; break; }
        lam *= 10;
      }
      if (!improved) break;
    }
    return { R: rodrigues(p.slice(0, 3)), t: p.slice(3), rms: Math.sqrt(cost / obj.length) };
  }

  // markers (js-aruco2 output) -> correspondences on card
  function correspondences(markers, cardScale) {
    const obj = [], img = [], ids = [];
    for (const m of markers) {
      const c = markerCornersCard(m.id, cardScale);
      if (!c) continue;
      ids.push(m.id);
      for (let k = 0; k < 4; k++) { obj.push(c[k]); img.push([m.corners[k].x, m.corners[k].y]); }
    }
    return { obj, img, ids };
  }

  function estimatePose(markers, K, cardScale) {
    // keep one detection per id (the one with lowest Hamming distance)
    const best = {};
    for (const m of markers) if (CARD.ids.includes(m.id) && (!best[m.id] || (m.hammingDistance || 0) < (best[m.id].hammingDistance || 0))) best[m.id] = m;
    let use = Object.values(best);
    if (use.length < 1) return null;
    let p, H, obj, img, ids;
    for (let pass = 0; pass < 2; pass++) {
      ({ obj, img, ids } = correspondences(use, cardScale));
      H = homography(obj, img);
      if (!H) return null;
      const p0 = poseFromH(H, K);
      p = refinePose(p0.R, p0.t, K, obj, img, 25);
      if (use.length < 3) break;
      // per-marker reprojection error -> drop outlier markers (mis-decoded / occluded corners)
      const errs = use.map((m, j) => {
        let e = 0; for (let k = 0; k < 4; k++) { const u = project(p.R, p.t, K, [obj[4 * j + k][0], obj[4 * j + k][1], 0]); e += Math.hypot(u[0] - img[4 * j + k][0], u[1] - img[4 * j + k][1]); }
        return e / 4;
      });
      const med = errs.slice().sort((a, b) => a - b)[errs.length >> 1];
      const keep = use.filter((m, j) => errs[j] <= Math.max(3, 4 * med));
      if (keep.length === use.length) break;
      use = keep;
    }
    // planar two-fold ambiguity: refine the mirrored solution too, keep the better one, remember the other
    const a0 = altPose(p.R, p.t), q = refinePose(a0.R, a0.t, K, obj, img, 30);
    let A = describe(p.R, p.t, p.rms), B = describe(q.R, q.t, q.rms);
    if (B.rms < A.rms) [A, B] = [B, A];
    const same = angleBetween(scale(A.C, 1 / A.dist), scale(B.C, 1 / B.dist)) < 3;
    const out = Object.assign(A, { H, nMarkers: ids.length, ids, obj, img, ambiguity: same ? 99 : B.rms / Math.max(A.rms, 1e-9), alt: same ? null : B });
    return out;
  }
  function describe(R, t, rms) {
    // camera centre in card frame: C = -R^T t
    const C = [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
    const dist = norm(C);
    const elev = Math.atan2(C[2], Math.hypot(C[0], C[1])) * 180 / Math.PI;
    const azim = (Math.atan2(C[1], C[0]) * 180 / Math.PI + 360) % 360;
    return { R, t, C, dist, elev, azim, rms };
  }
  // mirrored planar pose: reflect the plane normal about the viewing ray to the card origin
  function altPose(R, t) {
    const n = [R[2], R[5], R[8]], v = scale(t, 1 / norm(t)), nv = dot(n, v);
    const n2 = v.map((x, i) => 2 * nv * x - n[i]);
    const ax = cross(n, n2), s = norm(ax), ang = Math.atan2(s, dot(n, n2));
    const Q = rodrigues(s < 1e-12 ? [0, 0, 0] : scale(ax, ang / s));
    return { R: mul3(Q, R), t: t.slice() };
  }
  // swap a pose to its stored alternative
  function swapAlt(p) {
    if (!p.alt) return p;
    const a = p.alt, rest = { H: p.H, nMarkers: p.nMarkers, ids: p.ids, obj: p.obj, img: p.img, ambiguity: p.ambiguity };
    return Object.assign({}, a, rest, { alt: describe(p.R, p.t, p.rms), flipped: !p.flipped });
  }
  // pick, for each pose in temporal order, the solution consistent with its neighbours.
  // Seeds from the least ambiguous pose and propagates both ways by nearest viewing direction.
  function resolveSequence(poses) {
    const n = poses.length; if (!n) return poses;
    const out = poses.slice();
    let seed = 0; for (let i = 1; i < n; i++) if (out[i].ambiguity > out[seed].ambiguity) seed = i;
    const pick = (prev, p) => {
      if (!p.alt) return p;
      const dv = q => angleBetween(scale(q.C, 1 / q.dist), scale(prev.C, 1 / prev.dist));
      return dv(p.alt) < dv(p) ? swapAlt(p) : p;
    };
    for (let i = seed + 1; i < n; i++) out[i] = pick(out[i - 1], out[i]);
    for (let i = seed - 1; i >= 0; i--) out[i] = pick(out[i + 1], out[i]);
    return out;
  }

  // ---------- coverage / keyframe policy
  // viewing direction unit vector from card centre to camera
  const viewDir = pose => scale(pose.C, 1 / pose.dist);
  function angleBetween(a, b) { return Math.acos(Math.max(-1, Math.min(1, dot(a, b)))) * 180 / Math.PI; }

  // sharpness: variance of Laplacian on grey (Uint8 or Float array), w×h, sampled in ROI
  function sharpness(grey, w, h, roi) {
    const x0 = Math.max(1, roi ? roi.x0 | 0 : 1), x1 = Math.min(w - 2, roi ? roi.x1 | 0 : w - 2);
    const y0 = Math.max(1, roi ? roi.y0 | 0 : 1), y1 = Math.min(h - 2, roi ? roi.y1 | 0 : h - 2);
    let s = 0, s2 = 0, n = 0;
    for (let y = y0; y <= y1; y += 1) for (let x = x0; x <= x1; x += 1) {
      const i = y * w + x;
      const L = grey[i - 1] + grey[i + 1] + grey[i - w] + grey[i + w] - 4 * grey[i];
      s += L; s2 += L * L; n++;
    }
    if (!n) return 0;
    const m = s / n; return s2 / n - m * m;
  }

  const api = { focalConfident, altPose, swapAlt, resolveSequence, describe, CARD, markerCornersCard, markerTopLeftSvg, homography, applyH, focalFromHomographies, focalFromH, poseFromH, refinePose, estimatePose, correspondences, project, rodrigues, rotToVec, viewDir, angleBetween, sharpness, jacobiEig, inv3, mul3 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FoodCore = api;
})(typeof window !== 'undefined' ? window : this);
