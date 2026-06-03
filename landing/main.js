/* openfold3-finetune-kit landing — interactions
   1) hero canvas: a drifting protein-backbone trace with residue nodes and
      faint long-range "contacts" (domain-true, not a generic particle field)
   2) count-up hero stats
   3) scroll-reveal for below-the-fold sections
   All respect prefers-reduced-motion. */
(() => {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- scroll reveal ---------- */
  const revealables = document.querySelectorAll(".band .reveal, .cta .reveal");
  if (reduce || !("IntersectionObserver" in window)) {
    revealables.forEach((el) => el.classList.add("in"));
  } else {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("in");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.16 }
    );
    revealables.forEach((el) => io.observe(el));
  }

  /* ---------- count-up hero stats ---------- */
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || "";
    const prefix = el.dataset.prefix || "";
    if (reduce) {
      el.textContent = prefix + target + suffix;
      return;
    }
    const dur = 1100;
    const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = prefix + Math.round(target * eased) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  document.querySelectorAll("[data-count]").forEach(countUp);

  /* ---------- hero backbone trace ---------- */
  const canvas = document.querySelector(".hero__canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const TEAL = (a) => `rgba(94,234,212,${a})`;
  let w, h, dpr, pts, raf, t0;

  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    w = r.width;
    h = r.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function seed() {
    const n = Math.max(14, Math.min(30, Math.round(w / 60)));
    pts = Array.from({ length: n }, (_, i) => ({
      bx: (i / (n - 1)) * (w * 1.1) - w * 0.05,
      by: h * (0.35 + 0.3 * Math.sin(i * 0.9)),
      amp: 18 + (i % 5) * 9,
      ph: i * 0.7,
      sp: 0.4 + (i % 4) * 0.12,
    }));
  }

  function pos(p, time) {
    return { x: p.bx, y: p.by + Math.sin(time * p.sp + p.ph) * p.amp };
  }

  function frame(now) {
    const time = (now - t0) / 1000;
    ctx.clearRect(0, 0, w, h);
    const P = pts.map((p) => pos(p, time));

    // long-range contacts (sequence-distant, space-near) — faint chords
    for (let i = 0; i < P.length; i++) {
      for (let j = i + 3; j < P.length; j++) {
        const dx = P[i].x - P[j].x, dy = P[i].y - P[j].y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 12000) {
          ctx.strokeStyle = TEAL((1 - d2 / 12000) * 0.18);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(P[i].x, P[i].y);
          ctx.lineTo(P[j].x, P[j].y);
          ctx.stroke();
        }
      }
    }
    // backbone ribbon (smooth Catmull-Rom-ish via quadratic midpoints)
    ctx.strokeStyle = TEAL(0.5);
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(P[0].x, P[0].y);
    for (let i = 1; i < P.length - 1; i++) {
      const mx = (P[i].x + P[i + 1].x) / 2;
      const my = (P[i].y + P[i + 1].y) / 2;
      ctx.quadraticCurveTo(P[i].x, P[i].y, mx, my);
    }
    ctx.stroke();
    // residue nodes
    for (const p of P) {
      ctx.fillStyle = TEAL(0.85);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
    raf = requestAnimationFrame(frame);
  }

  function staticDraw() {
    ctx.clearRect(0, 0, w, h);
    const P = pts.map((p) => pos(p, 0));
    ctx.strokeStyle = TEAL(0.45);
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(P[0].x, P[0].y);
    for (let i = 1; i < P.length - 1; i++) {
      const mx = (P[i].x + P[i + 1].x) / 2, my = (P[i].y + P[i + 1].y) / 2;
      ctx.quadraticCurveTo(P[i].x, P[i].y, mx, my);
    }
    ctx.stroke();
    for (const p of P) {
      ctx.fillStyle = TEAL(0.8);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function init() {
    size();
    seed();
    cancelAnimationFrame(raf);
    if (reduce) {
      staticDraw();
    } else {
      t0 = performance.now();
      raf = requestAnimationFrame(frame);
    }
  }

  let tmr;
  window.addEventListener("resize", () => {
    clearTimeout(tmr);
    tmr = setTimeout(init, 180);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cancelAnimationFrame(raf);
    else if (!reduce) { t0 = performance.now(); raf = requestAnimationFrame(frame); }
  });
  init();
})();
