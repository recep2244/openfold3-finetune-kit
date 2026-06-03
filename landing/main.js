/* openfold3-finetune-kit landing — interactions
   1) molecular-network canvas in the hero
   2) scroll-reveal for .reveal elements
   Both respect prefers-reduced-motion. */
(() => {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- scroll reveal ---------- */
  const revealables = document.querySelectorAll(".reveal");
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

  /* ---------- hero molecular network ---------- */
  const canvas = document.querySelector(".hero__canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let w, h, dpr, nodes, raf;
  const TEAL = "rgba(94,234,212,";

  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    w = r.width;
    h = r.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function seedNodes() {
    const count = Math.max(26, Math.min(64, Math.round((w * h) / 26000)));
    nodes = Array.from({ length: count }, (_, i) => ({
      x: ((i * 97.13) % w),
      y: ((i * 53.71) % h),
      // deterministic-ish drift vectors (no Math.random dependency for repeatability)
      vx: (((i * 13) % 7) - 3) * 0.04,
      vy: (((i * 17) % 7) - 3) * 0.04,
      r: 1.3 + ((i * 7) % 5) * 0.35,
    }));
  }

  function step() {
    ctx.clearRect(0, 0, w, h);
    for (const n of nodes) {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;
    }
    // edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 17000) {
          const o = (1 - d2 / 17000) * 0.5;
          ctx.strokeStyle = TEAL + o.toFixed(3) + ")";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    // nodes
    for (const n of nodes) {
      ctx.fillStyle = TEAL + "0.85)";
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
    }
    raf = requestAnimationFrame(step);
  }

  function drawStatic() {
    ctx.clearRect(0, 0, w, h);
    for (const n of nodes) {
      ctx.fillStyle = TEAL + "0.7)";
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function init() {
    size();
    seedNodes();
    if (reduce) {
      drawStatic();
    } else {
      cancelAnimationFrame(raf);
      step();
    }
  }

  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(init, 180);
  });
  // pause when the tab is hidden
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cancelAnimationFrame(raf);
    else if (!reduce) step();
  });
  init();
})();
