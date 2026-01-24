// docs/main.js

document.addEventListener("DOMContentLoaded", () => {
    const nav = document.getElementById("nav");
    const navToggle = document.getElementById("navToggle");

    // Mobile nav toggle
    if (navToggle && nav) {
      navToggle.addEventListener("click", () => {
        const isOpen = nav.classList.toggle("open");
        navToggle.setAttribute("aria-expanded", String(isOpen));
      });

      // Close nav on link click (mobile)
      nav.querySelectorAll("a").forEach((a) => {
        a.addEventListener("click", () => {
          if (nav.classList.contains("open")) {
            nav.classList.remove("open");
            navToggle.setAttribute("aria-expanded", "false");
          }
        });
      });
    }

    // Active section highlight
    const links = Array.from(document.querySelectorAll(".nav a"));
    const sections = links
      .map((a) => document.querySelector(a.getAttribute("href")))
      .filter(Boolean);

    if ("IntersectionObserver" in window && sections.length) {
      const obs = new IntersectionObserver(
        (entries) => {
          // pick the most visible intersecting entry
          const visible = entries
            .filter((e) => e.isIntersecting)
            .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

          if (!visible) return;

          const id = "#" + visible.target.id;
          links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === id));
        },
        { threshold: [0.2, 0.35, 0.5, 0.65] }
      );

      sections.forEach((s) => obs.observe(s));
    }

    // Copy BibTeX
    const copyBtn = document.getElementById("copyBibtex");
    const bibtexBlock = document.getElementById("bibtexBlock");
    const status = document.getElementById("copyStatus");

    if (copyBtn && bibtexBlock) {
      copyBtn.addEventListener("click", async () => {
        const text = bibtexBlock.innerText.trim();
        try {
          await navigator.clipboard.writeText(text);
          if (status) status.textContent = "Copied!";
          copyBtn.textContent = "Copied";
          setTimeout(() => {
            copyBtn.textContent = "Copy BibTeX";
            if (status) status.textContent = "";
          }, 1200);
        } catch (e) {
          // fallback: select text
          const range = document.createRange();
          range.selectNodeContents(bibtexBlock);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          if (status) status.textContent = "Select + copy (Ctrl/Cmd+C).";
        }
      });
    }
  });
