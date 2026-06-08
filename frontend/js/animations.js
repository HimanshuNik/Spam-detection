/**
 * SpamShield AI — Animations
 * ============================
 * Typing effect in hero section, scroll-reveal for all sections,
 * and animated stat counters when they enter the viewport.
 */

(function () {
    // ================================================================
    // 1. Typing Animation
    // ================================================================
    const phrases = [
        "Spam Instantly",
        "Phishing Attempts",
        "Fraudulent Content",
        "Scam Messages",
        "Malicious Links",
    ];

    const typingEl = document.getElementById("typing-text");
    if (typingEl) {
        let phraseIdx = 0;
        let charIdx = 0;
        let isDeleting = false;
        const TYPING_SPEED = 80;
        const DELETING_SPEED = 45;
        const PAUSE_AFTER_TYPING = 2200;
        const PAUSE_AFTER_DELETING = 400;

        function typeStep() {
            const currentPhrase = phrases[phraseIdx];

            if (!isDeleting) {
                // Typing
                typingEl.textContent = currentPhrase.substring(0, charIdx + 1);
                charIdx++;

                if (charIdx === currentPhrase.length) {
                    // Pause, then start deleting
                    isDeleting = true;
                    setTimeout(typeStep, PAUSE_AFTER_TYPING);
                    return;
                }
                setTimeout(typeStep, TYPING_SPEED);
            } else {
                // Deleting
                typingEl.textContent = currentPhrase.substring(0, charIdx - 1);
                charIdx--;

                if (charIdx === 0) {
                    isDeleting = false;
                    phraseIdx = (phraseIdx + 1) % phrases.length;
                    setTimeout(typeStep, PAUSE_AFTER_DELETING);
                    return;
                }
                setTimeout(typeStep, DELETING_SPEED);
            }
        }

        // Start after a short delay for page load
        setTimeout(typeStep, 1200);
    }

    // ================================================================
    // 2. Scroll Reveal  (IntersectionObserver-based)
    // ================================================================
    const revealElements = document.querySelectorAll(".reveal");

    if ("IntersectionObserver" in window) {
        const revealObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("visible");
                        // Unobserve after first reveal (one-shot)
                        revealObserver.unobserve(entry.target);
                    }
                });
            },
            {
                threshold: 0.15,
                rootMargin: "0px 0px -40px 0px",
            }
        );

        revealElements.forEach((el) => {
            // Skip dynamically shown panels (e.g. result card starts display:none)
            if (el.id === "result-card") return;
            revealObserver.observe(el);
        });
    } else {
        // Fallback: just show everything
        revealElements.forEach((el) => el.classList.add("visible"));
    }

    // ================================================================
    // 3. Staggered Reveal for Stat Cards
    // ================================================================
    const statCards = document.querySelectorAll(".stat-card");

    if (statCards.length > 0 && "IntersectionObserver" in window) {
        const statsObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        // Stagger each card's reveal
                        statCards.forEach((card, idx) => {
                            card.style.opacity = "0";
                            card.style.transform = "translateY(30px)";
                            card.style.transition = `opacity 0.6s cubic-bezier(0.22,1,0.36,1) ${idx * 0.12}s, transform 0.6s cubic-bezier(0.22,1,0.36,1) ${idx * 0.12}s`;

                            requestAnimationFrame(() => {
                                card.style.opacity = "1";
                                card.style.transform = "translateY(0)";
                            });
                        });
                        statsObserver.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.2 }
        );

        // Observe the grid wrapper
        const statsGrid = document.querySelector(".stats-grid");
        if (statsGrid) statsObserver.observe(statsGrid);
    }
})();
