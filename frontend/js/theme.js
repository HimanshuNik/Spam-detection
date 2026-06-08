/**
 * SpamShield AI — Theme Toggle
 * ===============================
 * Dark/light mode with localStorage persistence
 * and prefers-color-scheme detection on first visit.
 */

(function () {
    const STORAGE_KEY = "spamshield-theme";
    const html = document.documentElement;
    const toggle = document.getElementById("theme-toggle");

    /**
     * Apply a theme and persist the choice.
     * @param {"dark"|"light"} theme
     */
    function setTheme(theme) {
        html.setAttribute("data-theme", theme);
        localStorage.setItem(STORAGE_KEY, theme);
    }

    /**
     * Determine the initial theme:
     *  1. Saved preference in localStorage
     *  2. OS-level prefers-color-scheme
     *  3. Default to dark
     */
    function getInitialTheme() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved === "dark" || saved === "light") return saved;

        if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
            return "light";
        }
        return "dark";
    }

    // Apply on load (runs before DOMContentLoaded because this script is in <body>)
    setTheme(getInitialTheme());

    // Toggle handler
    if (toggle) {
        toggle.addEventListener("click", () => {
            const current = html.getAttribute("data-theme");
            setTheme(current === "dark" ? "light" : "dark");
        });
    }
})();
