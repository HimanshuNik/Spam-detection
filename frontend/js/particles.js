/**
 * SpamShield AI — Three.js Particle Background
 * ===============================================
 * Creates a subtle, animated particle field that responds
 * to mouse movement. Non-distracting, color-coordinated.
 */

(function () {
    // Skip if Three.js failed to load
    if (typeof THREE === "undefined") {
        console.warn("[particles] Three.js not loaded — skipping particle background.");
        return;
    }

    const container = document.getElementById("particles-container");
    if (!container) return;

    // ---- Configuration ----
    const PARTICLE_COUNT = 180;
    const FIELD_SIZE = 50;
    const PARTICLE_SIZE = 2.5;
    const MOUSE_INFLUENCE = 0.00015;
    const BASE_ROTATION_SPEED = 0.0001;

    // ---- Scene Setup ----
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.z = 40;

    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false,
        powerPreference: "low-power",
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    container.appendChild(renderer.domElement);

    // ---- Particle Geometry ----
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const colors = new Float32Array(PARTICLE_COUNT * 3);
    const sizes = new Float32Array(PARTICLE_COUNT);

    // Accent palette (HSL → RGB approximations)
    const palette = [
        { r: 0.545, g: 0.361, b: 0.965 }, // Purple #8b5cf6
        { r: 0.231, g: 0.510, b: 0.965 }, // Blue   #3b82f6
        { r: 0.024, g: 0.714, b: 0.831 }, // Cyan   #06b6d4
    ];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const i3 = i * 3;

        // Random position within cube
        positions[i3]     = (Math.random() - 0.5) * FIELD_SIZE;
        positions[i3 + 1] = (Math.random() - 0.5) * FIELD_SIZE;
        positions[i3 + 2] = (Math.random() - 0.5) * FIELD_SIZE;

        // Random color from palette
        const c = palette[Math.floor(Math.random() * palette.length)];
        colors[i3]     = c.r;
        colors[i3 + 1] = c.g;
        colors[i3 + 2] = c.b;

        // Random size
        sizes[i] = Math.random() * PARTICLE_SIZE + 0.5;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    // ---- Particle Material ----
    const material = new THREE.PointsMaterial({
        size: PARTICLE_SIZE,
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
    });

    const particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);

    // ---- Mouse Tracking ----
    const mouse = { x: 0, y: 0 };

    document.addEventListener("mousemove", (e) => {
        mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    // ---- Resize Handler ----
    function onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    window.addEventListener("resize", onResize);

    // ---- Respect Reduced Motion ----
    const prefersReducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

    // ---- Theme-Aware Opacity ----
    function updateOpacity() {
        const isLight = document.documentElement.getAttribute("data-theme") === "light";
        material.opacity = isLight ? 0.25 : 0.5;
    }

    // Watch for theme changes
    const observer = new MutationObserver(updateOpacity);
    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
    });
    updateOpacity();

    // ---- Animation Loop ----
    function animate() {
        requestAnimationFrame(animate);

        if (!prefersReducedMotion) {
            // Gentle base rotation
            particleSystem.rotation.y += BASE_ROTATION_SPEED;
            particleSystem.rotation.x += BASE_ROTATION_SPEED * 0.5;

            // Mouse-influenced rotation
            particleSystem.rotation.y += mouse.x * MOUSE_INFLUENCE;
            particleSystem.rotation.x += mouse.y * MOUSE_INFLUENCE;
        }

        renderer.render(scene, camera);
    }

    animate();
})();
