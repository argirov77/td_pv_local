(function () {
    const slides = Array.from(document.querySelectorAll('.slide'));
    const prevBtn = document.getElementById('prev');
    const nextBtn = document.getElementById('next');
    const breadcrumbs = document.getElementById('breadcrumbs');
    let current = 0;

    function updateBreadcrumbs() {
        breadcrumbs.innerHTML = '';
        slides.forEach((slide, index) => {
            const crumb = document.createElement('div');
            crumb.className = 'crumb' + (index === current ? ' active' : '');
            const dot = document.createElement('span');
            dot.className = 'dot';
            const label = document.createElement('span');
            label.textContent = slide.dataset.title || `Слайд ${index + 1}`;
            crumb.appendChild(dot);
            crumb.appendChild(label);
            crumb.addEventListener('click', () => goTo(index));
            breadcrumbs.appendChild(crumb);
        });
    }

    function updateNav() {
        prevBtn.disabled = current === 0;
        nextBtn.disabled = current === slides.length - 1;
    }

    function goTo(index) {
        const nextIndex = Math.max(0, Math.min(index, slides.length - 1));
        if (nextIndex === current) return;
        slides[current].style.display = 'none';
        current = nextIndex;
        slides[current].style.display = '';
        updateBreadcrumbs();
        updateNav();
    }

    function next() { goTo(current + 1); }
    function prev() { goTo(current - 1); }

    slides.forEach((slide, idx) => {
        if (idx !== 0) slide.style.display = 'none';
    });

    prevBtn.addEventListener('click', prev);
    nextBtn.addEventListener('click', next);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowRight') next();
        if (event.key === 'ArrowLeft') prev();
    });

    updateBreadcrumbs();
    updateNav();
})();
