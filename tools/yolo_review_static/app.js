(() => {
  "use strict";

  const data = window.YOLO_REVIEW_DATA;
  if (!data) {
    document.body.innerHTML = "<p class='notice'>Missing data.js. Rebuild the gallery.</p>";
    return;
  }

  const PAGE_SIZE = 36;
  const taskNames = ["presence", "awake", "pacifier"];
  const elements = {
    gallery: document.querySelector("#gallery"),
    pagination: document.querySelector("#pagination"),
    count: document.querySelector("#result-count"),
    location: document.querySelector("#location-filter"),
    outcome: document.querySelector("#outcome-filter"),
    presence: document.querySelector("#presence-filter"),
    awake: document.querySelector("#awake-filter"),
    pacifier: document.querySelector("#pacifier-filter"),
    sort: document.querySelector("#sort-filter"),
    search: document.querySelector("#search-filter"),
    reset: document.querySelector("#reset-filters"),
    lightbox: document.querySelector("#lightbox"),
    lightboxImage: document.querySelector("#lightbox-image"),
    lightboxCaption: document.querySelector("#lightbox-caption"),
    lightboxClose: document.querySelector("#lightbox-close"),
  };
  let page = 1;

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const percentage = (value, digits = 1) =>
    value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;

  const friendlyTime = (value) => {
    const date = new Date(value);
    return new Intl.DateTimeFormat("en-GB", {
      dateStyle: "medium",
      timeStyle: "medium",
      timeZone: "Europe/Madrid",
    }).format(date);
  };

  const decisionLabel = (value) =>
    ({
      present: "present",
      absent: "absent",
      awake: "awake",
      asleep: "asleep",
      yes: "yes",
      no: "no",
      unknown: "abstained",
      not_run: "not run",
    })[value] || value;

  const badge = (outcome) =>
    `<span class="badge badge--${escapeHtml(outcome)}">${escapeHtml(outcome.replace("_", " "))}</span>`;

  function renderTop() {
    const summary = data.summary;
    document.querySelector("#meta-line").innerHTML = [
      data.meta.artifact_version,
      `${summary.frames.toLocaleString()} ${data.meta.split} frames`,
      Object.keys(summary.locations).join(" + "),
      `generated ${friendlyTime(data.meta.generated_at)}`,
    ]
      .map((item) => `<span>${escapeHtml(item)}</span>`)
      .join("");
    document.querySelector("#reference-note").textContent = data.meta.reference_note;
    document.querySelector("#summary").innerHTML = [
      ["Frames", summary.frames.toLocaleString(), ""],
      ["Matches", summary.overall.match.toLocaleString(), ""],
      ["Mismatches", summary.overall.mismatch.toLocaleString(), "danger"],
      ["Abstentions", summary.overall.abstain.toLocaleString(), "warning"],
      [
        "Head localized",
        percentage(summary.detail_available / summary.frames, 0),
        "",
      ],
    ]
      .map(
        ([label, value, kind]) => `
          <article class="metric ${kind ? `metric--${kind}` : ""}">
            <span class="metric__value">${value}</span>
            <span class="metric__label">${label}</span>
          </article>`,
      )
      .join("");
    document.querySelector("#task-table").innerHTML = `
      <table class="task-table">
        <thead><tr>
          <th>Feature</th><th>Labeled frames</th><th>Coverage</th>
          <th>Agreement when decisive</th><th>Mismatches</th><th>Abstentions</th>
        </tr></thead>
        <tbody>
          ${taskNames
            .map((task) => {
              const item = summary.tasks[task];
              return `<tr>
                <td>${task}</td>
                <td>${item.labeled.toLocaleString()}</td>
                <td>${percentage(item.coverage)}</td>
                <td>${percentage(item.agreement)}</td>
                <td>${item.mismatches.toLocaleString()}</td>
                <td>${item.abstentions.toLocaleString()}</td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>`;
    Object.keys(summary.locations).forEach((location) => {
      elements.location.insertAdjacentHTML(
        "beforeend",
        `<option value="${escapeHtml(location)}">${escapeHtml(location)} · ${summary.locations[location]}</option>`,
      );
    });
  }

  function scoreRow(task, item) {
    const score = item.score;
    const [negative, positive] = item.thresholds;
    const fill = score == null ? 0 : Math.max(0, Math.min(100, score * 100));
    return `
      <div class="score-row">
        <div class="score-row__top">
          <span class="score-row__task">${escapeHtml(task)}</span>
          <span class="score-row__value">${score == null ? "not run" : percentage(score)} · ${escapeHtml(decisionLabel(item.decision))}</span>
        </div>
        <div class="score-bar" title="negative ≤ ${percentage(negative)} · positive ≥ ${percentage(positive)}">
          <span class="score-bar__fill" style="width:${fill}%"></span>
          <span class="score-bar__abstain" style="left:${negative * 100}%;width:${(positive - negative) * 100}%"></span>
          <span class="score-bar__marker" style="left:${negative * 100}%"></span>
          <span class="score-bar__marker" style="left:${positive * 100}%"></span>
        </div>
        <div class="teacher-row">
          <span>Teacher <strong>${escapeHtml(item.reference ?? "not labeled")}</strong></span>
          ${badge(item.outcome)}
        </div>
      </div>`;
  }

  function visualButton(src, label, className = "") {
    if (!src) {
      return `<div class="missing-head"><span>Pose could not produce a decisive head crop</span></div>`;
    }
    return `
      <button class="visual-button ${className}" type="button" data-lightbox="${escapeHtml(src)}" data-caption="${escapeHtml(label)}">
        <img src="${escapeHtml(src)}" alt="${escapeHtml(label)}" loading="lazy" />
        <span class="visual-label">${escapeHtml(label)}</span>
      </button>`;
  }

  function card(frame) {
    const prediction = frame.prediction;
    const searchableId = frame.frame_id.slice(0, 8);
    return `
      <article class="card card--${escapeHtml(frame.overall_outcome)}">
        <div class="visuals">
          ${visualButton(frame.images.frame, "Full frame + ROIs", "visual-button--frame")}
          ${visualButton(frame.images.roi, `Selected ROI · ${frame.winner_roi}`, "visual-button--small")}
          ${visualButton(frame.images.head, "Pose head input", "visual-button--small")}
        </div>
        <div class="card__body">
          <div class="card__header">
            <div>
              <p class="card__time">${escapeHtml(friendlyTime(frame.captured_at))}</p>
              <p class="card__meta">${escapeHtml(frame.location_id)} · ${escapeHtml(searchableId)} · ${escapeHtml(frame.winner_roi)}</p>
            </div>
            ${badge(frame.overall_outcome)}
          </div>
          <div class="prediction-strip">
            <div class="prediction"><span>Presence</span><strong>${escapeHtml(decisionLabel(frame.tasks.presence.decision))}</strong></div>
            <div class="prediction"><span>State</span><strong>${escapeHtml(decisionLabel(frame.tasks.awake.decision))}</strong></div>
            <div class="prediction"><span>Pacifier</span><strong>${escapeHtml(decisionLabel(frame.tasks.pacifier.decision))}</strong></div>
          </div>
          <div class="score-list">${taskNames.map((task) => scoreRow(task, frame.tasks[task])).join("")}</div>
          <div class="roi-list">
            ${frame.roi_profiles
              .map(
                (roi) => `<span class="roi-pill ${roi.selected ? "roi-pill--winner" : ""}">
                  ${escapeHtml(roi.name)} · ${percentage(roi.score)}
                </span>`,
              )
              .join("")}
            <span class="roi-pill">runtime confidence · ${percentage(prediction.confidence)}</span>
          </div>
        </div>
      </article>`;
  }

  function currentFrames() {
    const search = elements.search.value.trim().toLowerCase();
    const fields = {
      location_id: elements.location.value,
      overall_outcome: elements.outcome.value,
    };
    let frames = data.frames.filter((frame) => {
      if (fields.location_id !== "all" && frame.location_id !== fields.location_id) return false;
      if (fields.overall_outcome !== "all" && frame.overall_outcome !== fields.overall_outcome) return false;
      if (elements.presence.value !== "all" && frame.tasks.presence.decision !== elements.presence.value) return false;
      if (elements.awake.value !== "all" && frame.tasks.awake.decision !== elements.awake.value) return false;
      if (elements.pacifier.value !== "all" && frame.tasks.pacifier.decision !== elements.pacifier.value) return false;
      if (search) {
        const haystack = [
          frame.frame_id,
          frame.captured_at,
          frame.location_id,
          frame.winner_roi,
          frame.winner_surface,
          frame.relative_path,
        ]
          .join(" ")
          .toLowerCase();
        if (!haystack.includes(search)) return false;
      }
      return true;
    });
    const sort = elements.sort.value;
    frames = [...frames].sort((a, b) => {
      if (sort === "newest") return b.captured_at.localeCompare(a.captured_at);
      if (sort === "oldest") return a.captured_at.localeCompare(b.captured_at);
      if (sort === "confidence") return a.prediction.confidence - b.prediction.confidence;
      return a.review_priority - b.review_priority || a.captured_at.localeCompare(b.captured_at);
    });
    return frames;
  }

  function render() {
    const frames = currentFrames();
    const pageCount = Math.max(1, Math.ceil(frames.length / PAGE_SIZE));
    page = Math.min(page, pageCount);
    const start = (page - 1) * PAGE_SIZE;
    const visible = frames.slice(start, start + PAGE_SIZE);
    elements.count.textContent = `${frames.length.toLocaleString()} frames · showing ${frames.length ? start + 1 : 0}–${Math.min(start + PAGE_SIZE, frames.length)}`;
    elements.gallery.innerHTML = visible.length
      ? visible.map(card).join("")
      : "<div class='notice'><strong>No frames match these filters.</strong></div>";
    elements.pagination.innerHTML = `
      <button type="button" data-page="${page - 1}" ${page === 1 ? "disabled" : ""}>←</button>
      <span>Page ${page} of ${pageCount}</span>
      <button type="button" data-page="${page + 1}" ${page === pageCount ? "disabled" : ""}>→</button>`;
  }

  function resetFilters() {
    [elements.location, elements.outcome, elements.presence, elements.awake, elements.pacifier].forEach(
      (element) => {
        element.value = "all";
      },
    );
    elements.sort.value = "review";
    elements.search.value = "";
    page = 1;
    render();
  }

  renderTop();
  render();
  [elements.location, elements.outcome, elements.presence, elements.awake, elements.pacifier, elements.sort].forEach(
    (element) =>
      element.addEventListener("change", () => {
        page = 1;
        render();
      }),
  );
  elements.search.addEventListener("input", () => {
    page = 1;
    render();
  });
  elements.reset.addEventListener("click", resetFilters);
  elements.pagination.addEventListener("click", (event) => {
    const button = event.target.closest("[data-page]");
    if (!button || button.disabled) return;
    page = Number(button.dataset.page);
    render();
    document.querySelector(".controls").scrollIntoView({ behavior: "smooth" });
  });
  elements.gallery.addEventListener("click", (event) => {
    const button = event.target.closest("[data-lightbox]");
    if (!button) return;
    elements.lightboxImage.src = button.dataset.lightbox;
    elements.lightboxImage.alt = button.dataset.caption;
    elements.lightboxCaption.textContent = button.dataset.caption;
    elements.lightbox.showModal();
  });
  elements.lightboxClose.addEventListener("click", () => elements.lightbox.close());
  elements.lightbox.addEventListener("click", (event) => {
    if (event.target === elements.lightbox) elements.lightbox.close();
  });
})();
